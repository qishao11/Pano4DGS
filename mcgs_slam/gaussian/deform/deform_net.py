"""Time-conditioned deformation field for 4D Gaussian Splatting (4DGS).

Design (see panoramic_support_feasibility.md section 4): the Gaussian set stays a single
*canonical* GaussianModel (unchanged). Before each render() call, this network maps
(canonical xyz, timestamp) -> a small per-Gaussian correction (Δxyz, Δrotation, Δscale)
that is applied on the fly and discarded -- nothing about a Gaussian's time-varying state is
stored persistently, so densify/prune/reset and the global pose/scale correction in
gs_backend.py::process_global_track_data() (which writes directly into the canonical
_xyz/_scaling/_rotation tensors) need no changes at all.

The final linear layer is zero-initialized, so a freshly-constructed DeformNet is an exact
no-op (Δ≡0) until trained -- this lets the P4a "Step 1" parity check (wire it in, confirm
output is bit-for-bit identical to the no-deform baseline) hold trivially.
"""
import torch
from torch import nn


def fourier_encode(x, num_freqs):
    """x: (...,C) -> (..., C*(1+2*num_freqs)) with sin/cos bands at frequencies 2^0..2^(F-1)."""
    if num_freqs == 0:
        return x
    freqs = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)) * torch.pi
    # (..., C, F)
    xf = x.unsqueeze(-1) * freqs
    enc = torch.cat([torch.sin(xf), torch.cos(xf)], dim=-1)  # (..., C, 2F)
    enc = enc.flatten(-2)  # (..., C*2F)
    return torch.cat([x, enc], dim=-1)


def quat_multiply(q1, q2):
    """Hamilton product, both (...,4) in (w,x,y,z) order -- matches this codebase's Gaussian
    rotation convention (see gaussian/utils/general_utils.py::build_rotation, which reads
    q[:,0] as the real part). NOT the same order as cubemap.py's camera-extrinsics quaternions
    (those are (x,y,z,w), an unrelated convention for a different consumer)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def axis_angle_to_quat(aa):
    """Small-vector axis-angle (...,3) -> unit quaternion (...,4) in (w,x,y,z) order.
    At aa=0 this returns the identity quaternion exactly, so a zero-initialized DeformNet
    composes to an identity rotation delta."""
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True)
    half = theta * 0.5
    # sin(half)/theta is well-defined (-> 0.5) as theta -> 0; avoid 0/0 explicitly.
    sinc_half = torch.where(theta > 1e-8, torch.sin(half) / theta.clamp_min(1e-8), torch.full_like(theta, 0.5))
    xyz = aa * sinc_half
    w = torch.cos(half)
    return torch.cat([w, xyz], dim=-1)


class DeformNet(nn.Module):
    def __init__(self, xyz_freqs=6, t_freqs=4, hidden_dim=128, n_layers=3, t_scale=1.0,
                 max_dxyz=1.0, max_daxis_angle=1.0, max_dscale=2.0):
        """max_dxyz/max_daxis_angle/max_dscale: hard bounds on the network's raw output,
        applied via tanh (see forward()). Without this, `fourier_encode()`'s raw-coordinate
        high-frequency bands (2^0..2^(xyz_freqs-1) * pi, i.e. up to ~100 rad/unit at the
        default xyz_freqs=6) make the encoding -- and so the head's output -- extremely
        sensitive to small weight updates once the zero-initialized head starts moving off
        zero. Combined with the dxyz/dscale_log -> xyz/_scaling feedback loop (see
        apply_deform()), this was observed to runaway-diverge within ~100 iterations on the
        panoramic (cubemap) + 4DGS combined test (dscale_log and dxyz both grew roughly
        exponentially from near-0 starting at iteration ~16, reaching |dscale_log|~5,
        i.e. an e^5~148x Gaussian size blowup, before the training loss went NaN -- see
        panoramic_4dgs_status.md section 3.4). Bounding the output by construction removes
        this failure mode regardless of the underlying cause; the bounds are generous enough
        (dxyz up to 1 world unit, dscale_log up to +-2 i.e. up to ~7.4x size change) not to
        constrain genuine motion in normal use, while making catastrophic divergence
        impossible. tanh(0)=0 so the zero-initialized head is still an exact no-op at init."""
        super().__init__()
        self.t_scale = t_scale  # divides raw timestamps before encoding (see forward())
        self.max_dxyz = max_dxyz
        self.max_daxis_angle = max_daxis_angle
        self.max_dscale = max_dscale

        in_dim = 3 * (1 + 2 * xyz_freqs) + 1 * (1 + 2 * t_freqs)
        self.xyz_freqs = xyz_freqs
        self.t_freqs = t_freqs

        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
            d = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, 3 + 3 + 3)  # dxyz(3), daxis_angle(3), dscale_log(3)

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, xyz, t, ramp=1.0):
        """xyz: (N,3) canonical positions. t: python float / 0-d tensor (one timestamp per
        render call -- every Gaussian in the scene is queried at the same instant).
        ramp: float in [0,1], see gs_backend.py::render_at() -- multiplies the head's raw
        output *before* tanh (not after). This is a curriculum mechanism added after
        observing (panoramic_4dgs_status.md section 3.15) that an on/off warmup
        (`deform_warmup_iters`, fully skip deform_net for N iterations then fully enable it)
        only delayed saturation by roughly the warmup length, never prevented it: once
        unfrozen, dxyz/dscale_log still raced to their tanh bound within ~100 iterations
        regardless of the delay. Scaling the pre-tanh value keeps tanh's *argument* small
        early in training even if the underlying linear head wants to produce something
        large -- so tanh'(arg) stays meaningfully nonzero and gradient (both photometric and
        regularization) keeps flowing, instead of the on/off warmup's failure mode where the
        network can still saturate tanh (and so kill its own gradient) within the first
        ~100 iterations after being unfrozen. Scaling after tanh (i.e. ramp * tanh(x)) would
        NOT have this property: the tanh argument itself could still blow up and saturate,
        silently killing gradient flow even while the visible output stays small.
        Returns (dxyz, drot_quat, dscale_log), each broadcastable against the Gaussian tensors."""
        N = xyz.shape[0]
        t_tensor = torch.full((N, 1), float(t) / self.t_scale, device=xyz.device, dtype=xyz.dtype)

        enc = torch.cat([fourier_encode(xyz, self.xyz_freqs),
                          fourier_encode(t_tensor, self.t_freqs)], dim=-1)
        feat = self.backbone(enc)
        raw = self.head(feat)  # pre-ramp, pre-tanh -- the network's "true" internal magnitude;
                                # returned separately so regularization_loss() can penalize it
                                # directly (see that method's docstring for why).
        out = raw * ramp
        dxyz, daxis_angle, dscale_log = out.split([3, 3, 3], dim=-1)
        dxyz = torch.tanh(dxyz) * self.max_dxyz
        daxis_angle = torch.tanh(daxis_angle) * self.max_daxis_angle
        dscale_log = torch.tanh(dscale_log) * self.max_dscale
        drot_quat = axis_angle_to_quat(daxis_angle)
        return dxyz, drot_quat, dscale_log, raw

    def regularization_loss(self, dxyz, drot_quat, dscale_log, raw=None, raw_weight=0.01):
        """L1 penalty on tanh's *output* (dxyz/drot/dscale) encourages Delta->0 the normal
        way -- but once tanh saturates, its own gradient vanishes (tanh'~=0), so this L1 term
        alone cannot undo saturation once it happens (panoramic_4dgs_status.md section 3.15
        confirmed this empirically: 20x this L1 weight had zero effect on an already-saturated
        dxyz). `raw` is forward()'s pre-tanh, pre-ramp value -- the network's true internal
        magnitude, independent of the curriculum ramp. An L2 penalty directly on `raw` has a
        gradient that does NOT vanish when tanh saturates (d(raw^2)/d(raw) = 2*raw stays
        nonzero no matter how large raw is), so it keeps pulling the network's *internal*
        representation back toward small values even while the visible tanh output looks
        "stuck" at the bound."""
        identity_w = torch.ones_like(drot_quat[..., 0])
        rot_penalty = (drot_quat[..., 0] - identity_w).abs()  # 0 when drot_quat == identity
        loss = dxyz.abs().mean() + rot_penalty.mean() + dscale_log.abs().mean()
        if raw is not None:
            loss = loss + raw_weight * raw.pow(2).mean()
        return loss


def apply_deform(xyz, scaling_log, rotation_quat, dxyz, drot_quat, dscale_log):
    """Compose canonical Gaussian params with a DeformNet output.
    scaling_log: pre-activation (_scaling); rotation_quat: post-activation, normalized get_rotation."""
    new_xyz = xyz + dxyz
    new_scaling_log = scaling_log + dscale_log
    new_rotation = quat_multiply(drot_quat, rotation_quat)
    return new_xyz, new_scaling_log, new_rotation
