# Pano4DGS

Panoramic 4D Gaussian splatting from equirectangular video: reconstruct a 360°
scene, and render full spheres at moments no camera ever captured.

Built on [MCGS-SLAM](https://github.com/mcgs-slam/mcgs-slam) (MIT, Copyright 2025
Zhihao Cao).

---

## What "panoramic 4DGS" means here

Four separate claims, each measured separately:

| claim | what it means | measured |
|---|---|---|
| **panoramic** | the output is a real equirectangular image, not a cubemap face | 1536×768, **100% sphere covered** |
| **4D** | every Gaussian carries a time centre, a temporal radius and a velocity | 16 703 dynamic rows over 13 observed instants |
| **unobserved** | full spheres rendered at instants nobody captured | **8/8**, coverage 100% |
| **measured** | scored in the equirectangular domain, per frame | **27.5 dB**, worst frame 27.2, spread **0.58 dB** |

The fourth one matters more than it looks. Every rendering number this project
reported for months came from cubemap faces — and faces are pinhole images, the
one domain where the panoramic problem has been defined away. Under the old
four-face configuration the face-domain score looked healthy while **53% of every
output sphere was black**. `tools/eval_erp_panorama.py` scores the panorama
itself, and reports `covered` (quality where the faces look) separately from
`full` (quality over the whole sphere).

---

## Install

```bash
conda env create -f environment.yaml     # creates mcgs_slam_v1
conda activate mcgs_slam_v1
pip install -e thirdparty/lietorch
pip install -e thirdparty/diff-gaussian-rasterization
pip install -e thirdparty/simple-knn
```

`thirdparty/eigen` and `thirdparty/.../glm` are vendored headers and are already
in the tree; the two CUDA extensions build against them directly.

Needs a CUDA GPU. Everything below was run on an RTX 3090; a full run takes about
8 minutes and roughly 5 GB of GPU memory.

## Run

```bash
python demo.py --calib calib/equirect_6face.yml \
               --imagedir data/synth_erp_room_dynamic_fast_hires \
               --config config/config_gaussian_4d.yaml \
               --stride 1 --rgbd \
               --output output/pano4d
```

`--calib calib/equirect_6face.yml` is the six-face cubemap decomposition
(front/right/back/left/**up/down**). The four-face variant
(`calib/equirect_test.yml`) covers only 46.5% of the sphere and leaves the poles
black.

Outputs:

| path | contents |
|---|---|
| `renders/erp_panorama/` | 360° panoramas at observed instants |
| `renders/erp_panorama_interp/` | **360° panoramas at instants nobody observed** |
| `renders/time_sweep/` | single-face time sweep, 49 frames of which 36 are unobserved |
| `4dgs_final.pt` | the 4D map: per-Gaussian time centre, temporal radius, velocity |

## Check

```bash
python tools/export_synthetic_gt_poses.py \
       --sequence data/synth_erp_room_dynamic_fast_hires

python tools/verify_panoramic_4dgs.py --run output/pano4d \
       --sequence data/synth_erp_room_dynamic_fast_hires \
       --gt-poses data/synth_erp_room_dynamic_fast_hires_gt_poses.txt
```

```
   [PASS] panoramic: equirectangular 2:1 output             1536x768 over 8 instants
   [PASS] panoramic: sphere coverage                        100.0%
   [PASS] 4D: per-Gaussian time centre / radius / velocity  16703 dynamic rows over 13 observed times
   [PASS] 4D: velocity assigned to the dynamic rows         100.0%
   [PASS] unobserved: full sphere between observations      8/8 at never-observed times, coverage 100.0%
   [PASS] unobserved: only the object moves                 1.60% of pixels differ from t=0
   [PASS] measured: ERP-domain PSNR (covered)               mean 27.53 dB over 8 frames
   [PASS] measured: no frame hidden by the mean             worst 27.24 dB, spread 0.58 dB
   [PASS] localisation: ATE                                 RMSE 0.109 m unaligned-scale, 0.048 m Sim3
```

The spread check exists because a mean of 25.9 dB sat on top of a frame that had
collapsed to 16.3 for four consecutive runs before anyone looked per-frame. The
verifier is checked in both directions — it fails the pre-fix configuration on
spread (11.52 dB) and the four-face configuration on coverage (48.1%).

---

## Results

All numbers below are from this repository on the synthetic panoramic sequence,
with ground-truth depth. `n=3` means three independent runs of the same
configuration.

### Rendering and localisation

| | value |
|---|---|
| ERP-domain PSNR (`covered`) | **27.5 dB** / SSIM 0.918 |
| ERP-domain PSNR (`full`, whole sphere) | **27.4 dB** |
| worst frame / spread across frames | 27.2 dB / **0.58 dB** |
| sphere coverage | **100.0%** |
| ATE, no scale correction | **0.109 m** (3.4% of the 3.25 m path) |
| ATE, Sim3 aligned | 0.048 m |
| static geometry, wall residual | **0.02 m** on an 8 × 4.4 × 8 m room |

Localisation is identical across runs — tracking is deterministic here, and the
whole run-to-run spread lives in the Gaussian refinement.

The static-geometry number is worth one line of explanation, because it also
validates its own measurement: `tools/align_map_to_room.py` fits a similarity
onto the room's six known walls and recovers a translation of `(0, 0, -1.494)`,
which is the generator's first camera to within 6 mm — a pose the fit never saw.

### Six faces against four (n=3)

| | 4 faces | 6 faces |
|---|---|---|
| face-domain PSNR, *the four shared faces only* | 25.39 ± 0.40 | **28.92 ± 0.03** |
| ERP-domain `full` PSNR | 8.71 | **25.80** |
| sphere coverage | 46.5% | **99.9%** |

The +3.5 dB is measured on the four faces both configurations share, so it is not
the pole faces flattering the average — the extra views constrain the geometry,
they do not merely fill holes. Run-to-run variance also collapses from ±0.40 dB
to ±0.03.

### The temporal radius has two conflicting jobs (n=3)

A dynamic Gaussian's temporal radius decides two unrelated things, and they pull
opposite ways:

- **narrow** so a neighbouring bank cannot make a frame's own Gaussians a
  redundant explanation of it — a redundant row gets dimmed by the photometric
  loss and then pruned, until the bank is too faint to draw the object
- **wide** so a midpoint, half a step from both neighbours, can draw on either

One number cannot do both. Two failed attempts established that before splitting
them worked:

| | ERP `covered` | worst frame | spread |
|---|---|---|---|
| single radius 0.5 | 25.93 ± 0.30 | 16.33 | 10.04 |
| per-observation weight normalisation | 25.87 ± 0.29 | 19.30 | 7.20 |
| single radius 0.25 | 27.37 | 26.49 | 1.44 |
| **split 0.25 / 0.5** | **27.38 ± 0.21** | **26.60** | **0.73** |

The normalisation attempt is instructive: it hit its own target exactly (the
neighbours' relative weight fell from 1.385 to 0.271) and changed nothing, while
the single narrow radius carried *twice* that ghost weight and scored 7 dB
higher. Ghosting was never the binding constraint.

### Object velocity comes from geometry, not from the loss (n=3)

Eight attempts across three parameterisations failed to learn object motion
photometrically; a carried Gaussian is either redundant or absent, and neither
case requires the optimiser to place it correctly. Velocity is estimated from
bank-to-bank registration instead, outside the loss:

| | placement error |
|---|---|
| no velocity | 1.0920 ± 0.0089 |
| trimmed ICP | 0.9750 ± 0.0010 |
| **centroid difference** (default) | **0.9573 ± 0.0012** |
| ground-truth velocity (bound) | 0.5950 ± 0.0096 |

ICP was introduced when bank centroids were contaminated by outliers. With clean
banks that premise is gone and the cheaper estimator wins — in all six runs
measured, across two face configurations.

---

## Known limits

Stated plainly, because the rendering numbers above look better than the
geometry underneath them.

**Dynamic Gaussians project correctly and sit at the wrong depth.** Rendering the
object looks right because 90% of its Gaussians fall inside its silhouette from
the camera that seeded them; only **50.4%** sit at the right distance along that
ray. A single supervising view cannot constrain depth along its own ray — every
position on it reprojects to the same pixel — so the photometric loss is
indifferent, and an indifferent gradient is noise. Progress so far:

| | on-sphere | in-silhouette | centroid error |
|---|---|---|---|
| seeding from BA disparity | 21.7% | 86.3% | 1.151 |
| seeding from the depth prior | 44.8% | 85.0% | 0.908 |
| **+ frozen during refinement** | **50.4%** | 89.8% | 0.929 |

Back-projecting the same pixels with ground-truth depth puts 100% of them on the
object, so seeding is now exact and the remaining error is introduced by
refinement. Freezing positions recovered 5.6 points and revealed the next outlet:
the dynamic Gaussian count rose 39%, the optimiser reaching for densification
once sliding was blocked. Constraining which degrees of freedom may move, without
changing what the loss is indifferent to, keeps getting routed around.

**Validated on one synthetic sequence.** No real dynamic panoramic dataset with
both camera ground truth and a moving object has been found; three surveys came
up empty (Princeton365 publishes pinhole crops, 360VOTS has a fixed camera, JRDB
has no pose ground truth).

**Ground-truth depth throughout.** Monocular depth is unusable on this particular
room — its structure correlates *negatively* with truth, an out-of-distribution
failure caused by the adversarial checkerboard texture and the featureless
box geometry. That is a property of the test scene, not a measurement of
monocular depth, and the monocular path is therefore untested here.

**Interpolated frames show a streak** on the object's trailing edge, which is the
depth problem above made visible.

---

## Layout

```
mcgs_slam/gaussian/deform/gaussian_4d.py   4D primitives: temporal weights, time slicing
mcgs_slam/gaussian/deform/motion_estimation.py   velocity from bank registration
mcgs_slam/cubemap.py                       ERP <-> cubemap, and the stitch back to ERP
mcgs_slam/gs_backend.py                    rendering, refinement, panorama output
config/config_gaussian_4d.yaml             the production configuration
calib/equirect_6face.yml                   six-face decomposition (99.9% of the sphere)
tools/verify_panoramic_4dgs.py             the acceptance check above
tools/eval_erp_panorama.py                 PSNR/SSIM in the equirectangular domain
tools/align_map_to_room.py                 alignment that does not use the object
tools/export_synthetic_gt_poses.py         ground-truth camera trajectory
tools/make_synthetic_erp_room.py           the synthetic sequence generator
```

```bash
pytest tests/ -q     # 179 tests
```

## Licence

MIT. See [LICENSE](LICENSE). Vendored dependencies under `thirdparty/` keep their
own licences.
