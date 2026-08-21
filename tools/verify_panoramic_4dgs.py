"""Check that a finished run earns all four words of "panoramic 4DGS".

Each word is a separate claim and this project has, at various points, satisfied
some while quietly failing others:

  panoramic   render_erp_panoramas produced real 360 images while the four-face
              calib left 53% of every sphere black (section 3.47)
  4D          render_time_sweep proved times between observations are defined,
              but showed it on one 384x384 cubemap face
  unobserved  the two above never met: no full sphere was ever rendered at a
              time nobody observed, until section 3.47
  measured    every reported PSNR came from final_result_kf.json, whose per_cam
              entries are cubemap faces -- the one domain where the panoramic
              problem is defined away (section 3.47)

So a single mean is not evidence. This walks the four claims separately and
reports the number behind each.

The frame-spread check exists because a mean hid an 11 dB failure for four runs:
t=1 collapsed to 16.3 dB while the mean read a healthy 25.9 (sections
3.49/3.50). Worst-frame and spread catch that; an average never will.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint_replay import load_gaussians, observed_times  # noqa: E402

# a frame this far below its run's median is a failure the mean would hide
SPREAD_LIMIT_DB = 3.0
COVERAGE_LIMIT = 0.95


class Report:
    def __init__(self):
        self.rows = []

    def add(self, claim, ok, detail):
        self.rows.append((claim, ok, detail))

    def show(self):
        width = max(len(c) for c, _, _ in self.rows)
        print()
        for claim, ok, detail in self.rows:
            mark = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
            print(f"   [{mark}] {claim.ljust(width)}  {detail}")
        failed = [c for c, ok, _ in self.rows if ok is False]
        print()
        if failed:
            print(f"   {len(failed)} claim(s) unmet: {', '.join(failed)}")
            return 1
        print("   every claim met")
        return 0


def stamps_of(directory):
    """``{stamp: path}`` for ``erp_<stamp>_<tag>.jpg`` panoramas."""
    out = {}
    for path in Path(directory).glob("erp_*.jpg"):
        found = re.findall(r"erp_(\d+\.\d+)", path.name)
        if found:
            out[float(found[0])] = path
    return out


def check_panoramic(report, run):
    """Real 360 output, and how much of the sphere it actually covers."""
    observed_dir = run / "renders" / "erp_panorama"
    frames = stamps_of(observed_dir)
    if not frames:
        report.add("panoramic: 360 output exists", False,
                   f"no erp_*.jpg under {observed_dir}")
        return None
    image = cv2.imread(str(next(iter(frames.values()))))
    height, width = image.shape[:2]
    aspect_ok = abs(width / height - 2.0) < 1e-6

    covered = []
    for path in frames.values():
        frame = cv2.imread(str(path))
        covered.append(float((frame.sum(axis=2) > 0).mean()))
    coverage = float(np.mean(covered))
    report.add("panoramic: equirectangular 2:1 output", aspect_ok,
               f"{width}x{height} over {len(frames)} instants")
    report.add("panoramic: sphere coverage", coverage >= COVERAGE_LIMIT,
               f"{100 * coverage:.1f}% (four horizontal faces give 46.5%; "
               f"calib/equirect_6face.yml gives 99.9%)")
    return frames


def check_4d(report, run):
    """The map carries a time extent per Gaussian, and it is actually used."""
    checkpoint = run / "4dgs_final.pt"
    if not checkpoint.exists():
        report.add("4D: temporal parameters stored", False, "no 4dgs_final.pt")
        return None
    gaussians, mode = load_gaussians(str(checkpoint))
    if mode != "gaussian_4d":
        report.add("4D: temporal parameters stored", False,
                   f"checkpoint dynamic_mode is {mode!r}")
        return None

    dynamic = gaussians.dynamic_mask
    times = observed_times(gaussians)
    scale = gaussians.get_time_scale.reshape(-1)[dynamic]
    velocity = gaussians.get_velocity[dynamic]
    moving = float((velocity.norm(dim=1) > 0).float().mean())

    report.add("4D: per-Gaussian time centre / radius / velocity",
               bool(dynamic.any()) and float(scale.min()) > 0,
               f"{int(dynamic.sum())} dynamic rows over {len(times)} observed "
               f"times, temporal radius {float(scale.median()):.3f}")
    report.add("4D: velocity assigned to the dynamic rows", moving > 0.99,
               f"{100 * moving:.1f}% of dynamic rows carry a non-zero velocity")
    return times


def check_unobserved(report, run, observed_frames, observed_times_list):
    """Full spheres at instants nobody captured -- the claim that needs both."""
    interp_dir = run / "renders" / "erp_panorama_interp"
    frames = stamps_of(interp_dir)
    if not frames:
        report.add("unobserved: full sphere between observations", False,
                   f"no panoramas under {interp_dir}")
        return
    observed = {round(float(t), 6) for t in (observed_times_list or [])}
    novel = [s for s in frames if round(s, 6) not in observed]

    covered = [float((cv2.imread(str(p)).sum(axis=2) > 0).mean())
               for p in frames.values()]
    report.add("unobserved: full sphere between observations",
               len(novel) == len(frames) and min(covered) >= COVERAGE_LIMIT,
               f"{len(novel)}/{len(frames)} at never-observed times, "
               f"coverage {100 * min(covered):.1f}%")

    # the object must actually move, and the static map must not
    if not observed_frames:
        return
    stamp = sorted(frames)[0]
    nearest = min(observed_frames, key=lambda o: abs(o - stamp))
    a = cv2.imread(str(frames[stamp])).astype(float)
    b = cv2.imread(str(observed_frames[nearest])).astype(float)
    difference = np.abs(a - b).mean(axis=2)
    changed = float((difference > 20).mean())
    report.add("unobserved: only the object moves", 0.0 < changed < 0.15,
               f"{100 * changed:.2f}% of pixels differ from t={nearest:g} "
               f"(0% would mean the time override did nothing; a large "
               f"fraction would mean the static map moved too)")


def check_measured(report, run, sequence, faces):
    """Score the panorama in its own domain, and look at every frame."""
    if sequence is None:
        report.add("measured: ERP-domain PSNR", None,
                   "--sequence not given, skipped")
        return
    command = [sys.executable, str(Path(__file__).parent / "eval_erp_panorama.py"),
               "--renders", str(run / "renders" / "erp_panorama"),
               "--gt", str(sequence), "--faces", *faces]
    finished = subprocess.run(command, capture_output=True, text=True)
    if finished.returncode != 0:
        report.add("measured: ERP-domain PSNR", False,
                   finished.stderr.strip().splitlines()[-1:] or "failed")
        return
    per_frame = [float(m.group(1)) for m in
                 re.finditer(r"^\s+\d+\.\d{3}\s+\d+\.\d\s+(\d+\.\d+)",
                             finished.stdout, re.M)]
    if not per_frame:
        report.add("measured: ERP-domain PSNR", False, "no per-frame rows parsed")
        return
    spread = max(per_frame) - min(per_frame)
    report.add("measured: ERP-domain PSNR (covered)", True,
               f"mean {np.mean(per_frame):.2f} dB over {len(per_frame)} frames")
    report.add("measured: no frame hidden by the mean", spread <= SPREAD_LIMIT_DB,
               f"worst {min(per_frame):.2f} dB, spread {spread:.2f} dB "
               f"(limit {SPREAD_LIMIT_DB})")


def check_ate(report, run, poses):
    if poses is None or not Path(poses).exists():
        report.add("localisation: ATE", None,
                   "--gt-poses not given, skipped")
        return
    trajectory = run / "traj_mcgs.txt"
    if not trajectory.exists():
        report.add("localisation: ATE", False, "no traj_mcgs.txt")
        return
    values = {}
    for label, flag in (("SE3", "-a"), ("Sim3", "-as")):
        finished = subprocess.run(["evo_ape", "tum", str(poses), str(trajectory),
                                   flag], capture_output=True, text=True)
        found = re.search(r"rmse\s+([\d.]+)", finished.stdout)
        values[label] = float(found.group(1)) if found else float("nan")
    report.add("localisation: ATE", True,
               f"RMSE {values['SE3']:.3f} m unaligned-scale, "
               f"{values['Sim3']:.3f} m Sim3")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="an output/<name> directory")
    parser.add_argument("--sequence", default=None,
                        help="source ERP frames, enables the ERP-domain score")
    parser.add_argument("--gt-poses", default=None,
                        help="from tools/export_synthetic_gt_poses.py")
    parser.add_argument("--faces", nargs="+",
                        default=["front", "right", "back", "left", "up", "down"])
    args = parser.parse_args()

    run = Path(args.run)
    print(f"verifying {run}")
    report = Report()
    observed_frames = check_panoramic(report, run)
    times = check_4d(report, run)
    check_unobserved(report, run, observed_frames, times)
    check_measured(report, run, args.sequence, args.faces)
    check_ate(report, run, args.gt_poses)
    raise SystemExit(report.show())


if __name__ == "__main__":
    main()
