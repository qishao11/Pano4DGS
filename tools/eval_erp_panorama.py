"""Score the panorama itself, in the equirectangular domain.

Every rendering number this project has reported so far -- 29.08 on the static
room, 25.4 on the dynamic one -- comes from ``final_result_kf.json``, whose
``per_cam`` entries are cubemap *faces*. Faces are pinhole images, and the whole
point of the cubemap decomposition is that distortion does not exist in them. So
those numbers, however good, cannot say anything about how well the panoramic
output is reconstructed: they are measured in the one domain where the panoramic
problem has been defined away.

This tool closes that gap by comparing ``renders/erp_panorama/*.jpg`` against the
source ERP frames the sequence was built from. Two numbers are reported per
frame and they answer different questions:

  covered   quality where the faces actually look. This is what improves when
            the reconstruction improves.
  full      quality over the entire sphere, counting uncovered regions as the
            black they are rendered as. This is what improves when faces are
            *added*, and it is the number that says whether a panorama is
            actually a panorama -- the four-face calib leaves 53% of the sphere
            black and cannot score well here no matter how good the map is.

The covered mask is derived geometrically (stitching all-white faces through the
same projection) rather than by looking for black pixels, so genuinely black
scene content is never mistaken for missing coverage.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))

import cubemap  # noqa: E402


def coverage_mask(faces, erp_h, erp_w, face_size, fov_deg):
    """Which ERP pixels the given face set can see at all."""
    white = {f: np.full((face_size, face_size, 3), 255, np.uint8) for f in faces}
    _, covered = cubemap.cubemap_to_erp(white, erp_h, erp_w, fov_deg=fov_deg)
    return covered


def stamp_of(path):
    """Physical timestamp encoded in an ``erp_<stamp>[_tag].jpg`` filename."""
    numbers = re.findall(r"\d+\.\d+|\d+", os.path.basename(path))
    return float(numbers[0]) if numbers else None


def scores(render, truth, mask):
    """PSNR/SSIM inside ``mask`` and over the whole frame."""
    out = {}
    for name, m in (("covered", mask), ("full", np.ones_like(mask))):
        if not m.any():
            continue
        a = render.astype(np.float64)
        b = truth.astype(np.float64)
        mse = float((((a - b) ** 2).mean(axis=2) * m).sum() / m.sum())
        # 100 dB stands in for an exact match; a real render never reaches it,
        # and it keeps the mean finite instead of poisoning it with inf
        out[f"{name}_psnr"] = 100.0 if mse <= 0 else 10 * np.log10(255.0 ** 2 / mse)
        _, ssim_map = structural_similarity(
            cv2.cvtColor(render, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY), full=True)
        out[f"{name}_ssim"] = float((ssim_map * m).sum() / m.sum())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders", required=True,
                        help="a run's renders/erp_panorama directory")
    parser.add_argument("--gt", required=True,
                        help="the source ERP frame directory the run was built from")
    parser.add_argument("--faces", nargs="+",
                        default=["front", "right", "back", "left"],
                        help="face set the run used; decides the covered mask")
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--timescale", type=float, default=1.0,
                        help="same --timescale the run used, to map a panorama's "
                             "stamp back onto a source frame number")
    args = parser.parse_args()

    renders = sorted(Path(args.renders).glob("erp_*.jpg"))
    if not renders:
        raise SystemExit(f"no erp_*.jpg in {args.renders}")

    truths = {}
    for path in Path(args.gt).glob("*.png"):
        numbers = re.findall(r"\d+", path.stem)
        if numbers:
            truths[float(numbers[-1]) / args.timescale] = path

    print(f"renders : {args.renders}")
    print(f"truth   : {args.gt} ({len(truths)} frames)")
    print(f"faces   : {args.faces}\n")
    print(f"   {'stamp':>8}{'covered%':>10}{'cov PSNR':>10}{'cov SSIM':>10}"
          f"{'full PSNR':>11}{'full SSIM':>11}")

    totals, skipped = [], 0
    for path in renders:
        stamp = stamp_of(path)
        truth_path = truths.get(stamp)
        if truth_path is None:
            skipped += 1
            continue
        render = cv2.imread(str(path))
        truth = cv2.imread(str(truth_path))
        if render.shape != truth.shape:
            # downsample the render onto the source grid rather than upsampling
            # the source: the comparison should never credit invented resolution
            render = cv2.resize(render, (truth.shape[1], truth.shape[0]),
                                interpolation=cv2.INTER_AREA)
        h, w = truth.shape[:2]
        mask = coverage_mask(args.faces, h, w, h // 2, args.fov)
        row = scores(render, truth, mask)
        row["stamp"], row["covered"] = stamp, float(mask.mean())
        totals.append(row)
        print(f"   {stamp:>8.3f}{100 * row['covered']:>10.1f}"
              f"{row['covered_psnr']:>10.2f}{row['covered_ssim']:>10.3f}"
              f"{row['full_psnr']:>11.2f}{row['full_ssim']:>11.3f}")

    if skipped:
        print(f"\n   {skipped} panorama(s) had no matching source frame "
              f"(unobserved times have no ground truth -- expected for "
              f"erp_panorama_interp)")
    if not totals:
        raise SystemExit("nothing scored")
    mean = lambda k: sum(r[k] for r in totals) / len(totals)  # noqa: E731
    print(f"\n   mean over {len(totals)} frames: "
          f"covered {mean('covered_psnr'):.2f} dB / {mean('covered_ssim'):.3f} SSIM   "
          f"full {mean('full_psnr'):.2f} dB / {mean('full_ssim'):.3f} SSIM   "
          f"({100 * mean('covered'):.1f}% of sphere covered)")


if __name__ == "__main__":
    main()
