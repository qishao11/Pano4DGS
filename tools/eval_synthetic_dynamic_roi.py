#!/usr/bin/env python3
"""Post-hoc ROI evaluation for the synthetic moving-sphere panorama.

The main keyframe evaluator reports full-image metrics, but the synthetic sphere
occupies only a small part of each panorama.  This tool reconstructs the exact
cubemap ground truth from the ERP frames, derives an oracle sphere mask from the
known texture palette, and reports dynamic-ROI and static-background errors
separately for already-saved renders.

Saved renders are JPEGs, so these numbers are for controlled *relative* diagnosis
between runs.  They must not be mixed with the in-process full-image metrics,
which are computed from float tensors before JPEG encoding.
"""

import argparse
import json
import os
import re
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mcgs_slam.cubemap import erp_to_cubemap


FACE_NAMES = ("front", "right", "back", "left")
SPHERE_PALETTE_BGR = np.asarray(
    [
        [60, 220, 240],
        [40, 40, 220],
        [220, 220, 40],
    ],
    dtype=np.float32,
)


def dynamic_color_mask(image_bgr, threshold=0.02):
    """Return the synthetic sphere's oracle mask from a uint8 BGR image."""
    colors = image_bgr.astype(np.float32)[..., None, :]
    distances = np.linalg.norm(colors - SPHERE_PALETTE_BGR[None, None, :, :], axis=-1)
    return distances.min(axis=-1) <= float(threshold) * 255.0


class RegionAccumulator:
    def __init__(self):
        self.squared_error = 0.0
        self.absolute_error = 0.0
        self.value_count = 0
        self.pixel_count = 0
        self.visible_frames = 0

    def update(self, prediction, target, pixel_mask):
        if not pixel_mask.any():
            return
        difference = prediction.astype(np.float64) - target.astype(np.float64)
        selected = difference[pixel_mask]
        self.squared_error += float(np.square(selected).sum())
        self.absolute_error += float(np.abs(selected).sum())
        self.value_count += int(selected.size)
        self.pixel_count += int(pixel_mask.sum())
        self.visible_frames += 1

    def result(self):
        if self.value_count == 0:
            return {
                "psnr": None,
                "mae": None,
                "pixel_count": 0,
                "visible_frames": 0,
            }
        mse = self.squared_error / self.value_count
        psnr = float("inf") if mse == 0 else float(10.0 * np.log10((255.0 ** 2) / mse))
        return {
            "psnr": psnr,
            "mae": float(self.absolute_error / self.value_count),
            "pixel_count": self.pixel_count,
            "visible_frames": self.visible_frames,
        }


def _numbered_frames(dataset_dir):
    frames = {}
    for filename in os.listdir(dataset_dir):
        match = re.search(r"([+]?(?:\d*\.\d+|\d+))(?=\.[^.]+$)", filename)
        if match:
            frames[float(match.group(1))] = os.path.join(dataset_dir, filename)
    return frames


def evaluate_run(
    dataset_dir,
    run_dir,
    timestamps,
    face_size=384,
    fov=90.0,
    threshold=0.02,
):
    frame_paths = _numbered_frames(dataset_dir)
    missing = [timestamp for timestamp in timestamps if timestamp not in frame_paths]
    if missing:
        raise FileNotFoundError(f"missing ERP frames for timestamps: {missing}")

    gt_faces = {}
    for timestamp in timestamps:
        erp_image = cv2.imread(frame_paths[timestamp], cv2.IMREAD_COLOR)
        if erp_image is None:
            raise FileNotFoundError(frame_paths[timestamp])
        gt_faces[timestamp] = erp_to_cubemap(
            erp_image, face_size, faces=FACE_NAMES, fov_deg=fov)

    dynamic_all = RegionAccumulator()
    static_all = RegionAccumulator()
    per_camera = {}

    for cam_idx, face_name in enumerate(FACE_NAMES):
        image_dir = os.path.join(run_dir, "renders", "image_after_opt", f"cam{cam_idx}")
        render_paths = [
            os.path.join(image_dir, filename)
            for filename in sorted(os.listdir(image_dir))
            if filename.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if len(render_paths) != len(timestamps):
            raise ValueError(
                f"cam{cam_idx} has {len(render_paths)} renders, expected {len(timestamps)}")

        dynamic_cam = RegionAccumulator()
        static_cam = RegionAccumulator()
        for render_path, timestamp in zip(render_paths, timestamps):
            prediction = cv2.imread(render_path, cv2.IMREAD_COLOR)
            target = gt_faces[timestamp][face_name]
            if prediction is None:
                raise FileNotFoundError(render_path)
            if prediction.shape != target.shape:
                raise ValueError(
                    f"shape mismatch for {render_path}: {prediction.shape} vs {target.shape}")

            dynamic_mask = dynamic_color_mask(target, threshold=threshold)
            static_mask = ~dynamic_mask
            dynamic_cam.update(prediction, target, dynamic_mask)
            static_cam.update(prediction, target, static_mask)
            dynamic_all.update(prediction, target, dynamic_mask)
            static_all.update(prediction, target, static_mask)

        per_camera[str(cam_idx)] = {
            "face": face_name,
            "dynamic": dynamic_cam.result(),
            "static": static_cam.result(),
        }

    return {
        "run_dir": run_dir,
        "render_format_note": "post-hoc metrics from saved JPEG renders",
        "timestamps": timestamps,
        "dynamic": dynamic_all.result(),
        "static": static_all.result(),
        "per_cam": per_camera,
    }


def _parse_run(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must have the form NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("run must have the form NAME=PATH")
    return name, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="directory containing numbered ERP frames")
    parser.add_argument(
        "--run", action="append", required=True, type=_parse_run,
        help="repeatable NAME=OUTPUT_DIR entry")
    parser.add_argument(
        "--timestamps", required=True,
        help="comma-separated physical timestamps in each camera's saved-render order")
    parser.add_argument("--face-size", type=int, default=384)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--color-threshold", type=float, default=0.02)
    parser.add_argument("--output", help="optional path for the combined JSON result")
    args = parser.parse_args()

    timestamps = [float(value) for value in args.timestamps.split(",") if value.strip()]
    output = {
        name: evaluate_run(
            args.dataset,
            path,
            timestamps,
            face_size=args.face_size,
            fov=args.fov,
            threshold=args.color_threshold,
        )
        for name, path in args.run
    }
    text = json.dumps(output, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
