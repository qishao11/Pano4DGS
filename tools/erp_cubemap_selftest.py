#!/usr/bin/env python3
"""P0a validation for panoramic support: checks that mcgs_slam/cubemap.py's
ERP<->cubemap-face projection math is correct, using a synthetically generated
equirectangular test pattern -- no real panoramic dataset required.

Usage:
    python tools/erp_cubemap_selftest.py [--out DIR]

Generates a synthetic ERP grid pattern, splits it into cubemap faces, reprojects
the faces back onto the ERP grid, and reports the reconstruction error. Also
dumps the synthetic ERP image, each face crop, the reconstruction, and a diff
heatmap to --out for visual inspection (grid lines should stay straight and
continuous across face seams if the projection math is correct).
"""
import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mcgs_slam'))  # nopep8

import numpy as np
import cv2

import cubemap


def make_synthetic_erp(h=512, w=1024, grid_deg=15):
    """Longitude/latitude grid lines over a smooth hue gradient, so seam continuity
    and orientation are both easy to check visually."""
    yy, xx = np.mgrid[0:h, 0:w]
    lon = (xx / w) * 360.0          # 0..360
    lat = 90.0 - (yy / h) * 180.0   # +90..-90

    hue = (lon.astype(np.float32) % 360.0) / 2.0  # OpenCV hue range 0..180
    sat = np.full((h, w), 200, dtype=np.uint8)
    val = np.clip(255 - np.abs(lat) * 1.5, 60, 255).astype(np.uint8)
    hsv = np.stack([hue.astype(np.uint8), sat, val], axis=-1)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    grid = (np.mod(lon, grid_deg) < (grid_deg * w / w * 0.4)) | (np.mod(lat + 90, grid_deg) < (grid_deg * 0.4))
    lon_line = np.mod(lon, grid_deg) < (grid_deg / w * 720)
    lat_line = np.mod(lat + 90, grid_deg) < (grid_deg / h * 360)
    img[lon_line | lat_line] = (255, 255, 255)

    return img


def colorize_diff(diff, covered, vmax=40.0):
    heat = np.clip(diff / vmax * 255, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat[~covered] = (0, 0, 0)
    return heat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', '_erp_selftest_out'))
    ap.add_argument('--face_size', type=int, default=384)
    ap.add_argument('--fov', type=float, default=90.0)
    ap.add_argument('--max_mean_err', type=float, default=5.0,
                     help='fail if mean abs diff over covered pixels exceeds this (0-255 scale)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    erp = make_synthetic_erp()
    cv2.imwrite(os.path.join(args.out, 'erp_input.png'), erp)

    faces = ('front', 'right', 'back', 'left')
    face_imgs = cubemap.erp_to_cubemap(erp, args.face_size, faces=faces, fov_deg=args.fov)
    for name, im in face_imgs.items():
        cv2.imwrite(os.path.join(args.out, f'face_{name}.png'), im)

    recon, diff, covered = cubemap.erp_selftest(erp, face_size=args.face_size, faces=faces, fov_deg=args.fov)
    cv2.imwrite(os.path.join(args.out, 'erp_recon.png'), recon)
    cv2.imwrite(os.path.join(args.out, 'erp_diff_heatmap.png'), colorize_diff(diff, covered))

    mean_err = diff[covered].mean()
    max_err = diff[covered].max()
    coverage_pct = 100.0 * covered.sum() / covered.size

    print(f'Coverage: {coverage_pct:.1f}% of ERP pixels reconstructed by the 4 horizontal faces')
    print(f'Mean abs error (covered px): {mean_err:.3f} / 255')
    print(f'Max abs error  (covered px): {max_err:.3f} / 255')
    print(f'Outputs written to: {os.path.abspath(args.out)}')

    if mean_err > args.max_mean_err:
        print(f'FAIL: mean error {mean_err:.3f} exceeds threshold {args.max_mean_err}')
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
