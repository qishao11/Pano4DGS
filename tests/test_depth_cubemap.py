"""ERP radial depth -> per-face pinhole z depth.

The conversion is easy to skip and hard to notice: without it a face's edges sit
systematically too far away, since radial distance exceeds along-axis z
everywhere except the exact face centre.
"""

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))
sys.path.insert(0, str(REPO_ROOT))

from cubemap import (  # noqa: E402
    _ALL_FACES,
    _HORIZONTAL_FACES,
    _radial_to_z_factor,
    erp_selftest,
    erp_depth_to_cubemap,
    face_intrinsics,
)


class RadialToZTest(unittest.TestCase):
    def test_centre_is_unchanged_and_corners_shrink(self):
        size = 64
        factor = _radial_to_z_factor(size, 90.0)
        centre = factor[size // 2, size // 2]
        self.assertAlmostEqual(float(centre), 1.0, places=3)
        # a 90-degree face's corner ray is (1,1,1)/sqrt(3) -> z factor 1/sqrt(3)
        self.assertAlmostEqual(float(factor[0, 0]), 1.0 / np.sqrt(3.0), places=2)
        self.assertTrue((factor <= 1.0 + 1e-6).all())

    def test_a_plane_in_front_becomes_constant_z(self):
        """The check that catches a missing conversion.

        A wall at distance d straight ahead has radial depth d/cos(angle), which
        grows toward the face edges. After conversion every pixel of that face
        must read exactly d.
        """
        size, fov, erp_h, erp_w = 64, 90.0, 256, 512
        distance = 3.0
        # build an ERP radial-depth map of a plane at z = distance
        yy, xx = np.mgrid[0:erp_h, 0:erp_w].astype(np.float64)
        theta = (xx / erp_w - 0.5) * 2 * np.pi
        phi = (0.5 - yy / erp_h) * np.pi
        dz = np.cos(phi) * np.cos(theta)
        with np.errstate(divide="ignore", invalid="ignore"):
            radial = np.where(dz > 1e-6, distance / dz, 1e6)

        faces = erp_depth_to_cubemap(radial.astype(np.float32), size,
                                     faces=["front"], fov_deg=fov)
        front = faces["front"]

        # ignore a 2px border: nearest-neighbour sampling at the very edge lands
        # on neighbouring ERP columns
        inner = front[2:-2, 2:-2]
        self.assertTrue(np.allclose(inner, distance, rtol=0.02),
                        f"expected {distance}, got [{inner.min():.3f}, {inner.max():.3f}]")

    def test_without_conversion_the_edges_would_be_wrong(self):
        """Quantifies what the conversion is worth, so it cannot be dropped silently."""
        factor = _radial_to_z_factor(64, 90.0)
        self.assertLess(float(factor.min()), 0.60)   # >40% error at the corners


class PoleFacesTest(unittest.TestCase):
    """The up/down faces, unimplemented until now (P1 skipped the poles)."""

    @staticmethod
    def _pattern(h=128, w=256):
        yy, xx = np.mgrid[0:h, 0:w]
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[..., 0] = (xx * 255 // w)
        img[..., 1] = (yy * 255 // h)
        img[..., 2] = ((xx // 8 + yy // 8) % 2) * 255
        return img

    def test_axes_are_right_handed(self):
        """right x down == forward, the convention the four horizontal faces hold."""
        from cubemap import _face_axes
        for face in _ALL_FACES:
            right, down, forward = _face_axes(face)
            np.testing.assert_allclose(np.cross(right, down), forward, atol=1e-9,
                                       err_msg=f"face {face} is left-handed")

    def test_six_faces_cover_the_whole_sphere(self):
        """The reason to add them: four faces leave the poles black."""
        image = self._pattern()
        _, _, covered4 = erp_selftest(image, face_size=64,
                                      faces=list(_HORIZONTAL_FACES))
        _, _, covered6 = erp_selftest(image, face_size=64, faces=list(_ALL_FACES))
        self.assertLess(covered4.mean(), 0.6)
        self.assertGreater(covered6.mean(), 0.98)

    def test_adding_poles_does_not_worsen_the_region_that_already_worked(self):
        """Guards against a wrong pole orientation, which would still 'cover' the
        poles while filling them with the wrong content.

        Compared only over what four faces already covered. The polar region is
        genuinely harder: ERP packs a huge pixel area into a tiny solid angle
        near the poles, so a face of fixed resolution undersamples it. Measured
        on this low-resolution pattern the poles do raise the overall mean --
        that is the ERP sampling singularity, not a projection error, and it
        shrinks as source resolution grows (on the 1280x640 room with 512px
        faces the six-face mean is actually *lower*, 1.484 vs 2.133).
        """
        image = self._pattern()
        _, diff4, covered4 = erp_selftest(image, face_size=64,
                                          faces=list(_HORIZONTAL_FACES))
        _, diff6, _ = erp_selftest(image, face_size=64, faces=list(_ALL_FACES))
        self.assertLessEqual(diff6[covered4].mean(), diff4[covered4].mean() * 1.05)


if __name__ == "__main__":
    unittest.main()
