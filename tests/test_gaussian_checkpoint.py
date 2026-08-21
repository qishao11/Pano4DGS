import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))


@unittest.skipUnless(torch.cuda.is_available(), "GaussianModel requires CUDA")
class GaussianCheckpointTest(unittest.TestCase):
    @staticmethod
    def _populate_model(model):
        model._xyz = torch.nn.Parameter(torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda"))
        model._features_dc = torch.nn.Parameter(torch.zeros((2, 1, 3), device="cuda"))
        model._features_rest = torch.nn.Parameter(torch.zeros((2, 0, 3), device="cuda"))
        model._scaling = torch.nn.Parameter(torch.zeros((2, 3), device="cuda"))
        model._rotation = torch.nn.Parameter(torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device="cuda"))
        model._opacity = torch.nn.Parameter(torch.zeros((2, 1), device="cuda"))
        model.unique_kfIDs = torch.tensor([0, 501], dtype=torch.int32)
        model.n_obs = torch.tensor([3, 4], dtype=torch.int32)
        model.dynamic_score = torch.tensor([[0.0], [1.0]], device="cuda")
        model.dynamic_source_time = torch.tensor([[-1.0], [0.4]], device="cuda")
        model.dynamic_object_id = torch.tensor([[-1.0], [0.0]], device="cuda")
        # 4D parameters: distinct non-default values, so a round trip that
        # dropped or defaulted them would be visible rather than accidentally
        # matching the freshly-initialized state
        model._velocity = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, -0.25, 0.125]], device="cuda")
        model._time_scale_raw = torch.tensor([[0.3], [1.7]], device="cuda")
        model.max_radii2D = torch.tensor([1.0, 2.0], device="cuda")
        model.xyz_gradient_accum = torch.tensor([[0.1], [0.2]], device="cuda")
        model.denom = torch.tensor([[2.0], [3.0]], device="cuda")

    def test_dynamic_metadata_round_trip(self):
        from gaussian.scene.gaussian_model import GaussianModel

        model = GaussianModel(sh_degree=0, config={"Training": {"deform_cfg": {}}})
        self._populate_model(model)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "gaussians.pt"
            torch.save(model.checkpoint_state(), checkpoint_path)
            serialized_state = torch.load(checkpoint_path, map_location="cpu")

        restored = GaussianModel(
            sh_degree=0, config={"Training": {"deform_cfg": {}}})
        restored.restore_checkpoint_state(serialized_state)

        torch.testing.assert_close(restored._xyz, model._xyz)
        torch.testing.assert_close(restored.dynamic_score, model.dynamic_score)
        torch.testing.assert_close(
            restored.dynamic_source_time, model.dynamic_source_time)
        torch.testing.assert_close(
            restored.dynamic_object_id, model.dynamic_object_id)
        torch.testing.assert_close(restored.unique_kfIDs, model.unique_kfIDs)
        torch.testing.assert_close(restored.n_obs, model.n_obs)
        torch.testing.assert_close(restored._velocity, model._velocity)
        torch.testing.assert_close(restored._time_scale_raw, model._time_scale_raw)
        # the softplus view must survive too, not just the raw storage
        torch.testing.assert_close(restored.get_time_scale, model.get_time_scale)

    def test_checkpoint_predating_4d_parameters_still_loads(self):
        """Old checkpoints restore to "no motion, default radius".

        That is the state a fresh seed starts from, so a resumed run behaves the
        same as one that never had these parameters -- unlike the object-ID case
        in section 3.28, where defaulting to -1 would have silently relabelled
        dynamic rows as static.
        """
        from gaussian.scene.gaussian_model import GaussianModel

        config = {"Training": {"deform_cfg": {"time_scale_init": 0.5}}}
        model = GaussianModel(sh_degree=0, config=config)
        self._populate_model(model)
        state = model.checkpoint_state()
        del state["tensors"]["_velocity"]
        del state["tensors"]["_time_scale_raw"]

        restored = GaussianModel(sh_degree=0, config=config)
        restored.restore_checkpoint_state(state)

        self.assertEqual(restored._velocity.shape, (2, 3))
        torch.testing.assert_close(
            restored._velocity, torch.zeros((2, 3), device="cuda"))
        torch.testing.assert_close(
            restored.get_time_scale,
            torch.full((2, 1), 0.5, device="cuda"), atol=1e-6, rtol=0.0)

    def test_short_4d_column_is_rejected(self):
        """A silently short column would mis-assign motion to every later row."""
        from gaussian.scene.gaussian_model import GaussianModel

        model = GaussianModel(sh_degree=0, config={"Training": {"deform_cfg": {}}})
        self._populate_model(model)
        state = model.checkpoint_state()
        state["tensors"]["_velocity"] = state["tensors"]["_velocity"][:1]

        restored = GaussianModel(sh_degree=0, config={"Training": {"deform_cfg": {}}})
        with self.assertRaisesRegex(ValueError, "point-count mismatch"):
            restored.restore_checkpoint_state(state)

    def test_4d_parameters_stay_attached_to_the_optimizer(self):
        """Densify/prune must route them through the optimizer, not rebind them.

        A bare torch.cat or mask-slice would leave the optimizer stepping a
        tensor that no longer feeds the render -- silently, and it trains
        nothing. That is the failure section 3.27 had to chase down in
        ObjectSE3Trajectory.add_observation.
        """
        from munch import munchify
        from gaussian.scene.gaussian_model import GaussianModel

        model = GaussianModel(sh_degree=0, config={"Training": {"deform_cfg": {}}})
        self._populate_model(model)
        model.init_lr(1.0)
        model.training_setup(munchify({
            "percent_dense": 0.01, "position_lr_init": 0.00016,
            "position_lr_final": 0.0000016, "position_lr_max_steps": 30000,
            "feature_lr": 0.0025, "opacity_lr": 0.05, "scaling_lr": 0.001,
            "rotation_lr": 0.001,
        }))

        def group(name):
            return next(g for g in model.optimizer.param_groups
                        if g["name"] == name)

        for name, attribute in (("velocity", "_velocity"),
                                ("time_scale", "_time_scale_raw")):
            self.assertIs(group(name)["params"][0], getattr(model, attribute))

        # grow (densification) then shrink (prune), the two paths that rebuild
        # the parameter objects
        model.densification_postfix(
            new_xyz=torch.zeros((1, 3), device="cuda"),
            new_features_dc=torch.zeros((1, 1, 3), device="cuda"),
            new_features_rest=torch.zeros((1, 0, 3), device="cuda"),
            new_opacities=torch.zeros((1, 1), device="cuda"),
            new_scaling=torch.zeros((1, 3), device="cuda"),
            new_rotation=torch.zeros((1, 4), device="cuda"),
            new_kf_ids=torch.tensor([7], dtype=torch.int32),
            new_n_obs=torch.tensor([1], dtype=torch.int32),
        )
        self.assertEqual(model._velocity.shape[0], 3)
        for name, attribute in (("velocity", "_velocity"),
                                ("time_scale", "_time_scale_raw")):
            self.assertIs(group(name)["params"][0], getattr(model, attribute))

        model.prune_points(torch.tensor([False, True, False], device="cuda"))
        self.assertEqual(model._velocity.shape[0], 2)
        for name, attribute in (("velocity", "_velocity"),
                                ("time_scale", "_time_scale_raw")):
            self.assertIs(group(name)["params"][0], getattr(model, attribute))

    def test_backend_file_round_trip_restores_time_and_dynamic_state(self):
        from gs_backend import GSBackEnd
        from utils.utils import load_config

        config = load_config(str(REPO_ROOT / "config/config_oracle_time_slice.yaml"))
        with tempfile.TemporaryDirectory() as directory:
            backend = GSBackEnd(config, directory)
            self._populate_model(backend.gaussians)
            backend.time_metadata = {
                "origin": 1000.0,
                "scale": 1.0,
                "unit": "seconds",
                "value_semantics": "(source_timestamp - origin) / scale",
            }
            checkpoint_path = Path(directory) / "4dgs.pt"
            backend.save_checkpoint(str(checkpoint_path))

            restored = GSBackEnd.from_checkpoint(str(checkpoint_path))

            self.assertEqual(restored.time_metadata, backend.time_metadata)
            torch.testing.assert_close(
                restored.gaussians.dynamic_score,
                backend.gaussians.dynamic_score,
            )
            torch.testing.assert_close(
                restored.gaussians.dynamic_source_time,
                backend.gaussians.dynamic_source_time,
            )
            torch.testing.assert_close(
                restored.gaussians.dynamic_object_id,
                backend.gaussians.dynamic_object_id,
            )
            torch.testing.assert_close(
                restored.gaussians.get_opacity_at_time(0.4),
                torch.full((2, 1), 0.5, device="cuda"),
            )

    def test_backend_file_round_trip_restores_deform_net(self):
        from gs_backend import GSBackEnd
        from utils.utils import load_config

        config = load_config(str(REPO_ROOT / "config/config_rampfix.yaml"))
        config["Training"]["deform"] = True
        with tempfile.TemporaryDirectory() as directory:
            backend = GSBackEnd(config, directory)
            self._populate_model(backend.gaussians)
            with torch.no_grad():
                next(backend.deform_net.parameters()).fill_(0.125)
            checkpoint_path = Path(directory) / "4dgs_deform.pt"
            backend.save_checkpoint(str(checkpoint_path))

            restored = GSBackEnd.from_checkpoint(str(checkpoint_path))

            for expected, actual in zip(
                    backend.deform_net.parameters(),
                    restored.deform_net.parameters()):
                torch.testing.assert_close(actual, expected)

    def test_backend_file_round_trip_restores_object_trajectories(self):
        from gs_backend import GSBackEnd
        from utils.utils import load_config

        config = load_config(str(REPO_ROOT / "config/config_rampfix.yaml"))
        config["Training"]["dynamic_mode"] = "object_se3"
        with tempfile.TemporaryDirectory() as directory:
            backend = GSBackEnd(config, directory)
            self._populate_model(backend.gaussians)
            backend.object_trajectories.observe(
                2, 0.0, [0.0, 0.0, 0.0])
            backend.object_trajectories.observe(
                2, 1.0, [2.0, 0.0, 0.0])
            checkpoint_path = Path(directory) / "4dgs_object.pt"
            backend.save_checkpoint(str(checkpoint_path))

            restored = GSBackEnd.from_checkpoint(str(checkpoint_path))

            self.assertEqual(restored.dynamic_mode, "object_se3")
            translation, _ = restored.object_trajectories.evaluate(2, 0.25)
            torch.testing.assert_close(
                translation, torch.tensor([0.5, 0.0, 0.0], device="cuda"))

    def test_restore_rejects_mismatched_dynamic_rows(self):
        from gaussian.scene.gaussian_model import GaussianModel

        model = GaussianModel(sh_degree=0, config={"Training": {"deform_cfg": {}}})
        state = {
            "tensors": {
                "_xyz": torch.zeros((2, 3)),
                "_features_dc": torch.zeros((2, 1, 3)),
                "_features_rest": torch.zeros((2, 0, 3)),
                "_scaling": torch.zeros((2, 3)),
                "_rotation": torch.zeros((2, 4)),
                "_opacity": torch.zeros((2, 1)),
                "unique_kfIDs": torch.zeros(2),
                "n_obs": torch.zeros(2),
                "dynamic_score": torch.zeros((1, 1)),
                "dynamic_source_time": torch.zeros((2, 1)),
            }
        }

        with self.assertRaisesRegex(ValueError, "point-count mismatch"):
            model.restore_checkpoint_state(state)


if __name__ == "__main__":
    unittest.main()
