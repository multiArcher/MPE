"""MAIC integration: shapes, losses, targets, checkpoints and execution pruning."""

import copy
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.maic_controller import MAICMAC
from utils.maker import MACMaker, LearnerMaker


def make_args(**overrides):
    config = yaml.safe_load((ROOT / "src/config/default.yaml").read_text(encoding="utf-8"))
    config.update(yaml.safe_load((ROOT / "src/config/algs/maic.yaml").read_text(encoding="utf-8")))
    config.update(
        n_agents=3,
        n_actions=5,
        state_shape=10,
        device="cpu",
        use_cuda=False,
        hidden_dim=16,
        latent_dim=4,
        attention_dim=8,
        nn_hidden_size=16,
        target_update_interval=2,
        learner_log_interval=1,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def make_batch(args, length=5):
    scheme = {
        "obs": {"vshape": 8, "group": "agents"},
        "state": {"vshape": 10},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (5,), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    batch = EpisodeBatch(
        scheme,
        {"agents": args.n_agents},
        2,
        length,
        preprocess={"actions": ("actions_onehot", [OneHot(args.n_actions)])},
        device=args.device,
    )
    batch.update(
        {
            "obs": th.randn(2, length, args.n_agents, 8),
            "state": th.randn(2, length, 10),
            "actions": th.randint(1, args.n_actions, (2, length, args.n_agents, 1)),
            "avail_actions": th.ones(2, length, args.n_agents, args.n_actions),
            "reward": th.randn(2, length, 1),
            "terminated": th.zeros(2, length, 1),
        }
    )
    batch["avail_actions"][..., 0] = 0
    batch["terminated"][0, 1] = 1
    batch["filled"][0, 3:] = 0
    batch["terminated"][1, length - 2] = 1
    return batch


class Logger:
    def __init__(self):
        self.stats = {}

    def log_stat(self, key, value, t):
        self.stats[key] = value


class MAICTest(unittest.TestCase):
    def setUp(self):
        th.set_num_threads(1)
        th.manual_seed(7)
        random.seed(7)
        self.args = make_args()
        self.batch = make_batch(self.args)
        self.mac = MACMaker.make("maic_mac", self.batch.scheme, {"agents": 3}, self.args)

    def learner(self, mac=None, args=None):
        return LearnerMaker.make(
            "maic_learner", mac or self.mac, self.batch.scheme, Logger(), args or self.args
        )

    def test_forward_masks_illegal_actions_and_observation_delay(self):
        self.assertEqual(self.mac.agent.hidden_dim, 16)
        self.mac.init_hidden(2)
        selected = self.mac.select_actions(self.batch, 0, 0, bs=[1], test_mode=True)
        self.assertEqual(tuple(selected.shape), (1, 3))
        self.assertTrue((selected != 0).all())

        self.mac.init_hidden(2)
        q_test, losses = self.mac.forward(self.batch, 0, test_mode=True, train_mode=True)
        self.assertEqual(tuple(q_test.shape), (2, 3, 5))
        self.assertIn("mi_loss", losses)
        self.assertIn("entropy_loss", losses)
        self.assertTrue(th.isfinite(losses["mi_loss"]))
        self.assertTrue(losses["entropy_loss"].requires_grad)

        self.mac.init_hidden(2)
        q_again, _ = self.mac.forward(self.batch, 0, test_mode=True, train_mode=False)
        th.testing.assert_close(q_test, q_again)

        self.mac.observation_delay_model.enabled = True
        self.mac.observation_delay_model.apply_train = True
        self.mac.observation_delay_model.delay_mean = 2
        self.mac.train()
        inputs = self.mac._build_inputs(self.batch, 3).reshape(2, 3, -1)
        th.testing.assert_close(inputs[..., :8], self.batch["obs"][:, 1])

    def test_train_updates_targets_and_checkpoint(self):
        learner = self.learner()
        old = copy.deepcopy(self.mac.agent.state_dict())
        learner.train(self.batch, 0, 0)
        self.assertTrue(
            any(not th.equal(v, old[k]) for k, v in self.mac.agent.state_dict().items())
        )
        self.assertIn("loss/mi_loss", learner.logger.stats)
        self.assertIn("loss/entropy_loss", learner.logger.stats)
        self.assertIn("loss/td_loss", learner.logger.stats)
        learner.train(self.batch, 20, 2)
        for key, value in self.mac.agent.state_dict().items():
            th.testing.assert_close(value, learner.target_mac.agent.state_dict()[key])
        for key, value in learner.mixer.state_dict().items():
            th.testing.assert_close(value, learner.target_mixer.state_dict()[key])
        self.assertTrue(all(th.isfinite(th.tensor(v)) for v in learner.logger.stats.values()))

        with tempfile.TemporaryDirectory() as path:
            learner.save_models(path)
            loaded = self.learner(copy.deepcopy(self.mac))
            loaded.load_models(path)
            for original, restored in (
                (learner.mac, loaded.mac),
                (learner.target_mac, loaded.target_mac),
                (learner.mixer, loaded.mixer),
                (learner.target_mixer, loaded.target_mixer),
            ):
                for key, value in original.state_dict().items():
                    th.testing.assert_close(value, restored.state_dict()[key])
            th.manual_seed(9)
            learner.train(self.batch, 40, 4)
            th.manual_seed(9)
            loaded.train(self.batch, 40, 4)
            for key, value in learner.mac.state_dict().items():
                th.testing.assert_close(value, loaded.mac.state_dict()[key])

    def test_auxiliary_weights_and_vdn(self):
        args = make_args(mi_loss_weight=0, entropy_loss_weight=0, mixer="vdn")
        mac = MAICMAC(self.batch.scheme, {}, args)
        learner = self.learner(mac, args)
        learner.train(self.batch, 0, 2)
        self.assertNotIn("loss/mi_loss", learner.logger.stats)
        self.assertNotIn("loss/entropy_loss", learner.logger.stats)
        self.assertIn("loss/td_loss", learner.logger.stats)

    def test_rejects_unsupported_settings(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            MAICMAC(self.batch.scheme, {}, make_args(n_agents=1))
        with self.assertRaisesRegex(ValueError, "unstandardised"):
            self.learner(args=make_args(standardise_rewards=True))
        with self.assertRaisesRegex(ValueError, "common_reward"):
            self.learner(args=make_args(common_reward=False))
        with self.assertRaisesRegex(ValueError, "qmix or vdn"):
            self.learner(args=make_args(mixer=None))

    def test_empty_mask_is_a_no_op(self):
        learner = self.learner()
        before = copy.deepcopy(self.mac.state_dict())
        empty = make_batch(self.args)
        empty["filled"][:] = 0
        learner.train(empty, 0, 0)
        for key, value in before.items():
            th.testing.assert_close(value, self.mac.state_dict()[key])

    @unittest.skipUnless(th.cuda.is_available(), "CUDA unavailable")
    def test_cuda_train_and_move_to_cpu(self):
        learner = self.learner().to("cuda")
        self.batch.to("cuda")
        learner.train(self.batch, 0, 2)
        learner.to("cpu")
        self.batch.to("cpu")
        learner.train(self.batch, 30, 4)


if __name__ == "__main__":
    unittest.main()
