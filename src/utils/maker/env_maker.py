from utils.custom_logging import PyMARLLogger
from envs.multiagentenv import MultiAgentEnv
from .maker import Maker


class EnvMaker(Maker):
    """Factory class for creating environments."""

    @staticmethod
    def make_mpe(*args, **kwargs) -> MultiAgentEnv:
        from envs.mpe_wrapper import MPEWrapper

        return MPEWrapper(*args, **kwargs)

    @staticmethod
    def make_delayed_mpe(*args, **kwargs) -> MultiAgentEnv:
        from envs.wrappers import DelayedObservationWrapper

        delay_kwargs = EnvMaker._pop_delay_kwargs(kwargs)
        return DelayedObservationWrapper(EnvMaker.make_mpe(*args, **kwargs), **delay_kwargs)

    @staticmethod
    def _check_and_prepare_smac_kwargs(kwargs):
        """Check and prepare kwargs for SMAC environments."""
        assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
        assert kwargs[
            "common_reward"
        ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
        # del kwargs["common_reward"]
        # del kwargs["reward_scalarisation"]
        assert "map_name" in kwargs, "Please specify the map_name in the env_args"
        return kwargs

    @staticmethod
    def make_gymma(*args, **kwargs) -> MultiAgentEnv:
        from envs.gymma import GymmaWrapper

        assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
        return GymmaWrapper(*args, **kwargs)

    @staticmethod
    def make_smaclite(*args, **kwargs) -> MultiAgentEnv:
        from envs.smaclite_wrapper import SMACliteWrapper

        kwargs = EnvMaker._check_and_prepare_smac_kwargs(kwargs)
        return SMACliteWrapper(*args, **kwargs)

    @staticmethod
    def make_sc2(*args, **kwargs) -> MultiAgentEnv:
        from envs.smac_wrapper import SMACWrapper

        kwargs = EnvMaker._check_and_prepare_smac_kwargs(kwargs)
        return SMACWrapper(*args, **kwargs)

    @staticmethod
    def make_delayed_sc2(*args, **kwargs) -> MultiAgentEnv:
        from envs.smac_wrapper import SMACWrapper
        from envs.wrappers import DelayedObservationWrapper

        kwargs = EnvMaker._check_and_prepare_smac_kwargs(kwargs)
        delay_kwargs = EnvMaker._pop_delay_kwargs(kwargs)
        env = SMACWrapper(*args, **kwargs)
        return DelayedObservationWrapper(env, **delay_kwargs)

    @staticmethod
    def make_sc2v2(*args, **kwargs) -> MultiAgentEnv:
        if len(args) != 0:
            PyMARLLogger(
                "main", PyMARLLogger.DEBUG
            ).warning(
                "SMACv2 should NOT be initialized by placeholder args. Unwanted behavior may occur."
            )

        from envs.smacv2_wrapper import SMACv2Wrapper

        kwargs = EnvMaker._check_and_prepare_smac_kwargs(kwargs)

        return SMACv2Wrapper(*args, **kwargs)

    @staticmethod
    def make_emulate_sc2v2(*args, **kwargs) -> MultiAgentEnv:
        from envs.emulate_sc2v2 import EmulateSMACv2
        kwargs = EnvMaker._check_and_prepare_smac_kwargs(kwargs)
        return EmulateSMACv2(*args, **kwargs)

    @staticmethod
    def _pop_delay_kwargs(kwargs):
        delay_kwargs = {
            "delay_type": kwargs.pop("delay_type", "gaussian"),
            "delay_mean": kwargs.pop("delay_mean", 0.0),
            "delay_std": kwargs.pop("delay_std", 0.0),
            "max_delay": kwargs.pop("max_delay", 0),
            "delay_per_agent": kwargs.pop("delay_per_agent", True),
            "delay_during_training": kwargs.pop("delay_during_training", False),
        }
        # "delay" (the legacy fixed-delay scalar) is dropped: a fixed delay d is N(d, 0).
        kwargs.pop("delay", None)
        return delay_kwargs


