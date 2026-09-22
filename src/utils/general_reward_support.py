def test_alg_config_supports_reward(config):
    """
    Check whether algorithm supports specified reward configuration
    """
    if config["common_reward"]:
        # all algorithms support common reward
        return True
    else:
        if config["learner"] in ("coma_learner", "qtran_learner", "maic_learner"):
            # COMA, QTRAN and MAIC only support common reward
            return False
        elif config["learner"] == "q_learner" and (
            config["mixer"] == "vdn" or config["mixer"] == "qmix"
        ):
            # VDN and QMIX only support common reward
            return False
        else:
            return True
