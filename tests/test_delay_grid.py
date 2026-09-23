"""MPE and SMAC delay grids stay separate."""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval_scripts"))

import delay_study


def test_mpe_grid_matches_vil2c():
    cases, cells = delay_study.conditions("mpe")
    by_id = {case["id"]: case for case in cases}
    assert len(by_id) == len(cases) == 70
    assert [by_id[f"fixed_{value}"]["value"] for value in (0, 1, 2, 4, 6, 8, 10, 12, 14, 16)] == [
        0, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    assert all(case["cap"] == 16 for case in cases)
    assert len(cells["gaussian"]) == len(cells["uniform"]) == 35
    assert {cell["mean"] for cell in cells["gaussian"]} == {-2, 0, 2, 4, 6, 8, 10}
    assert {cell["std"] for cell in cells["gaussian"]} == {0, 2, 4, 6, 8}
    uniform = by_id["uniform_0_2"]
    assert math.isclose(uniform["high"] - uniform["low"], math.sqrt(12) * 2)
    assert by_id["mixture_balanced"] == dict(
        id="mixture_balanced", kind="mixture", means=[0, 4], stds=[4, 4],
        high_probability=0.5, cap=16)
    assert by_id["mixture_rare_severe"]["means"] == [0, 8]
    assert by_id["mixture_rare_severe"]["stds"] == [4, 4]
    assert by_id["mixture_rare_severe"]["high_probability"] == 0.1
    assert by_id["periodic_16"]["means"] == [0, 4]
    assert by_id["periodic_16"]["stds"] == [4, 4]
    assert by_id["markov_09"]["means"] == [0, 4]
    assert by_id["markov_09"]["stds"] == [2, 4]
    assert by_id["markov_09"]["stay_probability"] == 0.9


def test_smac_grid_is_unchanged():
    cases, cells = delay_study.conditions("smac")
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)) == 49
    assert {cell["mean"] for cell in cells["gaussian"]} == {-2, -1, 0, 1, 2}
    assert {cell["std"] for cell in cells["gaussian"]} == {0, 0.5, 1, 1.5, 2}
    assert all(case["cap"] == 8 for case in cases)


def test_profile_follows_the_training_env():
    assert delay_study.delay_profile({"env": "mpe"}) == "mpe"
    assert delay_study.delay_profile({"env": "delayed_mpe"}) == "mpe"
    assert delay_study.delay_profile({"env": "sc2"}) == "smac"
