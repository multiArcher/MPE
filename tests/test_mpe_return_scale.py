"""Theoretical MPE return colors stay fixed across algorithms."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval_scripts" / "plots"))

from mpe_return_scale import DIAGONAL, colormap, return_bounds


def test_spread_and_tag_use_fixed_anchors():
    assert return_bounds("simple_spread_v3", 25, {}, True, "sum") == (-100.0, 0.0)
    assert return_bounds("simple_spread_v3", 50, {"N": 5}, True, "mean") == (-100.0, 0.0)
    assert return_bounds("simple_tag_v3", 25, {}, True, "sum") == (0.0, 300.0)
    assert return_bounds("simple_tag_v3", 50, {"num_good": 2, "num_adversaries": 4}, False, "mean") == (0.0, 300.0)


def test_longer_episode_scales_the_same_step_range():
    short = return_bounds("simple_reference_v3", 25, {}, True, "sum")
    long = return_bounds("simple_reference_v3", 50, {}, True, "sum")
    assert short == (-2 * DIAGONAL * 25, 0)
    assert long[0] == short[0] * 2


def test_colormap_runs_from_pale_orange_to_deep_orange():
    pale = colormap()(0.0)
    deep = colormap()(1.0)
    assert pale[0] > 0.99 and pale[1] > 0.94 and pale[2] > 0.85
    assert deep[0] > 0.85 and deep[1] < 0.35 and deep[2] < 0.05
