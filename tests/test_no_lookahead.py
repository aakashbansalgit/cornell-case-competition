"""Each strategy's weights for month t must not change if month t's returns do.

Run from the repo root: python -m pytest tests  (or python tests/test_no_lookahead.py)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest as bt


def _weights_ignore_month(make_weights, t):
    r = bt.load_returns()
    shocked = r.copy()
    shocked.iloc[t] = -5 * shocked.iloc[t]
    before = make_weights(r).iloc[: t + 1].to_numpy()
    after = make_weights(shocked).iloc[: t + 1].to_numpy()
    return np.allclose(before, after)


def test_defensive_switch():
    for t in (30, 100, 200):
        assert _weights_ignore_month(bt.defensive_switch_weights, t)


def test_momentum_vol_target():
    for t in (30, 100, 200):
        assert _weights_ignore_month(bt.momentum_vol_target_weights, t)


def test_stop_loss_overlay():
    def stopped(r):
        return bt.with_stop_loss(r, bt.momentum_vol_target_weights(r))
    for t in (30, 100, 200):
        assert _weights_ignore_month(stopped, t)


if __name__ == "__main__":
    test_defensive_switch()
    test_momentum_vol_target()
    test_stop_loss_overlay()
    print("no strategy uses same-month returns")
