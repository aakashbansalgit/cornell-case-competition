"""
Defensive allocation strategies against a static 60/20/20 portfolio.

Rebuilt on public data after the 2025 Cornell Investment Portfolio Case
Competition. Every allocation for month t is decided from data available at
the end of month t-1, and every strategy pays the same cost on turnover.

    python backtest.py

Writes results/summary.csv, results/worst_months.csv and the figures in
figures/. Price data is cached in data/returns.csv after the first run.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "returns.csv"
START, END = "2003-09-01", "2025-01-01"

TX_COST = 0.001          # 10 bp per unit of turnover
STATIC = (0.60, 0.20, 0.20)  # equity, bonds, T-bills


# ---- data ------------------------------------------------------------------

def load_returns():
    """Monthly total returns for SPY and AGG, and a T-bill return.

    The T-bill return for month t is last month's 3-month bill rate divided by
    12, since that is the rate an investor could lock in at the start of t.
    """
    if DATA.exists():
        return pd.read_csv(DATA, index_col=0, parse_dates=True)

    import yfinance as yf

    px = yf.download(["SPY", "AGG"], start=START, end=END, interval="1mo",
                     auto_adjust=True, progress=False)["Close"]
    rets = px.pct_change().dropna()

    tb = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS",
                     index_col=0, parse_dates=True)["TB3MS"]
    tbill = (tb.shift(1) / 1200).reindex(rets.index)

    out = pd.DataFrame({"equity": rets["SPY"], "bonds": rets["AGG"],
                        "tbills": tbill}).dropna()
    DATA.parent.mkdir(exist_ok=True)
    out.to_csv(DATA)
    return out


# ---- strategies ------------------------------------------------------------
# Each returns a DataFrame of target weights indexed like the returns. Row t
# holds the weights for month t, built only from rows before t.

def static_weights(r):
    return pd.DataFrame([STATIC] * len(r), index=r.index, columns=r.columns)


def defensive_switch_weights(r, stress=(0.20, 0.35, 0.45), vol_window=12,
                             vol_multiple=1.5, losing_streak=3):
    """Move to the stress mix after three losing equity months in a row, or
    when trailing equity volatility runs above 1.5 times its own history.
    Return to 60/20/20 after a positive month with volatility back to normal.

    The stress mix is the one the team settled on in the competition."""
    eq = r["equity"]
    vol = eq.rolling(vol_window).std()
    vol_norm = vol.expanding().mean()          # history only, no full-sample mean

    weights, stressed, streak = [], False, 0
    for t in range(len(r)):
        if t > 0:
            last = eq.iloc[t - 1]
            streak = streak + 1 if last < 0 else 0
            v, v_bar = vol.iloc[t - 1], vol_norm.iloc[t - 1]
            high_vol = pd.notna(v) and v > vol_multiple * v_bar
            if not stressed and (streak >= losing_streak or high_vol):
                stressed = True
            elif stressed and last > 0 and pd.notna(v) and v <= v_bar:
                stressed = False
        weights.append(stress if stressed else STATIC)
    return pd.DataFrame(weights, index=r.index, columns=r.columns)


def momentum_vol_target_weights(r, max_equity=0.70, bonds=0.15, target_vol=0.08,
                                lookback=12):
    """Dual momentum with volatility targeting.

    Hold equities only when their trailing 12-month return beats T-bills over
    the same window. When they do, scale the equity weight down so that its
    trailing volatility contribution stays near 8% a year, keep 15% in bonds
    and put the rest in T-bills. Otherwise hold T-bills."""
    eq_12m = (1 + r["equity"]).rolling(lookback).apply(np.prod, raw=True) - 1
    tb_12m = (1 + r["tbills"]).rolling(lookback).apply(np.prod, raw=True) - 1
    vol = r["equity"].rolling(lookback).std() * np.sqrt(12)

    # Shift so month t only sees signals computed through month t-1.
    signal = (eq_12m > tb_12m).shift(1, fill_value=False)
    vol = vol.shift(1)

    w_eq = np.where(signal, max_equity * np.minimum(1, target_vol / vol), 0.0)
    w_eq = np.nan_to_num(w_eq)
    w_bd = np.where(signal, bonds, 0.0)
    return pd.DataFrame({"equity": w_eq, "bonds": w_bd,
                         "tbills": 1 - w_eq - w_bd}, index=r.index)


def with_stop_loss(r, weights, limit=0.05):
    """Overlay a stop: if the strategy ended last month more than 5% below its
    peak, hold T-bills this month and restart the peak from there."""
    w = weights.copy()
    nav, peak = 1.0, 1.0
    for t in range(len(r)):
        if nav < (1 - limit) * peak:
            w.iloc[t] = (0.0, 0.0, 1.0)
            peak = nav
        nav *= 1 + gross_return(r.iloc[[t]], w.iloc[[t]]).iloc[0]
        peak = max(peak, nav)
    return w


# ---- evaluation ------------------------------------------------------------

def gross_return(r, w):
    return (r[w.columns] * w).sum(axis=1)


def net_return(r, w):
    turnover = w.diff().abs().sum(axis=1).fillna(0)
    return gross_return(r, w) - TX_COST * turnover, turnover


def max_drawdown(ret):
    nav = (1 + ret).cumprod()
    return (1 - nav / nav.cummax()).max()


def summarise(ret, turnover, tbills):
    years = len(ret) / 12
    excess = ret - tbills
    return {
        "annual_return": (1 + ret).prod() ** (1 / years) - 1,
        "annual_vol": ret.std() * np.sqrt(12),
        "sharpe": excess.mean() / excess.std() * np.sqrt(12),
        "max_drawdown": max_drawdown(ret),
        "worst_month": ret.min(),
        "annual_turnover": turnover.sum() / years,
    }


def main():
    r = load_returns()

    strategies = {
        "Static 60/20/20": static_weights(r),
        "Defensive switch": defensive_switch_weights(r),
        "Momentum, vol target": momentum_vol_target_weights(r),
    }
    strategies["Momentum, vol target, stop"] = with_stop_loss(
        r, strategies["Momentum, vol target"])

    # Compare on the same months: drop the 12-month warm-up plus the month the
    # first signal is formed in.
    first = r.index[13]
    rets, rows = {}, {}
    for name, w in strategies.items():
        ret, turnover = net_return(r, w)
        ret, turnover = ret[first:], turnover[first:]
        rets[name] = ret
        rows[name] = summarise(ret, turnover, r["tbills"][first:])

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    summary = pd.DataFrame(rows).T
    summary.to_csv(out / "summary.csv", float_format="%.4f")

    rets = pd.DataFrame(rets)
    worst = r["equity"][first:].nsmallest(6).index.sort_values()
    worst_months = rets.loc[worst].assign(SPY=r["equity"].loc[worst])
    worst_months.to_csv(out / "worst_months.csv", float_format="%.4f")

    figs = ROOT / "figures"
    figs.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    (1 + rets).cumprod().plot(ax=ax, logy=True)
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_xlabel("")
    ax.set_title("Net of 10 bp per unit of turnover, signals lagged one month")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "cumulative.png", dpi=120)

    fig, ax = plt.subplots(figsize=(10, 4))
    nav = (1 + rets).cumprod()
    (nav / nav.cummax() - 1).plot(ax=ax)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "drawdowns.png", dpi=120)

    pd.set_option("display.float_format", "{:.4f}".format)
    print(f"{first:%b %Y} to {r.index[-1]:%b %Y}, {len(rets)} months\n")
    print(summary)
    print("\nSix worst SPY months")
    print(worst_months)


if __name__ == "__main__":
    main()
