"""
The strategies run forward: each month's allocation, recorded before the month.

    python live.py

In the first days of each month every strategy's weights for that month are
computed from data through the last month end, exactly as in the backtest,
and added to results/live_allocations.csv. Later runs fill in each month's
net return once it has closed. The stop-loss overlay is left out because it
needs the strategy's running value from the start of the sample, which the
live record does not have.
"""
from datetime import date
from pathlib import Path

import pandas as pd

from backtest import (TX_COST, defensive_switch_weights, momentum_vol_target_weights,
                      static_weights)

ROOT = Path(__file__).parent
LEDGER = ROOT / "results" / "live_allocations.csv"
README = ROOT / "README.md"
START, END = "<!-- live:start -->", "<!-- live:end -->"
PICK_WINDOW_DAYS = 10
STRATEGIES = {"Static 60/20/20": static_weights,
              "Defensive switch": defensive_switch_weights,
              "Momentum, vol target": momentum_vol_target_weights}


def monthly_returns():
    """Completed months only, indexed by the first of the month like the
    backtest, plus an empty row for the current month."""
    import yfinance as yf
    px = yf.download(["SPY", "AGG"], start="2003-08-01", auto_adjust=True,
                     progress=False)["Close"]
    this_month = pd.Timestamp(date.today()).to_period("M")
    month_end = px.groupby(px.index.to_period("M")).tail(1)
    month_end.index = month_end.index.to_period("M")
    rets = month_end.pct_change().dropna()
    rets = rets[rets.index < this_month]
    rets.index = rets.index.to_timestamp()

    tb = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS",
                     index_col=0, parse_dates=True)["TB3MS"]
    idx = rets.index.append(pd.DatetimeIndex([this_month.to_timestamp()]))
    tbill = (tb.shift(1, freq="MS") / 1200).reindex(idx)
    r = pd.DataFrame({"equity": rets["SPY"], "bonds": rets["AGG"]}).reindex(idx)
    r["tbills"] = tbill
    return r


def allocations(r):
    month = r.index[-1]
    return {name: f(r).loc[month] for name, f in STRATEGIES.items()}, month


def score(ledger, r):
    for i, row in ledger[ledger["net_return"].isna()].iterrows():
        m = pd.Timestamp(row["month"] + "-01")
        if m not in r.index or r.loc[m, ["equity", "bonds"]].isna().any():
            continue
        w = row[["equity", "bonds", "tbills"]].astype(float)
        prev = ledger[(ledger.strategy == row.strategy) & (ledger.month < row.month)]
        turnover = (w - prev.iloc[-1][["equity", "bonds", "tbills"]].astype(float)).abs().sum() \
            if len(prev) else 0.0
        ledger.loc[i, "net_return"] = float((r.loc[m, w.index] * w).sum() - TX_COST * turnover)
    return ledger


def write_section(lines):
    text = README.read_text(encoding="utf-8")
    before, rest = text.split(START)
    README.write_text(before + START + "\n" + "\n".join(lines) + "\n" + END + rest.split(END)[1],
                      encoding="utf-8")


def update_readme(ledger):
    if ledger.empty:
        nxt = (pd.Timestamp(date.today()).to_period("M") + 1).strftime("%B %Y")
        write_section([f"The first allocations will be recorded in the first days of {nxt}."])
        return
    lines = ["| Month | Strategy | Equity | Bonds | T-bills | Net return |", "|---|---|---|---|---|---|"]
    for _, r in ledger.sort_values(["month", "strategy"], ascending=[False, True]).iterrows():
        ret = f"{r.net_return:+.2%}" if pd.notna(r.net_return) else "pending"
        lines.append(f"| {r.month} | {r.strategy} | {r.equity:.0%} | {r.bonds:.0%} | {r.tbills:.0%} | {ret} |")
    done = ledger.dropna(subset=["net_return"])
    if len(done):
        grow = done.groupby("strategy")["net_return"].apply(lambda s: (1 + s).prod() - 1)
        n = done["month"].nunique()
        lines += ["", f"Over {n} completed month{'s' if n > 1 else ''}: "
                  + ", ".join(f"{k} {v:+.1%}" for k, v in grow.items()) + "."]
    write_section(lines)


def main():
    r = monthly_returns()
    cols = ["month", "strategy", "equity", "bonds", "tbills", "net_return"]
    ledger = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame(columns=cols)
    ledger = score(ledger, r)

    weights, month = allocations(r)
    label = month.strftime("%Y-%m")
    if label not in ledger["month"].astype(str).tolist() and date.today().day <= PICK_WINDOW_DAYS:
        new = pd.DataFrame([{"month": label, "strategy": k, **w.round(4).to_dict()}
                            for k, w in weights.items()])
        ledger = new if ledger.empty else pd.concat([ledger, new], ignore_index=True)
        print(new.to_string(index=False))
    ledger.to_csv(LEDGER, index=False)
    update_readme(ledger)


if __name__ == "__main__":
    main()
