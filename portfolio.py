"""Portfolio maths: positions (average cost), XIRR, TWR and daily value series."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from parsers import BONUS, BUY, DEPOSIT, DIVIDEND, FEE, INTEREST, SELL, TAX, WITHDRAWAL

EPS = 1e-9


def xirr(dates, amounts) -> float:
    """Money-weighted annual return. amounts from investor view (in = negative)."""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    a = pd.Series(amounts, dtype=float).reset_index(drop=True)
    mask = a.abs() > 1e-9
    d, a = d[mask], a[mask]
    if len(a) < 2 or (a > 0).all() or (a < 0).all():
        return np.nan
    t = (d - d.min()).dt.days.to_numpy() / 365.25
    v = a.to_numpy()
    f = lambda r: np.sum(v / (1 + r) ** t)
    try:
        return brentq(f, -0.9999, 100)
    except ValueError:
        return np.nan


def positions(ev: pd.DataFrame, price_now: dict[str, float], today=None) -> pd.DataFrame:
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    rows = []
    sec = ev[ev["type"].isin([BUY, SELL, DIVIDEND])].sort_values("date")
    for (acc, key), g in sec.groupby(["account", "key"], sort=False):
        shares = cost = realized = divs = tax = fees = bought = 0.0
        for r in g.itertuples():
            if r.type == BUY:
                shares += r.shares
                cost += -r.amount
                bought += -r.amount
                fees += r.fee
            elif r.type == SELL:
                q = min(r.shares, shares) if shares > EPS else r.shares
                basis = cost / shares * q if shares > EPS else 0.0
                realized += r.amount - basis
                cost -= basis
                shares -= q
                fees += r.fee
                if shares < 1e-7:
                    shares, cost = 0.0, 0.0
            else:
                divs += r.amount
                tax += r.tax
        price = price_now.get(key, np.nan)
        value = shares * price if shares > EPS and not np.isnan(price) else (0.0 if shares <= EPS else np.nan)
        unreal = value - cost if not np.isnan(value) else np.nan
        flows = list(g["amount"])
        dates = list(g["date"])
        if shares > EPS and not np.isnan(value):
            flows.append(value)
            dates.append(today)
        names = g["name"].replace("", np.nan).dropna()
        rows.append(dict(
            account=acc, key=key, name=names.iloc[-1] if len(names) else key,
            shares=shares, avg_price=cost / shares if shares > EPS else np.nan,
            cost=cost, price=price, value=value, unrealized=unreal,
            unrealized_pct=unreal / cost if cost > EPS and not np.isnan(unreal) else np.nan,
            realized=realized, dividends=divs, dividend_tax=tax, fees=fees, bought=bought,
            total_return=(0 if np.isnan(unreal) else unreal) + realized + divs,
            xirr=xirr(dates, flows), first_buy=g["date"].min(),
        ))
    return pd.DataFrame(rows)


def combine_positions(pos: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the same security over all accounts."""
    if pos.empty:
        return pos
    agg = pos.groupby("key").agg(
        name=("name", "first"), accounts=("account", lambda s: ", ".join(sorted(set(s)))),
        shares=("shares", "sum"), cost=("cost", "sum"), price=("price", "first"),
        value=("value", "sum"), realized=("realized", "sum"), dividends=("dividends", "sum"),
        fees=("fees", "sum"), bought=("bought", "sum"),
    ).reset_index()
    agg["avg_price"] = np.where(agg["shares"] > EPS, agg["cost"] / agg["shares"].where(agg["shares"] > EPS), np.nan)
    agg["unrealized"] = agg["value"] - agg["cost"]
    agg["unrealized_pct"] = agg["unrealized"] / agg["cost"].where(agg["cost"] > EPS)
    agg["total_return"] = agg["unrealized"].fillna(0) + agg["realized"] + agg["dividends"]
    return agg


def daily_series(ev: pd.DataFrame, prices: pd.DataFrame, ticker_of: dict[str, str],
                 manual: dict[str, float] | None = None) -> pd.DataFrame:
    """Per day: holdings value, cash, external flows, net purchases, dividends."""
    manual = manual or {}
    start = ev["date"].min().normalize()
    idx = pd.date_range(start, pd.Timestamp.today().normalize(), freq="D")
    day = ev.assign(day=ev["date"].dt.normalize())

    trades = day[day["type"].isin([BUY, SELL])].copy()
    trades["dq"] = np.where(trades["type"] == BUY, trades["shares"], -trades["shares"])
    q = trades.pivot_table(index="day", columns="key", values="dq", aggfunc="sum").reindex(idx).fillna(0).cumsum()
    q = q.where(q.abs() > 1e-7, 0.0)

    # fallback valuation for unpriced securities: running cost of purchases
    cost_flow = trades.pivot_table(index="day", columns="key", values="amount", aggfunc="sum").reindex(idx).fillna(0)
    holdings = pd.Series(0.0, index=idx)
    for key in q.columns:
        t = ticker_of.get(key)
        if key in manual and manual[key]:
            px = pd.Series(manual[key], index=idx)
        elif t and not prices.empty and t in prices.columns:
            px = prices[t].reindex(idx).ffill().bfill()
        else:
            px = None
        if px is not None and px.notna().any():
            holdings += (q[key] * px).fillna(0)
        else:
            holdings += (-cost_flow[key].cumsum()).where(q[key] > 0, 0.0)

    def daily_sum(types):
        s = day[day["type"].isin(types)].groupby("day")["amount"].sum()
        return s.reindex(idx).fillna(0.0)

    out = pd.DataFrame(index=idx)
    out["holdings"] = holdings
    out["cash"] = day.groupby("day")["amount"].sum().reindex(idx).fillna(0).cumsum()
    out["total"] = out["holdings"] + out["cash"]
    out["ext_flow"] = daily_sum([DEPOSIT, WITHDRAWAL])
    out["net_deposits"] = out["ext_flow"].cumsum()
    out["purchases"] = -daily_sum([BUY, SELL])
    out["dividends"] = daily_sum([DIVIDEND])
    out["invested"] = out["purchases"].cumsum()
    return out


def twr(values: pd.Series, flows: pd.Series, income: pd.Series | None = None) -> pd.Series:
    """Cumulative time-weighted return. flows = external money in (+) on each day (end of day)."""
    income = income if income is not None else pd.Series(0.0, index=values.index)
    prev = values.shift(1)
    r = (values + income - flows) / prev - 1
    r = r.where(prev > 1.0, 0.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return (1 + r).cumprod() - 1


def annualize(cum: float, days: int) -> float:
    if days < 365 or cum <= -1:
        return np.nan
    return (1 + cum) ** (365.25 / days) - 1


def account_summary(ev: pd.DataFrame, pos: pd.DataFrame, today=None) -> dict:
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    s = lambda types: ev.loc[ev["type"].isin(types), "amount"].sum()
    holdings = pos["value"].fillna(pos["cost"]).sum() if not pos.empty else 0.0
    cash = ev["amount"].sum()
    deposits, withdrawals = s([DEPOSIT]), s([WITHDRAWAL])
    total = holdings + cash
    ext = ev[ev["type"].isin([DEPOSIT, WITHDRAWAL])]
    acc_xirr = xirr(list(ext["date"]) + [today], list(-ext["amount"]) + [total])
    inv = ev[ev["type"].isin([BUY, SELL, DIVIDEND])]
    inv_xirr = xirr(list(inv["date"]) + [today], list(inv["amount"]) + [holdings])
    return dict(
        deposits=deposits, withdrawals=-withdrawals, net_deposits=deposits + withdrawals,
        holdings=holdings, cash=cash, total=total,
        cost=pos["cost"].sum() if not pos.empty else 0.0,
        unrealized=pos["unrealized"].sum() if not pos.empty else 0.0,
        realized=pos["realized"].sum() if not pos.empty else 0.0,
        dividends=pos["dividends"].sum() if not pos.empty else 0.0,
        dividend_tax=pos["dividend_tax"].sum() if not pos.empty else 0.0,
        interest=s([INTEREST]), bonus=s([BONUS]), other_fees=-s([FEE, TAX]),
        trade_fees=pos["fees"].sum() if not pos.empty else 0.0,
        result=total - (deposits + withdrawals),
        invest_result=(pos["unrealized"].fillna(0).sum() + pos["realized"].sum() + pos["dividends"].sum()) if not pos.empty else 0.0,
        account_xirr=acc_xirr, invest_xirr=inv_xirr,
        first=ev["date"].min(),
    )
