"""Parsers that turn broker exports into one normalized event table.

Every event has a signed ``amount`` in EUR = the effect on the cash balance
(buy = negative, sell/dividend/deposit = positive), fees and taxes included.
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

COLUMNS = ["date", "account", "broker", "type", "key", "isin", "name",
           "shares", "amount", "fee", "tax", "note"]

# Event types
BUY, SELL, DIVIDEND = "BUY", "SELL", "DIVIDEND"
INTEREST, BONUS, FEE, TAX = "INTEREST", "BONUS", "FEE", "TAX"
DEPOSIT, WITHDRAWAL = "DEPOSIT", "WITHDRAWAL"

CRYPTO_NAMES = {
    "BITCOIN": "BTC", "ETHEREUM": "ETH", "XRP": "XRP", "CARDANO": "ADA",
    "SOLANA": "SOL", "DOGECOIN": "DOGE", "LITECOIN": "LTC", "POLKADOT": "DOT",
    "CHAINLINK": "LINK", "AVALANCHE": "AVAX", "POLYGON": "MATIC",
}


def _nl_num(s: pd.Series) -> pd.Series:
    """'1.819,47' -> 1819.47"""
    s = s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _nl_scalar(x: str) -> float:
    return float(x.replace(".", "").replace(",", "."))


def _key(isin, name) -> str:
    if isinstance(isin, str) and isin.strip():
        return isin.strip()
    n = str(name).strip().upper()
    if n in CRYPTO_NAMES:
        return f"CRYPTO:{CRYPTO_NAMES[n]}"
    return f"NAME:{n}"


def _ev(**kw) -> dict:
    base = dict(shares=0.0, fee=0.0, tax=0.0, isin="", name="", key="", note="")
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def detect_broker(df: pd.DataFrame) -> str | None:
    cols = {str(c).strip() for c in df.columns}
    if {"transaction_id", "category", "type", "symbol", "amount"} <= cols:
        return "Trade Republic"
    if {"Action", "Time (UTC)", "ISIN"} <= cols:
        return "Trading 212"
    if ("Order Id" in cols or "Order ID" in cols) and (
            "Omschrijving" in cols or "Description" in cols):
        return "DEGIRO"
    return None


def read_any(data: bytes, filename: str, account: str | None = None) -> tuple[str, pd.DataFrame]:
    text = data.decode("utf-8-sig", errors="replace")
    sep = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
    raw = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
    broker = detect_broker(raw)
    if broker is None:
        raise ValueError(f"Onbekend bestandsformaat: {filename}")
    account = account or broker
    parser = {"Trade Republic": parse_trade_republic,
              "Trading 212": parse_trading212,
              "DEGIRO": parse_degiro}[broker]
    ev = parser(raw)
    ev["account"] = account
    ev["broker"] = broker
    return broker, ev[COLUMNS].sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Trade Republic
# --------------------------------------------------------------------------- #
def parse_trade_republic(df: pd.DataFrame) -> pd.DataFrame:
    num = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    amount, fee, tax, shares = num("amount"), num("fee"), num("tax"), num("shares")
    dates = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.tz_convert(None)
    out = []
    for i, r in df.iterrows():
        t = r["type"].upper()
        cat = r["category"].upper()
        cash = amount[i] + fee[i] + tax[i]
        common = dict(date=dates[i], fee=-fee[i], tax=-tax[i])
        isin, name = r["symbol"], r["name"]
        if cat == "TRADING" and t in ("BUY", "SELL"):
            out.append(_ev(type=BUY if t == "BUY" else SELL, key=_key(isin, name), isin=isin,
                           name=name, shares=abs(shares[i]), amount=cash, **common))
        elif "DIVIDEND" in t or "DISTRIBUTION" in t:
            out.append(_ev(type=DIVIDEND, key=_key(isin, name), isin=isin, name=name,
                           amount=cash, **common))
        elif "INTEREST" in t:
            out.append(_ev(type=INTEREST, amount=cash, note=t, **common))
        elif "SAVEBACK" in t or "REFERRAL" in t or "BONUS" in t or "REWARD" in t:
            out.append(_ev(type=BONUS, amount=cash, note=t, **common))
        elif t.startswith("TAX"):
            out.append(_ev(type=TAX, amount=cash, note=t, **common))
        elif "FEE" in t:
            out.append(_ev(type=FEE, amount=cash, note=t, **common))
        elif cash != 0:
            note = "Kaarttransactie" if t.startswith("CARD") else t
            out.append(_ev(type=DEPOSIT if cash > 0 else WITHDRAWAL, amount=cash, note=note, **common))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Trading 212
# --------------------------------------------------------------------------- #
def parse_trading212(df: pd.DataFrame) -> pd.DataFrame:
    def num(c):
        if c not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    total_col = "Total" if "Total" in df.columns else "Gross Total"
    total = num(total_col)
    shares, price = num("No. of shares"), num("Price / share")
    wht = num("Withholding tax")
    fee_cols = [c for c in df.columns if ("fee" in c.lower() or "tax" in c.lower())
                and not c.lower().startswith("currency") and c != "Withholding tax"]
    dates = pd.to_datetime(df["Time (UTC)"].str.slice(0, 19), errors="coerce")
    out = []
    for i, r in df.iterrows():
        a = r["Action"].lower()
        isin, name = r.get("ISIN", ""), r.get("Name", "")
        fee = float(sum(num(c)[i] for c in fee_cols))
        common = dict(date=dates[i])
        if "buy" in a:
            out.append(_ev(type=BUY, key=_key(isin, name), isin=isin, name=name,
                           shares=shares[i], amount=-abs(total[i]), fee=fee, **common))
        elif "sell" in a:
            out.append(_ev(type=SELL, key=_key(isin, name), isin=isin, name=name,
                           shares=shares[i], amount=abs(total[i]), fee=fee, **common))
        elif "dividend" in a:
            tax = wht[i]
            if tax and total_col == "Gross Total":
                # withholding tax is in the security currency: convert with the row's implied rate
                gross_foreign = shares[i] * price[i]
                if r.get("Currency (Withholding tax)", "") != r.get(f"Currency ({total_col})", "EUR") and gross_foreign:
                    tax = tax * total[i] / gross_foreign
                net = total[i] - tax
            else:
                net = total[i]
            out.append(_ev(type=DIVIDEND, key=_key(isin, name), isin=isin, name=name,
                           amount=net, tax=tax, **common))
        elif "interest" in a:
            out.append(_ev(type=INTEREST, amount=total[i], note=r["Action"], **common))
        elif a == "deposit":
            notes = r.get("Notes", "").lower()
            is_bonus = any(w in notes for w in ("free share", "promotion", "bonus", "reward", "cashback"))
            out.append(_ev(type=BONUS if is_bonus else DEPOSIT, amount=total[i],
                           note=r.get("Notes", ""), **common))
        elif a == "withdrawal":
            out.append(_ev(type=WITHDRAWAL, amount=-abs(total[i]), note=r.get("Notes", ""), **common))
        elif "cashback" in a or "promotion" in a:
            out.append(_ev(type=BONUS, amount=total[i], note=r["Action"], **common))
        elif "fee" in a:
            out.append(_ev(type=FEE, amount=-abs(total[i]), note=r["Action"], **common))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# DEGIRO (Rekeningoverzicht / Account.csv)
# --------------------------------------------------------------------------- #
_TRADE_RE = re.compile(r"^(Koop|Verkoop|Buy|Sell)\s+([\d\.,]+)\s*@\s*([\d\.,]+)\s*([A-Z]{3})", re.I)


def parse_degiro(df: pd.DataFrame) -> pd.DataFrame:
    c = list(df.columns)
    d = pd.DataFrame({
        "date": pd.to_datetime(df[c[0]] + " " + df[c[1]], format="%d-%m-%Y %H:%M", errors="coerce"),
        "product": df[c[3]].str.strip(), "isin": df[c[4]].str.strip(),
        "desc": df[c[5]].str.strip(), "fx": _nl_num(df[c[6]]),
        "ccy": df[c[7]].str.strip(), "amt": _nl_num(df[c[8]]),
        "oid": df[c[11]].str.strip() if len(c) > 11 else "",
    })
    d = d[d["amt"].notna()].copy()
    desc_l = d["desc"].str.lower()
    is_fx = desc_l.str.startswith(("valuta ", "fx ", "valutadebitering", "valutacreditering"))

    # FX-rates (foreign per EUR) known from currency conversions, used for rows without own EUR leg
    fx_rows = d[is_fx & d["fx"].notna()][["date", "ccy", "fx"]].sort_values("date")

    def to_eur(amount, ccy, date):
        if ccy == "EUR" or not ccy:
            return amount
        rows = fx_rows[fx_rows["ccy"] == ccy]
        if rows.empty:
            return amount
        idx = (rows["date"] - date).abs().idxmin()
        return amount / rows.loc[idx, "fx"]

    # EUR legs of currency conversions per order id
    eur_leg = d[is_fx & (d["ccy"] == "EUR") & (d["oid"] != "")].groupby("oid")["amt"].sum()

    out = []
    trades = d[d["desc"].str.match(_TRADE_RE)]
    fee_rows = d[desc_l.str.contains("transactiekosten|transaction and/or third|transaction fee|kosten van derden")]
    fees_by_oid = fee_rows[fee_rows["oid"] != ""].groupby("oid")["amt"].sum()
    trade_oids = set(trades["oid"])

    for oid, g in trades.groupby("oid", sort=False):
        fee = float(fees_by_oid.get(oid, 0.0))  # negative
        foreign = g[g["ccy"] != "EUR"]
        if foreign.empty:
            eur_total = g["amt"].sum()
        elif oid in eur_leg.index:
            eur_total = g[g["ccy"] == "EUR"]["amt"].sum() + eur_leg[oid]
        else:
            eur_total = sum(to_eur(r.amt, r.ccy, r.date) for r in g.itertuples())
        tot_shares = 0.0
        for r in g.itertuples():
            m = _TRADE_RE.match(r.desc)
            tot_shares += _nl_scalar(m.group(2))
        is_buy = m.group(1).lower() in ("koop", "buy")
        r0 = g.iloc[0]
        out.append(_ev(date=g["date"].min(), type=BUY if is_buy else SELL,
                       key=_key(r0["isin"], r0["product"]), isin=r0["isin"], name=r0["product"],
                       shares=tot_shares, amount=eur_total + fee, fee=-fee))

    for r in d.itertuples():
        dl = r.desc.lower()
        if r.oid in trade_oids or _TRADE_RE.match(r.desc) or dl.startswith(("valuta ", "fx ")):
            continue
        eur = to_eur(r.amt, r.ccy, r.date)
        common = dict(date=r.date, amount=eur, note=r.desc)
        if dl.startswith(("dividendbelasting", "dividend tax")):
            out.append(_ev(type=DIVIDEND, key=_key(r.isin, r.product), isin=r.isin,
                           name=r.product, tax=-eur, **common))
        elif dl.startswith(("dividend", "kapitaalsuitkering", "capital return")):
            out.append(_ev(type=DIVIDEND, key=_key(r.isin, r.product), isin=r.isin,
                           name=r.product, **common))
        elif "interest" in dl or "rente" in dl or "geldmarktfonds" in dl:
            out.append(_ev(type=INTEREST, **common))
        elif "transactiekosten" in dl or "kosten" in dl or "fee" in dl or "aansluiting" in dl:
            out.append(_ev(type=FEE, fee=-eur, **common))
        elif "deposit" in dl or "storting" == dl or dl.startswith("ideal storting"):
            if "reservation" not in dl:
                out.append(_ev(type=DEPOSIT, **common))
        elif "terugstorting" in dl or dl.startswith("withdrawal"):
            out.append(_ev(type=WITHDRAWAL, **common))
        # ignored: reservations, cash sweep, flatex internal transfers, processed withdrawals
    return pd.DataFrame(out)
