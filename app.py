"""Beleggingsdashboard - start met:  streamlit run app.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import parsers
import portfolio as pf
import prices

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
MANUAL = HERE / "manual_prices.json"

st.set_page_config(page_title="Beleggingsdashboard", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def eur(x, dec=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"€ {s}"


def pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x * 100:+.2f}%".replace(".", ",")


def load_manual() -> dict:
    return json.loads(MANUAL.read_text()) if MANUAL.exists() else {}


def meta_path(p: Path) -> Path:
    return p.with_suffix(".json")


@st.cache_data(show_spinner=False)
def parse_file(data: bytes, name: str, account: str):
    return parsers.read_any(data, name, account)


@st.cache_data(show_spinner="Koersen ophalen via Yahoo Finance…", ttl=3600)
def get_prices(tickers: tuple, start: str, day: str):
    return prices.history_eur(list(tickers), pd.Timestamp(start))


MONEY = lambda label: st.column_config.NumberColumn(label, format="€ %.2f")
PCT = lambda label: st.column_config.NumberColumn(label, format="%.2f%%")


# --------------------------------------------------------------------------- #
# Sidebar: uploads
# --------------------------------------------------------------------------- #
st.sidebar.title("📁 Bestanden")
st.sidebar.caption("Ondersteund: **Trade Republic** (Transactie-export), **DEGIRO** (Rekeningoverzicht / "
                   "Account.csv) en **Trading 212** (History export). Een nieuwe upload voor een bestaand "
                   "account vervangt het oude bestand.")
uploads = st.sidebar.file_uploader("Upload CSV-exports", type=["csv"], accept_multiple_files=True)
for up in uploads or []:
    data = up.getvalue()
    try:
        text = data.decode("utf-8-sig", errors="replace")
        sep = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
        head = pd.read_csv(pd.io.common.StringIO(text), sep=sep, nrows=2, dtype=str)
        broker = parsers.detect_broker(head)
    except Exception:
        broker = None
    if broker is None:
        st.sidebar.error(f"{up.name}: formaat niet herkend")
        continue
    with st.sidebar.form(f"form_{up.file_id}"):
        st.write(f"**{up.name}** → {broker}")
        acc = st.text_input("Accountnaam", value=broker, key=f"acc_{up.file_id}")
        if st.form_submit_button("Opslaan"):
            safe = "".join(ch for ch in acc if ch.isalnum() or ch in " -_").strip() or broker
            (DATA / f"{safe}.csv").write_bytes(data)
            meta_path(DATA / f"{safe}.csv").write_text(json.dumps({"account": acc, "broker": broker,
                                                                     "file": up.name}))
            st.success(f"Opgeslagen als account '{acc}'")

files = sorted(DATA.glob("*.csv"))
st.sidebar.subheader("Accounts")
if not files:
    st.sidebar.info("Nog geen bestanden opgeslagen.")
for f in files:
    meta = json.loads(meta_path(f).read_text()) if meta_path(f).exists() else {"account": f.stem}
    c1, c2 = st.sidebar.columns([4, 1])
    c1.write(f"**{meta['account']}**  \n<small>{meta.get('file', f.name)}</small>", unsafe_allow_html=True)
    if c2.button("🗑️", key=f"del_{f.name}"):
        f.unlink()
        meta_path(f).unlink(missing_ok=True)
        st.rerun()

if st.sidebar.button("🔄 Koersen verversen"):
    get_prices.clear()
    st.rerun()

if not files:
    st.title("📈 Beleggingsdashboard")
    st.info("Upload links je exports en klik op **Opslaan** om te beginnen.")
    st.stop()

# --------------------------------------------------------------------------- #
# Parse & price
# --------------------------------------------------------------------------- #
events = []
for f in files:
    meta = json.loads(meta_path(f).read_text()) if meta_path(f).exists() else {"account": f.stem}
    try:
        _, ev = parse_file(f.read_bytes(), f.name, meta["account"])
        events.append(ev)
    except Exception as e:
        st.error(f"Fout bij inlezen {f.name}: {e}")
ev = pd.concat(events, ignore_index=True).sort_values("date")
ev = ev[ev["date"].notna()]

sec = ev[ev["type"].isin([parsers.BUY, parsers.SELL, parsers.DIVIDEND])]
names = sec.groupby("key")["name"].agg(lambda s: next((x for x in s[::-1] if x), s.iloc[-1])).to_dict()
ticker_of = prices.resolve(names)
manual = {k: v for k, v in load_manual().items() if v}

price_df, ccys = get_prices(tuple(sorted(set(ticker_of.values()) - {""})),
                            str(ev["date"].min().date()), str(pd.Timestamp.today().date()))
price_now = {}
for k, t in ticker_of.items():
    if k in manual:
        price_now[k] = float(manual[k])
    elif t and t in price_df.columns and price_df[t].notna().any():
        price_now[k] = float(price_df[t].dropna().iloc[-1])
    else:
        price_now[k] = np.nan

accounts = sorted(ev["account"].unique())
pos_all = pf.positions(ev, price_now)

# --------------------------------------------------------------------------- #
# Header / filter
# --------------------------------------------------------------------------- #
st.title("📈 Beleggingsdashboard")
sel = st.multiselect("Accounts", accounts, default=accounts)
if not sel:
    st.stop()
evs = ev[ev["account"].isin(sel)]
pos = pos_all[pos_all["account"].isin(sel)]
summ = pf.account_summary(evs, pos)
series = pf.daily_series(evs, price_df, ticker_of, manual)
twr_inv = pf.twr(series["holdings"], series["purchases"], series["dividends"])
twr_acc = pf.twr(series["total"], series["ext_flow"])
days = (series.index[-1] - series.index[0]).days

missing = [names[k] for k, p in price_now.items() if np.isnan(p) and k in set(pos["key"])
           and pos.loc[pos["key"] == k, "shares"].sum() > 0]
if missing:
    st.warning("Geen koers gevonden voor: " + ", ".join(missing) +
               ". Deze worden gewaardeerd tegen kostprijs. Pas de ticker aan in het tabblad **Tickers**.")

k = st.columns(6)
k[0].metric("Totale waarde", eur(summ["total"]), help="Posities + cash")
k[1].metric("Waarde posities", eur(summ["holdings"]))
k[2].metric("Netto gestort", eur(summ["net_deposits"]), help="Stortingen − opnames (incl. kaartbetalingen TR)")
k[3].metric("Beleggingsresultaat", eur(summ["invest_result"]),
            pct(summ["invest_result"] / summ["cost"]) if summ["cost"] else None,
            help="Ongerealiseerd + gerealiseerd + netto dividend (na kosten). % t.o.v. huidige kostprijs.")
k[4].metric("Rendement (TWR)", pct(twr_inv.iloc[-1]),
            f"{pct(pf.annualize(twr_inv.iloc[-1], days))} p.j." if days >= 365 else None,
            help="Time-weighted return van de beleggingen: onafhankelijk van timing/grootte van stortingen.")
k[5].metric("Rendement (IRR p.j.)", pct(summ["invest_xirr"]),
            help="Money-weighted (XIRR) op basis van alle aan- en verkopen, dividenden en huidige waarde.")

tabs = st.tabs(["Overzicht", "Posities", "Per account", "Dividend & inkomsten", "Transacties", "Tickers"])

# --------------------------------------------------------------------------- #
with tabs[0]:
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=series.index, y=series["holdings"], name="Waarde posities", fill="tozeroy")
        fig.add_scatter(x=series.index, y=series["invested"], name="Netto ingelegd in posities",
                        line=dict(dash="dash"))
        fig.update_layout(title="Waarde beleggingen vs. inleg", height=420, hovermode="x unified",
                          yaxis_tickprefix="€ ", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=twr_inv.index, y=twr_inv * 100, name="Beleggingen (TWR)")
        fig.add_scatter(x=twr_acc.index, y=twr_acc * 100, name="Account incl. cash (TWR)")
        fig.update_layout(title="Cumulatief rendement %", height=420, hovermode="x unified",
                          yaxis_ticksuffix="%", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")

    rows = []
    for a in sel:
        e_a, p_a = ev[ev["account"] == a], pos_all[pos_all["account"] == a]
        s_a = pf.account_summary(e_a, p_a)
        ser = pf.daily_series(e_a, price_df, ticker_of, manual)
        t_a = pf.twr(ser["holdings"], ser["purchases"], ser["dividends"]).iloc[-1]
        rows.append({"Account": a, "Waarde posities": s_a["holdings"], "Cash": s_a["cash"],
                     "Totaal": s_a["total"], "Kostprijs": s_a["cost"],
                     "Ongerealiseerd": s_a["unrealized"], "Gerealiseerd": s_a["realized"],
                     "Dividend (netto)": s_a["dividends"], "Rente": s_a["interest"], "Bonus": s_a["bonus"],
                     "Beleggingsresultaat": s_a["invest_result"],
                     "Resultaat %": s_a["invest_result"] / s_a["cost"] * 100 if s_a["cost"] else np.nan,
                     "TWR %": t_a * 100, "IRR p.j. %": s_a["invest_xirr"] * 100})
    df = pd.DataFrame(rows)
    st.subheader("Per account")
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        **{c: MONEY(c) for c in df.columns if c not in ("Account", "Resultaat %", "TWR %", "IRR p.j. %")},
        "Resultaat %": PCT("Resultaat %"), "TWR %": PCT("TWR %"), "IRR p.j. %": PCT("IRR p.j. %")})

    st.caption(f"Kosten: transactiekosten {eur(summ['trade_fees'])} (zitten in kostprijs/resultaat), "
               f"overige kosten {eur(summ['other_fees'])}, ingehouden dividendbelasting "
               f"{eur(summ['dividend_tax'])}. Rente {eur(summ['interest'])}, bonussen/saveback "
               f"{eur(summ['bonus'])}. Totaal accountresultaat (waarde − netto gestort): "
               f"{eur(summ['result'])} · IRR incl. cash {pct(summ['account_xirr'])} p.j.")

# --------------------------------------------------------------------------- #
with tabs[1]:
    combine = st.toggle("Zelfde aandeel over accounts samenvoegen", value=True)
    show_closed = st.toggle("Toon gesloten posities", value=False)
    view = pf.combine_positions(pos) if combine else pos.copy()
    if not show_closed:
        view = view[view["shares"] > 1e-7]
    view = view.assign(ticker=view["key"].map(ticker_of),
                       weight=view["value"] / view["value"].sum() * 100,
                       unrealized_pct=view["unrealized_pct"] * 100,
                       total_pct=view["total_return"] / view["cost"].where(view["cost"] > 0) * 100)
    cols = ["name", "ticker", "key"] + (["accounts"] if combine else ["account"]) + [
        "shares", "avg_price", "price", "cost", "value", "weight", "unrealized", "unrealized_pct",
        "realized", "dividends", "total_return", "total_pct"] + ([] if combine else ["xirr"])
    if not combine:
        view["xirr"] = view["xirr"] * 100
    view = view.sort_values("value", ascending=False)
    st.dataframe(view[cols], hide_index=True, width="stretch", height=600, column_config={
        "name": "Naam", "ticker": "Ticker", "key": "ISIN", "accounts": "Accounts", "account": "Account",
        "shares": st.column_config.NumberColumn("Aantal", format="%.6f"),
        "avg_price": MONEY("GAK"), "price": MONEY("Koers"), "cost": MONEY("Kostprijs"),
        "value": MONEY("Waarde"), "weight": PCT("Weging"), "unrealized": MONEY("Ongerealiseerd"),
        "unrealized_pct": PCT("Ongereal. %"), "realized": MONEY("Gerealiseerd"),
        "dividends": MONEY("Dividend"), "total_return": MONEY("Totaal resultaat"),
        "total_pct": PCT("Totaal %"), "xirr": PCT("IRR p.j."),
    })
    st.caption("GAK = gemiddelde aankoopkoers incl. transactiekosten. IRR p.j. is geannualiseerd en kan bij "
               "posities die korter dan enkele maanden lopen extreem uitvallen.")
    c1, c2 = st.columns(2)
    open_pos = view[view["value"] > 0]
    c1.plotly_chart(px.pie(open_pos, values="value", names="name", hole=0.45, title="Verdeling"),
                    width="stretch")
    bar = view.sort_values("total_return")
    c2.plotly_chart(px.bar(bar, x="total_return", y="name", orientation="h", title="Totaal resultaat per positie",
                           color=bar["total_return"] > 0, color_discrete_map={True: "#2ca02c", False: "#d62728"},
                           height=max(400, 18 * len(bar))).update_layout(showlegend=False, xaxis_tickprefix="€ ",
                                                                         yaxis_title=None),
                    width="stretch")

# --------------------------------------------------------------------------- #
with tabs[2]:
    a = st.selectbox("Account", sel)
    e_a, p_a = ev[ev["account"] == a], pos_all[pos_all["account"] == a]
    s_a = pf.account_summary(e_a, p_a)
    ser = pf.daily_series(e_a, price_df, ticker_of, manual)
    m = st.columns(5)
    m[0].metric("Totale waarde", eur(s_a["total"]))
    m[1].metric("Cash", eur(s_a["cash"]))
    m[2].metric("Netto gestort", eur(s_a["net_deposits"]))
    m[3].metric("Beleggingsresultaat", eur(s_a["invest_result"]))
    m[4].metric("IRR p.j.", pct(s_a["invest_xirr"]))
    fig = go.Figure()
    fig.add_scatter(x=ser.index, y=ser["total"], name="Totale waarde (incl. cash)")
    fig.add_scatter(x=ser.index, y=ser["holdings"], name="Waarde posities")
    fig.add_scatter(x=ser.index, y=ser["net_deposits"], name="Netto gestort", line=dict(dash="dash"))
    fig.update_layout(height=420, hovermode="x unified", yaxis_tickprefix="€ ", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
with tabs[3]:
    inc = evs[evs["type"].isin([parsers.DIVIDEND, parsers.INTEREST, parsers.BONUS])].copy()
    inc["maand"] = inc["date"].dt.to_period("M").dt.to_timestamp()
    inc["soort"] = inc["type"].map({"DIVIDEND": "Dividend", "INTEREST": "Rente", "BONUS": "Bonus/saveback"})
    g = inc.groupby(["maand", "soort"])["amount"].sum().reset_index()
    st.plotly_chart(px.bar(g, x="maand", y="amount", color="soort", title="Inkomsten per maand (netto)")
                    .update_layout(yaxis_tickprefix="€ "), width="stretch")
    dv = inc[inc["type"] == parsers.DIVIDEND]
    yearly = dv.groupby([dv["date"].dt.year.rename("Jaar"), "account"])[["amount", "tax"]].sum().reset_index()
    c1, c2 = st.columns(2)
    c1.subheader("Dividend per jaar")
    c1.dataframe(yearly, hide_index=True, width="stretch",
                 column_config={"amount": MONEY("Netto"), "tax": MONEY("Belasting"), "account": "Account"})
    c2.subheader("Dividend per aandeel")
    per = dv.groupby("name")[["amount", "tax"]].sum().sort_values("amount", ascending=False).reset_index()
    c2.dataframe(per, hide_index=True, width="stretch",
                 column_config={"name": "Naam", "amount": MONEY("Netto"), "tax": MONEY("Belasting")})

# --------------------------------------------------------------------------- #
with tabs[4]:
    types = st.multiselect("Type", sorted(evs["type"].unique()),
                           default=[t for t in sorted(evs["type"].unique()) if t not in ("DEPOSIT", "WITHDRAWAL")])
    q = st.text_input("Zoek (naam/ISIN/notitie)")
    tv = evs[evs["type"].isin(types)]
    if q:
        m_ = tv[["name", "key", "note"]].astype(str).apply(lambda c: c.str.contains(q, case=False)).any(axis=1)
        tv = tv[m_]
    tv = tv.sort_values("date", ascending=False)
    st.dataframe(tv, hide_index=True, width="stretch", height=600, column_config={
        "date": st.column_config.DatetimeColumn("Datum", format="DD-MM-YYYY HH:mm"),
        "amount": MONEY("Bedrag"), "fee": MONEY("Kosten"), "tax": MONEY("Belasting"),
        "shares": st.column_config.NumberColumn("Aantal", format="%.6f")})
    st.download_button("Download als CSV", tv.to_csv(index=False).encode(), "transacties.csv", "text/csv")

# --------------------------------------------------------------------------- #
with tabs[5]:
    st.write("Tickers worden automatisch via ISIN op Yahoo Finance gezocht. Klopt een koers niet? "
             "Vul een andere Yahoo-ticker in (bijv. `IWDA.AS`) of een handmatige koers in euro.")
    tdf = pd.DataFrame({"key": list(names), "name": [names[k] for k in names],
                        "ticker": [ticker_of.get(k, "") for k in names],
                        "currency": [ccys.get(ticker_of.get(k, ""), "") for k in names],
                        "koers_eur": [price_now.get(k) for k in names],
                        "handmatig": [manual.get(k) for k in names]})
    edited = st.data_editor(tdf, hide_index=True, width="stretch", height=600,
                            disabled=["key", "name", "currency", "koers_eur"], column_config={
                                "key": "ISIN / sleutel", "name": "Naam", "ticker": "Yahoo ticker",
                                "currency": "Valuta", "koers_eur": MONEY("Koers (EUR)"),
                                "handmatig": st.column_config.NumberColumn("Handmatige koers €", format="%.4f")})
    if st.button("Opslaan en herberekenen"):
        m_ = prices.load_map()
        m_.update({r.key: (r.ticker or "").strip() for r in edited.itertuples()})
        prices.save_map(m_)
        MANUAL.write_text(json.dumps({r.key: float(r.handmatig) for r in edited.itertuples()
                                      if pd.notna(r.handmatig)}, indent=2))
        get_prices.clear()
        st.rerun()
