from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# ---------- Helpers ----------
def fmt_mn(x): 
    return f"{x:,.2f} mn" if pd.notnull(x) else "—"

def fmt_cur(x):
    return f"{x:,.0f}" if abs(x) < 1_000_000 else f"{x/1_000_000:,.2f} mn"

def last_period_delta(series: pd.Series, days=7):
    s = series.sort_index()
    if len(s) < days + 1: 
        return None
    curr = s.tail(days).sum()
    prev = s.tail(days*2).head(days).sum()
    return curr - prev

PLOTLY_TEMPLATE = "simple_white"

@st.cache_data
def load_all():
    trades   = pd.read_csv(DATA/"processed_trades.csv", parse_dates=["trade_date"])
    summary  = pd.read_csv(DATA/"summary_metrics.csv")
    pnl      = pd.read_csv(DATA/"pnl_daily.csv", parse_dates=["trade_date"]) if (DATA/"pnl_daily.csv").exists() else None
    varm     = pd.read_csv(DATA/"var_metrics.csv") if (DATA/"var_metrics.csv").exists() else None
    cp_expo  = pd.read_csv(DATA/"exposure_by_counterparty.csv") if (DATA/"exposure_by_counterparty.csv").exists() else None
    ac_expo  = pd.read_csv(DATA/"exposure_by_asset.csv") if (DATA/"exposure_by_asset.csv").exists() else None
    bi_path  = DATA/"bi_export.csv"
    return trades, summary, pnl, varm, cp_expo, ac_expo, bi_path

def sidebar_filters(df: pd.DataFrame):
    st.sidebar.header("Filters")
    dmin, dmax = df["trade_date"].min(), df["trade_date"].max()
    date_range = st.sidebar.date_input("Date range", (dmin, dmax))
    cps = st.sidebar.multiselect("Counterparty", sorted(df["counterparty"].unique()))
    acs = st.sidebar.multiselect("Asset class", sorted(df["asset_class"].unique()))
    return date_range, cps, acs

def apply_filters(df, date_range, cps, acs):
    d1, d2 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    m = (df["trade_date"]>=d1) & (df["trade_date"]<=d2)
    if cps: m &= df["counterparty"].isin(cps)
    if acs: m &= df["asset_class"].isin(acs)
    return df.loc[m].copy()

def line(df, x, y, title):
    fig = px.line(df, x=x, y=y, title=title, template=PLOTLY_TEMPLATE)
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), hovermode="x unified")
    fig.update_traces(hovertemplate="%{x}<br>%{y:,.2f}")
    return fig

def bar(df, x, y, title, orientation="v"):
    fig = px.bar(df, x=x, y=y, title=title, template=PLOTLY_TEMPLATE, orientation=orientation)
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), xaxis_title=None, yaxis_title=None)
    fig.update_traces(hovertemplate="%{x}: %{y:,.2f}")
    return fig

# ---------- Page ----------
st.set_page_config(page_title="TradeSphere Risk & Analytics", page_icon="📈", layout="wide")

# Subtle width + card style
st.markdown("""
<style>
    .block-container {max-width: 1200px;}
    .metric-card {background:#ffffff;border:1px solid #e5e7eb;border-radius:14px;padding:16px}
    .section-title {font-weight:700;margin-top:0.5rem;margin-bottom:0.25rem;}
    .small {color:#64748b;font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

st.title("TradeSphere Risk & Analytics Platform")
st.caption(" Treasury & risk — daily exposure, P&L, VaR and BI exports.")

trades, summary, pnl, varm, cp_expo, ac_expo, bi_path = load_all()
date_range, sel_cp, sel_ac = sidebar_filters(trades)
f = apply_filters(trades, date_range, sel_cp, sel_ac)

# ---------- KPIs ----------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Total Exposure</div>', unsafe_allow_html=True)
    st.metric(label="", value=fmt_mn(f["notional_mn"].sum()))
    st.markdown('</div>', unsafe_allow_html=True)

with k2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Total Trades</div>', unsafe_allow_html=True)
    st.metric(label="", value=f"{f['trade_id'].nunique():,}")
    st.markdown('</div>', unsafe_allow_html=True)

with k3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Total P&L</div>', unsafe_allow_html=True)
    delta = None
    if pnl is not None and not pnl.empty:
        _p = pnl.set_index("trade_date")["pnl_total"]
        delta = last_period_delta(_p, days=7)
    st.metric(label="", value=fmt_cur(f["pnl"].sum()),
              delta=(f"{delta:,.0f}" if delta is not None else None))
    st.markdown('</div>', unsafe_allow_html=True)

with k4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Portfolio VaR (95%)</div>', unsafe_allow_html=True)
    var_val = varm["VaR"].iloc[0] if varm is not None and not varm.empty else np.nan
    st.metric(label="", value=fmt_cur(var_val) if pd.notnull(var_val) else "—")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Exposures", "P&L", "Data"])

with tab1:
    st.subheader("Counterparty summary")
    st.dataframe(summary.style.format({"total_exposure_mn":"{:,.2f}", "pnl_total":"{:,.0f}", "pnl_avg":"{:,.0f}"}),
                 use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    if cp_expo is not None and not cp_expo.empty:
        c1.plotly_chart(bar(cp_expo.head(15), "counterparty", "exposure_mn", "Top counterparties (mn)"),
                        use_container_width=True)
    if ac_expo is not None and not ac_expo.empty:
        c2.plotly_chart(bar(ac_expo, "asset_class", "exposure_mn", "Exposure by asset class (mn)"),
                        use_container_width=True)

with tab3:
    if pnl is not None and not pnl.empty:
        p = pnl[(pnl["trade_date"]>=pd.to_datetime(date_range[0])) &
                (pnl["trade_date"]<=pd.to_datetime(date_range[1]))]
        st.plotly_chart(line(p, "trade_date", "pnl_total", "Daily Portfolio P&L"),
                        use_container_width=True)
    else:
        st.info("Run notebooks/Analytics.py to compute daily P&L and VaR.")

with tab4:
    st.write("Filtered trades")
    st.dataframe(f.sort_values("trade_date", ascending=False)
                   .style.format({"notional":"{:,.0f}","notional_mn":"{:,.2f}","pnl":"{:,.0f}"}),
                 use_container_width=True)
    col_dl1, col_dl2 = st.columns(2)
    col_dl1.download_button("Download filtered trades (CSV)",
                            data=f.to_csv(index=False).encode("utf-8"),
                            file_name="trades_filtered.csv")
    if bi_path.exists():
        col_dl2.download_button("Download BI export (CSV)",
                                data=open(bi_path,"rb").read(),
                                file_name="bi_export.csv")
