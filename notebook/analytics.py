from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def daily_pnl(df: pd.DataFrame) -> pd.DataFrame:
    out = (df.groupby("trade_date", as_index=False)["pnl"].sum()
             .rename(columns={"pnl":"pnl_total"})
             .sort_values("trade_date"))
    return out

def historical_var(pnl_series: pd.Series, alpha=0.95, lookback=60) -> float:
    s = pnl_series.dropna().tail(lookback)
    if len(s) < 5:
        return float("nan")
    return float(-np.quantile(s.values, 1 - alpha))

def exposure_tables(df: pd.DataFrame):
    cp = (df.groupby("counterparty", as_index=False)["notional_mn"].sum()
            .rename(columns={"notional_mn":"exposure_mn"})
            .sort_values("exposure_mn", ascending=False))
    ac = (df.groupby("asset_class", as_index=False)["notional_mn"].sum()
            .rename(columns={"notional_mn":"exposure_mn"})
            .sort_values("exposure_mn", ascending=False))
    return cp, ac

def bi_export(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out["yyyymm"] = out["trade_date"].dt.to_period("M").astype(str)
    return out[["trade_date","yyyymm","counterparty","asset_class","currency","trade_type","notional","notional_mn","pnl"]]

def tableau_long_export(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Long (tidy) format for Tableau: one row per metric.
    id_vars are dimensions; value_vars are numeric measures.
    """
    df2 = df.copy()
    df2["trade_date"] = pd.to_datetime(df2["trade_date"])
    df2["yyyymm"] = df2["trade_date"].dt.to_period("M").astype(str)

    long = pd.melt(
        df2,
        id_vars=["trade_date", "yyyymm", "counterparty", "asset_class", "currency", "trade_type"],
        value_vars=["notional", "notional_mn", "pnl"],
        var_name="metric",
        value_name="value",
    )
    long.to_csv(out_dir / "tableau_long.csv", index=False)
    return long

def tableau_long_daily_agg(long_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Aggregate long data by day + dims for fast Tableau workbooks.
    """
    agg = (long_df
           .groupby(["trade_date", "yyyymm", "counterparty", "asset_class", "currency", "trade_type", "metric"],
                    as_index=False)["value"].sum())
    agg.to_csv(out_dir / "tableau_long_daily_agg.csv", index=False)
    return agg

if __name__ == "__main__":
    # Load cleaned trades produced by transform step
    trades = pd.read_csv(DATA / "processed_trades.csv", parse_dates=["trade_date"])

    # 1) Daily portfolio P&L
    pnl = daily_pnl(trades)
    pnl.to_csv(DATA / "pnl_daily.csv", index=False)

    # 2) VaR (historical, 95%, 60d lookback)
    var_val = historical_var(pnl["pnl_total"], alpha=0.95, lookback=60)
    pd.DataFrame([{"alpha":0.95, "lookback_days":60, "VaR":var_val}]).to_csv(DATA / "var_metrics.csv", index=False)

    # 3) Exposure tables
    cp, ac = exposure_tables(trades)
    cp.to_csv(DATA / "exposure_by_counterparty.csv", index=False)
    ac.to_csv(DATA / "exposure_by_asset.csv", index=False)

    # 4) BI export for Power BI/Tableau
    bi = bi_export(trades)
    bi.to_csv(DATA / "bi_export.csv", index=False)

    # 5) Tableau long-format exports
    long = tableau_long_export(trades, DATA)
    tableau_long_daily_agg(long, DATA)

    print("Analytics written to data/: pnl_daily.csv, var_metrics.csv, exposure_by_counterparty.csv, exposure_by_asset.csv, bi_export.csv")
