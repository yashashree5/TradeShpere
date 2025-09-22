from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

REQUIRED = ["trade_id","counterparty","asset_class","currency","trade_type","trade_date","notional","pnl"]

def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["notional"]   = pd.to_numeric(df["notional"], errors="coerce")
    df["pnl"]        = pd.to_numeric(df["pnl"], errors="coerce")
    df = df.dropna(subset=REQUIRED).copy()
    df["notional_mn"] = df["notional"]/1e6
    df["year"]  = df["trade_date"].dt.year
    df["month"] = df["trade_date"].dt.month
    return df

def summarize_counterparty(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("counterparty", dropna=False)
              .agg(total_exposure_mn=("notional_mn","sum"),
                   trades=("trade_id","count"),
                   pnl_total=("pnl","sum"),
                   pnl_avg=("pnl","mean"))
              .reset_index()
              .sort_values("total_exposure_mn", ascending=False))

def main():
    ap = argparse.ArgumentParser(description="Clean trades and create summaries.")
    ap.add_argument("--input",  type=Path, default=Path("data/trades.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("data"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_and_clean(args.input)
    df.to_csv(args.outdir / "processed_trades.csv", index=False)

    cp = summarize_counterparty(df)
    cp.to_csv(args.outdir / "summary_metrics.csv", index=False)

    print("Wrote:")
    print("-", args.outdir / "processed_trades.csv")
    print("-", args.outdir / "summary_metrics.csv")

if __name__ == "__main__":
    main()
