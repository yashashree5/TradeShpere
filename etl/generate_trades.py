from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(11)

COUNTERPARTIES = ["Bank A", "Bank B", "Bank C", "Bank D"]
ASSET_CLASSES  = ["Equity", "Bond", "FX", "Commodity"]
CURRENCIES     = ["USD", "EUR", "GBP", "JPY"]
TRADE_TYPES    = ["Buy", "Sell"]

def make_trades(n=1200) -> pd.DataFrame:
    dates = pd.bdate_range(
        pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(120),
        pd.Timestamp.today().normalize()
    )
    rows = []
    for _ in range(n):
        # ensure we get a pandas Timestamp (not numpy.datetime64)
        tdate = dates[rng.integers(0, len(dates))]

        cp  = rng.choice(COUNTERPARTIES)
        ac  = rng.choice(ASSET_CLASSES, p=[0.45, 0.25, 0.20, 0.10])
        ccy = rng.choice(CURRENCIES)
        ttp = rng.choice(TRADE_TYPES, p=[0.55, 0.45])

        notional = float(rng.uniform(2e5, 4e6))
        pnl = float(rng.normal(0, notional * 0.02))

        rows.append({
            "trade_id": int(rng.integers(100000, 999999)),
            "counterparty": cp,
            "asset_class": ac,
            "currency": ccy,
            "trade_type": ttp,
            "trade_date": tdate.date(),  # now valid
            "notional": notional,
            "pnl": pnl,
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    df = make_trades()
    (data / "trades.csv").write_text(df.to_csv(index=False))
    print("Generated:", data / "trades.csv")
