"""Score a House-Prices submission and write a Harbor reward.

Reward = R2 (coefficient of determination, higher is better, clipped at 0).
Also reports MAE and RMSE. Always writes a reward file (0.0 on failure).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "SalePrice"
ID = "Id"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--answer", required=True)
    ap.add_argument("--reward-json", required=True)
    args = ap.parse_args()

    reward_path = Path(args.reward_json)
    reward_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(reward: float, **extra) -> int:
        payload = {"reward": float(reward), **extra}
        reward_path.write_text(json.dumps(payload), encoding="utf-8")
        print(json.dumps(payload))
        return 0

    try:
        answer = pd.read_csv(args.answer)
        submission = pd.read_csv(args.submission)
    except Exception as exc:  # noqa: BLE001
        return emit(0.0, error=f"failed to read inputs: {exc}")

    if ID not in submission.columns or TARGET not in submission.columns:
        return emit(0.0, error=f"submission must have columns [{ID}, {TARGET}]")

    merged = answer.merge(submission, on=ID, suffixes=("_true", "_pred"))
    if merged.empty:
        return emit(0.0, error="no Id overlap between submission and answer")

    y_true = pd.to_numeric(merged[f"{TARGET}_true"], errors="coerce").to_numpy()
    y_pred = pd.to_numeric(merged[f"{TARGET}_pred"], errors="coerce").to_numpy()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return emit(0.0, error="no numeric predictions")
    y_true, y_pred = y_true[mask], y_pred[mask]

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return emit(max(0.0, r2), r2=r2, mae=mae, rmse=rmse, n=int(mask.sum()))


if __name__ == "__main__":
    raise SystemExit(main())
