"""Stage AMP-Parkinsons-style data (synthetic, offline): protein/peptide
abundances + visit month -> UPDRS clinical score (regression, temporal)."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, pandas as pd
WORKSPACE = Path(os.environ.get("MLAB_WORKSPACE", "/workspace"))
ANSWER_DIR = Path(os.environ.get("MLAB_ANSWER_DIR", "/opt/mlab"))
def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True); ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11); n = 3000
    prot = {f"P{i:03d}": rng.lognormal(3, 1, n).round(2) for i in range(40)}   # protein abundances
    visit_month = rng.choice([0, 6, 12, 18, 24, 36], n)
    coef = rng.normal(0, 1, 40)
    base = sum(coef[i] * np.log1p(prot[f"P{i:03d}"]) for i in range(40))
    updrs = (10 + 3 * (base - base.mean()) / (base.std() + 1e-9) + 0.15 * visit_month + rng.normal(0, 3, n)).clip(0, 60).round(2)
    df = pd.DataFrame({"id": np.arange(n), "visit_month": visit_month, **prot, "updrs": updrs})
    cut = int(n * 0.8); tr, ho = df.iloc[:cut], df.iloc[cut:]
    tr.to_csv(WORKSPACE / "train.csv", index=False)
    ho.drop(columns=["updrs"]).to_csv(WORKSPACE / "test.csv", index=False)
    ho[["id", "updrs"]].to_csv(ANSWER_DIR / "answer.csv", index=False)
    print(f"[prepare] amp-parkinsons train={cut} test={n-cut}")
if __name__ == "__main__":
    main()
