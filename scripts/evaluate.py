from __future__ import annotations

import argparse
import csv
import math
from typing import Dict, List, Tuple


def _read_speeds(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Predicted speeds CSV from outputs/")
    ap.add_argument("--gt", default=None, help="Optional ground truth CSV with columns: timestamp_s,speed_mps")
    args = ap.parse_args()

    pred_rows = _read_speeds(args.pred)
    pred_v = [_to_float(r.get("speed_mps_smoothed", "nan")) for r in pred_rows]
    pred_v = [v for v in pred_v if v == v and v >= 0.0]
    if not pred_v:
        raise RuntimeError("No valid predicted speeds found")

    mean = sum(pred_v) / len(pred_v)
    p95 = sorted(pred_v)[int(0.95 * (len(pred_v) - 1))]
    print(f"pred_count={len(pred_v)} mean_mps={mean:.3f} p95_mps={p95:.3f}")

    if args.gt is None:
        return

    gt_rows = _read_speeds(args.gt)
    gt_v = [_to_float(r.get("speed_mps", "nan")) for r in gt_rows]
    gt_v = [v for v in gt_v if v == v and v >= 0.0]
    if not gt_v:
        raise RuntimeError("No valid ground truth speeds found")

    n = min(len(pred_v), len(gt_v))
    errs = [abs(pred_v[i] - gt_v[i]) for i in range(n)]
    mae = sum(errs) / n
    rmse = (sum(e * e for e in errs) / n) ** 0.5
    print(f"aligned_n={n} mae_mps={mae:.3f} rmse_mps={rmse:.3f}")


if __name__ == "__main__":
    main()

