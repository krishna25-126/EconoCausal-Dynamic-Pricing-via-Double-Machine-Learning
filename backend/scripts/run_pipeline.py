"""End-to-end demo: data -> causal model -> evaluation -> optimized allocation.

    python scripts/run_pipeline.py --budget 5000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from econocausal import service  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000, help="historical rows")
    ap.add_argument("--budget", type=float, default=5000)
    ap.add_argument("--trees", type=int, default=300)
    args = ap.parse_args()

    s = service.Session()
    info = service.load_demo(s, args.n)
    print(f"Loaded {info['rows']:,} historical customers "
          f"({info['share_discounted']:.0%} got a discount).")
    print(f"  Naive purchase-rate gap (discounted - not): "
          f"{info['purchase_rate_discounted'] - info['purchase_rate_not_discounted']:+.3f}  <- misleading")

    r = service.train(s, args.trees)
    print(f"\nEffect of $1 discount on purchase probability")
    print(f"  Naive regression : {r['naive_effect_per_dollar']:+.4f}")
    print(f"  Double ML        : {r['average_effect_per_dollar']:+.4f}")
    if r["validation"]:
        v = r["validation"]
        print(f"  Ground truth     : {v['mean_true']:+.4f}   (correlation of individual estimates: {v['correlation']:.2f})")
    m = r["matching"]
    print(f"\nPropensity matching: max |SMD| {m['max_abs_smd_before']:.2f} -> {m['max_abs_smd_after']:.2f}")
    print(f"Qini coefficient on randomized hold-out: {r['qini']['qini_coefficient']:.1f}")

    a = service.allocate(s, args.budget)
    print(f"\nAllocation with ${args.budget:,.0f} budget: spend ${a['expected_spend']:,.0f}, "
          f"{a['customers_discounted']:,} of {a['customers']:,} customers discounted")
    for st in a["strategies"]:
        key = "true_profit_gain" if "true_profit_gain" in st else "expected_profit_gain"
        print(f"  {st['strategy']:<22} profit gain {st[key]:>9,.0f}")
    print("\n" + a["summary"])


if __name__ == "__main__":
    main()