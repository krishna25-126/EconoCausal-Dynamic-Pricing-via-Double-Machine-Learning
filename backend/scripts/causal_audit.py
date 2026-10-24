"""Mid-project causal audit: try to refute the causal estimate (DoWhy).

    python scripts/causal_audit.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from econocausal import data, refutation  # noqa: E402


def main():
    df = data.generate_observational(6000)
    a = refutation.run_audit(df, simulations=20)
    print(a["estimand"])
    print(f"Estimated effect: {a['estimated_effect_per_dollar']:+.5f} per $1\n")
    for t in a["tests"]:
        status = "PASS" if t["passed"] else "FAIL"
        print(f"[{status}] {t['test']:<22} original {t['original_effect']:+.5f}  new {t['new_effect']:+.5f}")
        print(f"        {t['what_it_checks']}")
    print("\nOverall:", "all refutation tests passed" if a["all_passed"] else "some tests failed, investigate")


if __name__ == "__main__":
    main()