"""Run all three SG-5 methods on the basic and borderline test cases.

Usage:  python run_evaluation.py
Writes: sg5_method_comparison.csv
"""

import csv

from decision_rules import make_decision
from weighted_fusion import weighted_decision
from hybrid_fusion import hybrid_decision
from test_cases import basic_cases, borderline_cases

METHODS = {
    "Rule-Based": make_decision,
    "Weighted Fusion": weighted_decision,
    "Hybrid Fusion": hybrid_decision,
}


def evaluate(title, cases):
    print("=" * 75)
    print(title)
    print("=" * 75)
    correct = {m: 0 for m in METHODS}
    rows = []
    for name, (data, expected) in cases.items():
        row = {"Test Case": name, "Expected": expected}
        print(f"\n{name}  (expected: {expected})")
        for method, fn in METHODS.items():
            result = fn(data)
            ok = result["state"] == expected
            correct[method] += ok
            row[method] = result["state"]
            print(f"  {method:<16} {str(result['state']):<8} "
                  f"score={result['drowsiness_score']}  correct={ok}")
        rows.append(row)

    print("\nAccuracy:")
    for method in METHODS:
        n = len(cases)
        print(f"  {method:<16} {correct[method]}/{n} = {100 * correct[method] / n:.1f}%")
    print()
    return rows


if __name__ == "__main__":
    evaluate("BASIC TEST CASES", basic_cases)
    rows = evaluate("BORDERLINE TEST CASES", borderline_cases)

    with open("sg5_method_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Test Case", "Expected", *METHODS])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved: sg5_method_comparison.csv")
