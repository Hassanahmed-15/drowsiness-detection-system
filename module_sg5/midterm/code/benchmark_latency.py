"""Measure per-decision latency of each SG-5 method.

Usage:  python benchmark_latency.py [iterations]
Runs every test case through each method many times and reports
mean / median / p99 latency per call and the equivalent decisions per second.
"""

import platform
import statistics
import sys
import time

from decision_rules import make_decision
from weighted_fusion import weighted_decision
from hybrid_fusion import hybrid_decision
from test_cases import basic_cases, borderline_cases

METHODS = {
    "Rule-Based": make_decision,
    "Weighted Fusion": weighted_decision,
    "Hybrid Fusion": hybrid_decision,
}

ITERATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
inputs = [d for d, _ in list(basic_cases.values()) + list(borderline_cases.values())]

print(f"Python {platform.python_version()} on {platform.platform()}")
print(f"Processor: {platform.processor() or 'unknown'}")
print(f"{ITERATIONS} rounds x {len(inputs)} inputs per method\n")
print(f"{'Method':<16} {'mean (us)':>10} {'median (us)':>12} {'p99 (us)':>10} {'decisions/s':>14}")

for name, fn in METHODS.items():
    for d in inputs:  # warm-up
        fn(d)
    samples = []
    for _ in range(ITERATIONS):
        for d in inputs:
            t0 = time.perf_counter_ns()
            fn(d)
            samples.append(time.perf_counter_ns() - t0)
    samples.sort()
    mean_us = statistics.fmean(samples) / 1000
    median_us = samples[len(samples) // 2] / 1000
    p99_us = samples[int(len(samples) * 0.99)] / 1000
    print(f"{name:<16} {mean_us:>10.2f} {median_us:>12.2f} {p99_us:>10.2f} {1e6 / mean_us:>14,.0f}")
