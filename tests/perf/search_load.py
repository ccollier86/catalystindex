from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from catalystindex.services.search import SearchOptions
from tests.perf.base import build_perf_context, cycle_queries, load_sample_corpus


def run_search_load(iterations: int) -> Sequence[float]:
    context = build_perf_context(namespace="search-load")
    load_sample_corpus(context, repeats=2)
    latencies: list[float] = []
    for mode, query in cycle_queries(iterations):
        options = SearchOptions(mode=mode)
        start = time.perf_counter()
        execution = context.search.retrieve(context.tenant, query=query, options=options)
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not execution.results:
            raise RuntimeError(f"Search returned no results for mode={mode} query='{query}'")
        latencies.append(latency_ms)
    return latencies


def format_summary(latencies: Sequence[float]) -> str:
    if not latencies:
        return "No iterations executed"
    mean_latency = statistics.fmean(latencies)
    p95 = sorted(latencies)[max(int(len(latencies) * 0.95) - 1, 0)]
    return (
        f"iterations={len(latencies)} avg_ms={mean_latency:.2f} "
        f"min_ms={min(latencies):.2f} max_ms={max(latencies):.2f} p95_ms={p95:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run search load smoke test")
    parser.add_argument("--iterations", type=int, default=12, help="Number of search iterations to execute")
    args = parser.parse_args()
    latencies = run_search_load(args.iterations)
    summary = format_summary(latencies)
    print(summary)


if __name__ == "__main__":
    main()
