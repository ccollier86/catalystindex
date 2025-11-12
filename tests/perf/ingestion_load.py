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

from catalystindex.policies.resolver import resolve_policy
from tests.perf.base import SAMPLE_DOCUMENTS, build_perf_context


def run_ingestion_load(iterations: int) -> Sequence[float]:
    context = build_perf_context(namespace="ingestion-load")
    latencies: list[float] = []
    for index in range(iterations):
        spec = SAMPLE_DOCUMENTS[index % len(SAMPLE_DOCUMENTS)]
        document_id = f"{spec['document_id']}-run-{index}"
        policy = resolve_policy(str(spec["policy"]), str(spec["schema"]))
        start = time.perf_counter()
        result = context.ingestion.ingest(
            tenant=context.tenant,
            document_id=document_id,
            document_title=str(spec["title"]),
            content=str(spec["content"]),
            policy=policy,
            parser_name="plain_text",
            document_metadata={"source": "load-test"},
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not result.chunks:
            raise RuntimeError(f"No chunks produced for {document_id}")
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
    parser = argparse.ArgumentParser(description="Run ingestion load smoke test")
    parser.add_argument("--iterations", type=int, default=10, help="Number of ingestion iterations to execute")
    args = parser.parse_args()
    latencies = run_ingestion_load(args.iterations)
    summary = format_summary(latencies)
    print(summary)


if __name__ == "__main__":
    main()
