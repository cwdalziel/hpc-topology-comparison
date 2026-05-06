#!/usr/bin/env python3
"""
Aggregate benchmark result CSVs into one graph-ready CSV.

Input layout expected (from run_benchmarks.sh / run_3d_stencil_scaling.sh):
  results/<np>/<program>_results.csv
  results/<study>/<np>/<program>_results.csv

Where <study> is typically "strong" or "weak".
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class ResultRow:
    study: str
    program: str
    topology: str
    nodes: int
    time_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate results CSVs and compute speedup metrics for plotting. "
            "Output is long-form so you can plot one line per topology vs nodes."
        )
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing benchmark CSVs (default: results)",
    )
    parser.add_argument(
        "--output",
        default="results/aggregated_speedups.csv",
        help="Output CSV path (default: results/aggregated_speedups.csv)",
    )
    return parser.parse_args()


def program_from_filename(name: str) -> str:
    suffix = "_results.csv"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected results filename: {name}")
    return name[: -len(suffix)]


def discover_rows(results_dir: Path) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for csv_path in sorted(results_dir.rglob("*_results.csv")):
        rel = csv_path.relative_to(results_dir)
        parts = rel.parts

        # Supported:
        #   <np>/<program>_results.csv
        #   <study>/<np>/<program>_results.csv
        if len(parts) == 2:
            study = "default"
            np_part, file_name = parts
        elif len(parts) == 3:
            study, np_part, file_name = parts
        else:
            continue

        try:
            nodes = int(np_part)
        except ValueError:
            continue

        program = program_from_filename(file_name)
        rows.extend(read_result_file(csv_path, study, program, nodes))

    return rows


def read_result_file(path: Path, study: str, program: str, nodes: int) -> Iterable[ResultRow]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            topology = (raw.get("topology") or "").strip()
            time_str = (raw.get("time_s") or "").strip()
            if not topology or not time_str:
                continue
            if time_str.upper() == "ERROR":
                continue
            try:
                time_s = float(time_str)
            except ValueError:
                continue
            yield ResultRow(
                study=study,
                program=program,
                topology=topology,
                nodes=nodes,
                time_s=time_s,
            )


def write_aggregated_csv(rows: list[ResultRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Baseline is min nodes per (study, program, topology)
    grouped: dict[tuple[str, str, str], list[ResultRow]] = {}
    for row in rows:
        grouped.setdefault((row.study, row.program, row.topology), []).append(row)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "study",
            "program",
            "topology",
            "nodes",
            "time_s",
            "baseline_nodes",
            "baseline_time_s",
            "time_speedup",
            "throughput_speedup",
            "parallel_efficiency",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for key in sorted(grouped):
            series = sorted(grouped[key], key=lambda r: r.nodes)
            base = series[0]
            for row in series:
                time_speedup = base.time_s / row.time_s
                throughput_speedup = time_speedup * (row.nodes / base.nodes)
                parallel_efficiency = time_speedup / (row.nodes / base.nodes)
                writer.writerow(
                    {
                        "study": row.study,
                        "program": row.program,
                        "topology": row.topology,
                        "nodes": row.nodes,
                        "time_s": f"{row.time_s:.12g}",
                        "baseline_nodes": base.nodes,
                        "baseline_time_s": f"{base.time_s:.12g}",
                        "time_speedup": f"{time_speedup:.12g}",
                        "throughput_speedup": f"{throughput_speedup:.12g}",
                        "parallel_efficiency": f"{parallel_efficiency:.12g}",
                    }
                )


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output = Path(args.output)

    rows = discover_rows(results_dir)
    if not rows:
        raise SystemExit(f"No result rows found under: {results_dir}")

    write_aggregated_csv(rows, output)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
