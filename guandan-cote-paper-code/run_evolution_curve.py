# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
CLIENTS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run current-platform COTE evolution-curve checkpoints.")
    parser.add_argument("--points", type=int, default=5, help="Number of checkpoints to evaluate.")
    parser.add_argument("--episodes-per-point", type=int, default=20)
    parser.add_argument("--server-games", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--our-clients", default="1,3")
    parser.add_argument("--opponent-policy", choices=["random", "cote", "danzero"], default="random")
    parser.add_argument("--use-local-model", action="store_true")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else ROOT.parent / ".run_logs" / (
        "evolution_curve_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["COTE_STATE_PATH"] = str(output_dir / "state_client{client_id}.json")
    env.setdefault("PYTHONUNBUFFERED", "1")

    records: List[Dict[str, Any]] = []
    for point in range(args.points):
        command = [
            sys.executable,
            str(CLIENTS_DIR / "evaluate_cote.py"),
            "--policy",
            "cote",
            "--opponent-policy",
            args.opponent_policy,
            "--our-clients",
            args.our_clients,
            "--episodes",
            str(args.episodes_per_point),
            "--server-games",
            str(args.server_games),
            "--timeout",
            str(args.timeout),
            "--keep-logs",
        ]
        if args.use_local_model:
            command.append("--use-local-model")
        proc = subprocess.run(command, cwd=str(CLIENTS_DIR), env=env, text=True, capture_output=True)
        summary = parse_summary(proc.stdout)
        summary["checkpoint"] = point
        summary["returncode"] = proc.returncode
        summary["stderr_tail"] = proc.stderr[-1200:]
        records.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    write_outputs(output_dir, records, "evolution_curve")
    return 0


def parse_summary(stdout: str) -> Dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        return {"parse_error": "no JSON summary in stdout", "stdout_tail": stdout[-1200:]}
    try:
        return json.loads(stdout[start : end + 1])
    except json.JSONDecodeError as exc:
        return {"parse_error": str(exc), "stdout_tail": stdout[-1200:]}


def write_outputs(output_dir: Path, records: List[Dict[str, Any]], stem: str) -> None:
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "checkpoint",
        "episodes",
        "episode_win_rate",
        "average_point_diff",
        "point_diff_sum",
        "edge_retention_avg",
        "edge_count_avg",
        "avg_tokens_per_deal",
        "log_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            edge = record.get("edge_retention", {}) if isinstance(record.get("edge_retention"), dict) else {}
            tokens = record.get("token_usage", {}) if isinstance(record.get("token_usage"), dict) else {}
            writer.writerow(
                {
                    "checkpoint": record.get("checkpoint"),
                    "episodes": record.get("episodes"),
                    "episode_win_rate": record.get("episode_win_rate"),
                    "average_point_diff": record.get("average_point_diff"),
                    "point_diff_sum": record.get("point_diff_sum"),
                    "edge_retention_avg": edge.get("average_edge_retention"),
                    "edge_count_avg": edge.get("average_edge_count"),
                    "avg_tokens_per_deal": tokens.get("average_total_tokens_per_deal"),
                    "log_dir": record.get("log_dir"),
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
