# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
CLIENTS_DIR = Path(__file__).resolve().parent


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run both available COTE-vs-DanZero_plus seat directions.")
    parser.add_argument("--episodes-per-seat", type=int, default=20)
    parser.add_argument("--server-games", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--topology-init", default="dense", choices=["dense"])
    parser.add_argument("--use-local-model", action="store_true")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--duplicate-randomization",
        action="store_true",
        help="Require exact mirrored deal replay instead of the default seat-swap approximation.",
    )
    return parser.parse_args(argv)


def validate_duplicate_randomization(args: argparse.Namespace) -> None:
    if not getattr(args, "duplicate_randomization", False):
        return
    supported = os.environ.get("COTE_EXACT_DEAL_REPLAY", "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }
    if not supported:
        raise RuntimeError(
            "Duplicate Randomization requires exact deal replay support from the offline server. "
            "Set COTE_EXACT_DEAL_REPLAY=1 only after fixed mirrored deals are actually available."
        )


def main() -> int:
    args = parse_args()
    validate_duplicate_randomization(args)
    output_dir = Path(args.output_dir) if args.output_dir else ROOT.parent / ".run_logs" / (
        "danzero_mirrored_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    for seats in ("1,3", "0,2"):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["COTE_TOPOLOGY_INIT"] = args.topology_init
        env["COTE_STATE_PATH"] = str(output_dir / f"seats_{seats.replace(',', '_')}" / "state_client{client_id}.json")
        command = [
            sys.executable,
            str(CLIENTS_DIR / "evaluate_cote.py"),
            "--policy",
            "cote",
            "--opponent-policy",
            "danzero",
            "--our-clients",
            seats,
            "--episodes",
            str(args.episodes_per_seat),
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
        summary["seat_direction"] = seats
        summary["returncode"] = proc.returncode
        summary["stderr_tail"] = proc.stderr[-1200:]
        records.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    aggregate = aggregate_records(records)
    aggregate["note"] = (
        "This is a bidirectional seat-swap approximation. It is not exact Duplicate Randomization "
        "because the offline server does not expose mirrored deal replay."
    )
    payload = {"aggregate": aggregate, "runs": records}
    (output_dir / "danzero_mirrored.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0 if aggregate.get("episode_win_rate", 0.0) >= 0.60 else 2


def parse_summary(stdout: str) -> Dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        return {"parse_error": "no JSON summary in stdout", "stdout_tail": stdout[-1200:]}
    try:
        return json.loads(stdout[start : end + 1])
    except json.JSONDecodeError as exc:
        return {"parse_error": str(exc), "stdout_tail": stdout[-1200:]}


def aggregate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_episodes = 0
    measured_wins = 0
    point_diff_sum = 0
    edge_retentions: List[float] = []
    edge_counts: List[float] = []
    total_tokens = 0
    decision_attempts = 0
    decision_successes = 0
    node_attempts = 0
    node_successes = 0
    edge_attempts = 0
    edge_successes = 0
    log_dirs: List[str] = []
    for record in records:
        episodes = int(record.get("episodes") or 0)
        total_episodes += episodes
        our_team = record.get("our_team")
        team_wins = record.get("team_wins", {})
        if our_team in (0, 1) and isinstance(team_wins, dict):
            measured_wins += int(team_wins.get(f"team{our_team}", 0) or 0)
        point_diff_sum += int(record.get("point_diff_sum") or 0)
        edge = record.get("edge_retention", {})
        if isinstance(edge, dict):
            if edge.get("average_edge_retention") is not None:
                edge_retentions.append(float(edge["average_edge_retention"]))
            if edge.get("average_edge_count") is not None:
                edge_counts.append(float(edge["average_edge_count"]))
        tokens = record.get("token_usage", {})
        if isinstance(tokens, dict):
            total_tokens += int(tokens.get("total_tokens", 0) or 0)
        model = record.get("model_control", {})
        if isinstance(model, dict):
            decision_attempts += int(model.get("decision_attempts", 0) or 0)
            decision_successes += int(model.get("decision_successes", 0) or 0)
            node_attempts += int(model.get("node_attempts", 0) or 0)
            node_successes += int(model.get("node_successes", 0) or 0)
            edge_attempts += int(model.get("edge_attempts", 0) or 0)
            edge_successes += int(model.get("edge_successes", 0) or 0)
        if record.get("log_dir"):
            log_dirs.append(str(record["log_dir"]))

    win_rate = measured_wins / total_episodes if total_episodes else 0.0
    return {
        "episodes": total_episodes,
        "measured_wins": measured_wins,
        "opponent_wins": total_episodes - measured_wins,
        "episode_win_rate": round(win_rate, 4),
        "point_diff_sum": point_diff_sum,
        "average_point_diff": round(point_diff_sum / total_episodes, 4) if total_episodes else None,
        "average_edge_retention": round(sum(edge_retentions) / len(edge_retentions), 4) if edge_retentions else None,
        "average_edge_count": round(sum(edge_counts) / len(edge_counts), 4) if edge_counts else None,
        "total_tokens": total_tokens,
        "average_total_tokens_per_deal": round(total_tokens / total_episodes, 4) if total_episodes else 0.0,
        "decision_attempts": decision_attempts,
        "decision_successes": decision_successes,
        "node_attempts": node_attempts,
        "node_successes": node_successes,
        "edge_attempts": edge_attempts,
        "edge_successes": edge_successes,
        "log_dirs": log_dirs,
    }


if __name__ == "__main__":
    raise SystemExit(main())
