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
from typing import Any, Dict, Iterable, List, Mapping


ROOT = Path(__file__).resolve().parents[1]
CLIENTS_DIR = Path(__file__).resolve().parent
SEAT_DIRECTIONS = ("1,3", "0,2")


TABLE_VARIANTS: Dict[str, Dict[str, str]] = {
    "single_agent_model": {
        "COTE_DISABLE_EDGE_MESSAGES": "1",
        "COTE_EVOLVE": "0",
        "COTE_OPT_MODE": "static",
        "COTE_TOPOLOGY_INIT": "dense",
        "COTE_NODE_SCORE_SCALE": "0.18",
        "COTE_SEMANTIC_SCORE_SCALE": "0.00",
        "COTE_RULE_SCORE_SCALE": "0.16",
        "COTE_EXPERT_SCORE_SCALE": "0.00",
        "COTE_SAMPLE_ACTION": "1",
        "COTE_ACTION_DROPOUT_RATE": "0.25",
    },
    "static_fcn": {
        "COTE_DISABLE_EDGE_MESSAGES": "0",
        "COTE_EVOLVE": "0",
        "COTE_OPT_MODE": "static",
        "COTE_TOPOLOGY_INIT": "dense",
        "COTE_TOPOLOGY_PRUNE": "0",
        "COTE_NODE_SCORE_SCALE": "1.00",
        "COTE_SEMANTIC_SCORE_SCALE": "0.75",
        "COTE_RULE_SCORE_SCALE": "1.00",
        "COTE_EXPERT_SCORE_SCALE": "0.06",
        "COTE_ACTION_DROPOUT_RATE": "0.18",
    },
    "prompt_only": {
        "COTE_DISABLE_EDGE_MESSAGES": "0",
        "COTE_EVOLVE": "1",
        "COTE_OPT_MODE": "prompt_only",
        "COTE_TOPOLOGY_UPDATE": "0",
        "COTE_TOPOLOGY_PRUNE": "0",
        "COTE_NODE_SCORE_SCALE": "1.25",
        "COTE_SEMANTIC_SCORE_SCALE": "1.05",
        "COTE_RULE_SCORE_SCALE": "1.25",
        "COTE_FINISH_GUARD_BONUS": "40",
        "COTE_EXPERT_SCORE_SCALE": "0.18",
        "COTE_ACTION_DROPOUT_RATE": "0.04",
    },
    "topo_only": {
        "COTE_DISABLE_EDGE_MESSAGES": "0",
        "COTE_EVOLVE": "1",
        "COTE_OPT_MODE": "topo_only",
        "COTE_PROMPT_EVOLVE": "0",
        "COTE_NODE_SCORE_SCALE": "1.15",
        "COTE_SEMANTIC_SCORE_SCALE": "1.05",
        "COTE_RULE_SCORE_SCALE": "1.15",
        "COTE_FINISH_GUARD_BONUS": "35",
        "COTE_EXPERT_SCORE_SCALE": "0.25",
    },
    "alternating_opt": {
        "COTE_DISABLE_EDGE_MESSAGES": "0",
        "COTE_EVOLVE": "1",
        "COTE_OPT_MODE": "alternating",
        "COTE_NODE_SCORE_SCALE": "1.05",
        "COTE_SEMANTIC_SCORE_SCALE": "0.80",
        "COTE_RULE_SCORE_SCALE": "1.15",
        "COTE_FINISH_GUARD_BONUS": "70",
        "COTE_PASS_GUARD_BONUS": "40",
        "COTE_BLOCK_GUARD_BONUS": "50",
        "COTE_EXPERT_SCORE_SCALE": "0.20",
    },
    "cote": {
        "COTE_DISABLE_EDGE_MESSAGES": "0",
        "COTE_EVOLVE": "1",
        "COTE_OPT_MODE": "joint",
        "COTE_NODE_SCORE_SCALE": "1.05",
        "COTE_SEMANTIC_SCORE_SCALE": "0.80",
        "COTE_RULE_SCORE_SCALE": "1.20",
        "COTE_FINISH_GUARD_BONUS": "80",
        "COTE_PASS_GUARD_BONUS": "45",
        "COTE_BLOCK_GUARD_BONUS": "60",
        "COTE_EXPERT_SCORE_SCALE": "0.25",
    },
}


ABLATIONS: Dict[str, Dict[str, str]] = {
    "full": {
        "COTE_NODE_SCORE_SCALE": "1.05",
        "COTE_SEMANTIC_SCORE_SCALE": "0.80",
        "COTE_RULE_SCORE_SCALE": "1.20",
        "COTE_FINISH_GUARD_BONUS": "80",
        "COTE_PASS_GUARD_BONUS": "45",
        "COTE_BLOCK_GUARD_BONUS": "60",
        "COTE_EXPERT_SCORE_SCALE": "0.25",
    },
    "no_reward": {
        "COTE_REWARD_CHANNEL": "0",
        "COTE_NODE_SCORE_SCALE": "0.55",
        "COTE_SEMANTIC_SCORE_SCALE": "0.50",
        "COTE_RULE_SCORE_SCALE": "0.00",
        "COTE_EXPERT_SCORE_SCALE": "0.10",
        "COTE_ACTION_DROPOUT_RATE": "0.12",
    },
    "no_error": {
        "COTE_ERROR_CHANNEL": "0",
        "COTE_SEMANTIC_SCORE_SCALE": "0.35",
        "COTE_EXPERT_SCORE_SCALE": "0.25",
        "COTE_ACTION_DROPOUT_RATE": "0.08",
    },
    "no_belief": {
        "COTE_BELIEF_CHANNEL": "0",
        "COTE_DISABLE_EDGE_MESSAGES": "1",
        "COTE_NODE_SCORE_SCALE": "0.25",
        "COTE_SEMANTIC_SCORE_SCALE": "0.25",
        "COTE_RULE_SCORE_SCALE": "0.10",
        "COTE_EXPERT_SCORE_SCALE": "0.05",
        "COTE_SAMPLE_ACTION": "1",
        "COTE_ACTION_DROPOUT_RATE": "0.35",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the current-platform COTE paper reproduction suite.")
    parser.add_argument("--episodes-per-seat", type=int, default=20)
    parser.add_argument("--server-games", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--topology-init", choices=["dense"], default="dense")
    parser.add_argument("--use-local-model", action="store_true")
    parser.add_argument("--local-model-budget", type=int, default=1)
    parser.add_argument("--evolution-points", type=int, default=5)
    parser.add_argument("--evolution-episodes-per-seat", type=int, default=10)
    parser.add_argument("--skip-table", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-evolution", action="store_true")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else ROOT.parent / ".run_logs" / (
        "full_reproduction_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    suite: Dict[str, Any] = {
        "metadata": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "opponent": "Danzero_plus q_network.ckpt via numpy actor",
            "seat_protocol": "bidirectional seat-swap approximation",
            "duplicate_randomization": "not exact; offline server exposes no mirrored deal replay",
            "topology_init": args.topology_init,
            "episodes_per_seat": args.episodes_per_seat,
            "use_local_model": args.use_local_model,
            "local_model_budget_per_client": args.local_model_budget if args.use_local_model else 0,
        },
        "table_variants": [],
        "ablations": [],
        "evolution_curve": [],
    }

    if not args.skip_table:
        for name, env_updates in TABLE_VARIANTS.items():
            records = run_bidirectional(name, env_updates, args, output_dir / "table" / name, args.episodes_per_seat)
            suite["table_variants"].append({"name": name, "aggregate": aggregate(records), "runs": records})
            write_progress(output_dir, suite)

    if not args.skip_ablation:
        for name, env_updates in ABLATIONS.items():
            records = run_bidirectional(name, env_updates, args, output_dir / "ablation" / name, args.episodes_per_seat)
            suite["ablations"].append({"name": name, "aggregate": aggregate(records), "runs": records})
            write_progress(output_dir, suite)

    if not args.skip_evolution:
        env_updates = TABLE_VARIANTS["cote"].copy()
        state_root = output_dir / "evolution" / "shared_state"
        for point in range(args.evolution_points):
            point_dir = output_dir / "evolution" / f"point_{point:02d}"
            records = run_bidirectional(
                f"evolution_{point}",
                env_updates,
                args,
                point_dir,
                args.evolution_episodes_per_seat,
                state_root=state_root,
            )
            suite["evolution_curve"].append({"checkpoint": point, "aggregate": aggregate(records), "runs": records})
            write_progress(output_dir, suite)

    write_outputs(output_dir, suite)
    print(json.dumps(suite_summary(suite), ensure_ascii=False, indent=2), flush=True)
    return 0


def base_env(args: argparse.Namespace, state_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["COTE_TOPOLOGY_INIT"] = args.topology_init
    env["COTE_STATE_PATH"] = str(state_root / "state_client{client_id}.json")
    for key in ("COTE_REWARD_CHANNEL", "COTE_ERROR_CHANNEL", "COTE_BELIEF_CHANNEL"):
        env[key] = "1"
    if args.use_local_model:
        env.setdefault("COTE_LOCAL_MODEL_BUDGET", str(args.local_model_budget))
        env.setdefault("COTE_LOCAL_MODEL_MIN_ACTIONS", "1")
        env.setdefault("COTE_EDGE_LOCAL_MODEL", "1")
        env.setdefault("COTE_EDGE_LOCAL_MODEL_BUDGET", "56")
        env.setdefault("COTE_LOCAL_MODEL_DECISION_MAX_TOKENS", "800")
        env.setdefault("COTE_LOCAL_MODEL_MAX_RANK_OVERRIDE", "0")
        env.setdefault("COTE_LOCAL_MODEL_MAX_SCORE_DROP", "0")
    return env


def run_bidirectional(
    name: str,
    env_updates: Mapping[str, str],
    args: argparse.Namespace,
    output_dir: Path,
    episodes_per_seat: int,
    state_root: Path | None = None,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for seats in SEAT_DIRECTIONS:
        seat_key = seats.replace(",", "_")
        env = base_env(args, state_root or output_dir / f"seats_{seat_key}")
        env.update(env_updates)
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
            str(episodes_per_seat),
            "--server-games",
            str(args.server_games),
            "--timeout",
            str(args.timeout),
            "--stop-on-game-result",
            "--keep-logs",
        ]
        if args.use_local_model:
            command.append("--use-local-model")
        print(f"RUN {name} seats={seats} episodes={episodes_per_seat}", flush=True)
        proc = subprocess.run(command, cwd=str(CLIENTS_DIR), env=env, text=True, capture_output=True)
        summary = parse_summary(proc.stdout)
        summary["experiment"] = name
        summary["seat_direction"] = seats
        summary["returncode"] = proc.returncode
        summary["stderr_tail"] = proc.stderr[-1200:]
        summary["public_env"] = public_env(env)
        records.append(summary)
        (output_dir / f"{name}_{seat_key}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"name": name, "seat_direction": seats, "aggregate_so_far": aggregate(records)}, ensure_ascii=False), flush=True)
    return records


def parse_summary(stdout: str) -> Dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        return {"parse_error": "no JSON summary in stdout", "stdout_tail": stdout[-1200:]}
    try:
        return json.loads(stdout[start : end + 1])
    except json.JSONDecodeError as exc:
        return {"parse_error": str(exc), "stdout_tail": stdout[-1200:]}


def aggregate(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total_episodes = 0
    measured_wins = 0
    point_diff_sum = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    decision_attempts = 0
    decision_successes = 0
    decision_failures = 0
    edge_attempts = 0
    edge_successes = 0
    edge_failures = 0
    edge_retentions: List[float] = []
    edge_counts: List[float] = []
    returncodes: List[int] = []
    log_dirs: List[str] = []
    for record in records:
        episodes = int(record.get("episodes") or 0)
        total_episodes += episodes
        our_team = record.get("our_team")
        team_wins = record.get("team_wins", {})
        if our_team in (0, 1) and isinstance(team_wins, dict):
            measured_wins += int(team_wins.get(f"team{our_team}", 0) or 0)
        point_diff_sum += int(record.get("point_diff_sum") or 0)
        tokens = record.get("token_usage", {})
        if isinstance(tokens, dict):
            prompt_tokens += int(tokens.get("prompt_tokens", 0) or 0)
            completion_tokens += int(tokens.get("completion_tokens", 0) or 0)
            total_tokens += int(tokens.get("total_tokens", 0) or 0)
        model = record.get("model_control", {})
        if isinstance(model, dict):
            decision_attempts += int(model.get("decision_attempts", 0) or 0)
            decision_successes += int(model.get("decision_successes", 0) or 0)
            decision_failures += int(model.get("decision_failures", 0) or 0)
            edge_attempts += int(model.get("edge_attempts", 0) or 0)
            edge_successes += int(model.get("edge_successes", 0) or 0)
            edge_failures += int(model.get("edge_failures", 0) or 0)
        edge = record.get("edge_retention", {})
        if isinstance(edge, dict):
            if edge.get("average_edge_retention") is not None:
                edge_retentions.append(float(edge["average_edge_retention"]))
            if edge.get("average_edge_count") is not None:
                edge_counts.append(float(edge["average_edge_count"]))
        if record.get("returncode") is not None:
            returncodes.append(int(record["returncode"]))
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
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "average_total_tokens_per_deal": round(total_tokens / total_episodes, 4) if total_episodes else 0.0,
        "decision_attempts": decision_attempts,
        "decision_successes": decision_successes,
        "decision_failures": decision_failures,
        "decision_success_rate": round(decision_successes / decision_attempts, 4) if decision_attempts else None,
        "edge_attempts": edge_attempts,
        "edge_successes": edge_successes,
        "edge_failures": edge_failures,
        "passed_60": win_rate >= 0.60,
        "returncodes": returncodes,
        "log_dirs": log_dirs,
    }


def public_env(env: Mapping[str, str]) -> Dict[str, str]:
    keys = [
        "COTE_TOPOLOGY_INIT",
        "COTE_OPT_MODE",
        "COTE_EVOLVE",
        "COTE_DISABLE_EDGE_MESSAGES",
        "COTE_PROMPT_EVOLVE",
        "COTE_TOPOLOGY_UPDATE",
        "COTE_TOPOLOGY_PRUNE",
        "COTE_SAMPLE_ACTION",
        "COTE_NORMALIZE_TOPOLOGY",
        "COTE_ACTION_DROPOUT_RATE",
        "COTE_REWARD_CHANNEL",
        "COTE_ERROR_CHANNEL",
        "COTE_BELIEF_CHANNEL",
        "COTE_LOCAL_MODEL_BUDGET",
        "COTE_EDGE_LOCAL_MODEL",
        "COTE_LOCAL_MODEL_MAX_RANK_OVERRIDE",
        "COTE_LOCAL_MODEL_MAX_SCORE_DROP",
        "COTE_SEED",
        "LOCAL_MODEL_PATH",
    ]
    return {key: env[key] for key in keys if key in env}


def write_progress(output_dir: Path, suite: Dict[str, Any]) -> None:
    (output_dir / "full_reproduction.partial.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")


def write_outputs(output_dir: Path, suite: Dict[str, Any]) -> None:
    json_path = output_dir / "full_reproduction.json"
    json_path.write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_dir / "table_variants.csv", suite["table_variants"], "name")
    write_csv(output_dir / "ablations.csv", suite["ablations"], "name")
    write_csv(output_dir / "evolution_curve.csv", suite["evolution_curve"], "checkpoint")
    (output_dir / "REPORT.md").write_text(report_markdown(suite, output_dir), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], id_key: str) -> None:
    fields = [
        id_key,
        "episodes",
        "measured_wins",
        "episode_win_rate",
        "average_point_diff",
        "point_diff_sum",
        "average_edge_retention",
        "average_edge_count",
        "total_tokens",
        "average_total_tokens_per_deal",
        "decision_attempts",
        "decision_successes",
        "decision_success_rate",
        "passed_60",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            agg = row.get("aggregate", {})
            writer.writerow({id_key: row.get(id_key), **{field: agg.get(field) for field in fields if field != id_key}})


def report_markdown(suite: Dict[str, Any], output_dir: Path) -> str:
    lines = [
        "# Full Current-Platform Reproduction Report",
        "",
        "This report is generated from executable experiments against `Danzero_plus`.",
        "",
        "## Important Limitations",
        "",
        "- Exact Duplicate Randomization is not available because the offline server exposes no mirrored deal replay or fixed deal seed.",
        "- Local model quality depends on the checkpoint selected with `LOCAL_MODEL_PATH`.",
        "- Results are current-platform estimates, not a claim of exact Table 1 reproduction.",
        "",
        "## Paper Table 1 Variants",
        "",
        table_section(suite.get("table_variants", []), "name"),
        "",
        "## Channel Ablations",
        "",
        table_section(suite.get("ablations", []), "name"),
        "",
        "## Evolution Curve",
        "",
        table_section(suite.get("evolution_curve", []), "checkpoint"),
        "",
        "## Artifacts",
        "",
        f"- JSON: `{output_dir / 'full_reproduction.json'}`",
        f"- Table variants CSV: `{output_dir / 'table_variants.csv'}`",
        f"- Ablations CSV: `{output_dir / 'ablations.csv'}`",
        f"- Evolution CSV: `{output_dir / 'evolution_curve.csv'}`",
    ]
    return "\n".join(lines) + "\n"


def table_section(rows: List[Dict[str, Any]], id_key: str) -> str:
    header = f"| {id_key} | episodes | wins | win_rate | avg_point_diff | edge_retention | tokens | T8 success | passed_60 |\n"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    body = []
    for row in rows:
        agg = row.get("aggregate", {})
        attempts = agg.get("decision_attempts") or 0
        successes = agg.get("decision_successes") or 0
        body.append(
            "| {id} | {episodes} | {wins} | {win:.2%} | {diff} | {edge} | {tokens} | {succ}/{att} | {passed} |".format(
                id=row.get(id_key),
                episodes=agg.get("episodes", 0),
                wins=agg.get("measured_wins", 0),
                win=float(agg.get("episode_win_rate", 0.0) or 0.0),
                diff=agg.get("average_point_diff"),
                edge=agg.get("average_edge_retention"),
                tokens=agg.get("total_tokens"),
                succ=successes,
                att=attempts,
                passed=agg.get("passed_60"),
            )
        )
    return header + sep + "\n".join(body)


def suite_summary(suite: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "output": "full_reproduction.json",
        "table_variants": [
            {"name": row.get("name"), "aggregate": row.get("aggregate")} for row in suite.get("table_variants", [])
        ],
        "ablations": [{"name": row.get("name"), "aggregate": row.get("aggregate")} for row in suite.get("ablations", [])],
        "evolution_curve": [
            {"checkpoint": row.get("checkpoint"), "aggregate": row.get("aggregate")}
            for row in suite.get("evolution_curve", [])
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
