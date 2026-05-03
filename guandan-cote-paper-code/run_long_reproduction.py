# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import run_full_reproduction_suite as suite


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed long COTE reproduction experiments.")
    parser.add_argument("--mode", choices=["main", "table", "ablation", "all"], default="main")
    parser.add_argument("--only", default="", help="Comma-separated experiment names to run within the selected mode.")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--episodes-per-seat", type=int, default=500)
    parser.add_argument(
        "--server-games",
        type=int,
        default=0,
        help="Offline server game budget per seat direction. 0 means use episodes-per-seat.",
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--topology-init", choices=["dense"], default="dense")
    parser.add_argument("--use-local-model", action="store_true")
    parser.add_argument("--local-model-budget", type=int, default=1)
    parser.add_argument("--max-chunks-per-seat", type=int, default=50)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    output_dir = Path(args.output_dir) if args.output_dir else ROOT.parent / ".run_logs" / (
        "long_reproduction_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runner_args = argparse.Namespace(
        topology_init=args.topology_init,
        use_local_model=args.use_local_model,
        local_model_budget=args.local_model_budget,
        server_games=args.server_games if args.server_games > 0 else args.episodes_per_seat,
        timeout=args.timeout,
    )

    payload: Dict[str, Any] = {
        "metadata": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "mode": args.mode,
            "seeds": seeds,
            "episodes_per_seat": args.episodes_per_seat,
            "episodes_per_seed": args.episodes_per_seat * len(suite.SEAT_DIRECTIONS),
            "topology_init": args.topology_init,
            "use_local_model": args.use_local_model,
            "local_model_budget_per_client": args.local_model_budget if args.use_local_model else 0,
            "opponent": "Danzero_plus q_network.ckpt via numpy actor",
            "seat_protocol": "bidirectional seat-swap approximation",
            "duplicate_randomization": "not exact; offline server exposes no mirrored deal replay",
        },
        "experiments": [],
    }

    plan = experiment_plan(args.mode, args.only)
    for seed in seeds:
        print(f"SEED {seed} start", flush=True)
        with temporary_env({"COTE_SEED": str(seed)}):
            for group, name, env_updates in plan:
                run_dir = output_dir / f"seed_{seed:03d}" / group / name
                records = run_repeated_bidirectional(
                    name,
                    env_updates,
                    runner_args,
                    run_dir,
                    args.episodes_per_seat,
                    seed,
                    args.max_chunks_per_seat,
                )
                aggregate = suite.aggregate(records)
                payload["experiments"].append(
                    {
                        "group": group,
                        "name": name,
                        "seed": seed,
                        "aggregate": aggregate,
                        "runs": records,
                    }
                )
                write_outputs(output_dir, payload)
        print(f"SEED {seed} done", flush=True)

    write_outputs(output_dir, payload)
    print(json.dumps({"output_dir": str(output_dir), "summary": summarize(payload)}, ensure_ascii=False, indent=2))
    return 0


def parse_seed_list(raw: str) -> List[int]:
    values: List[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(item))
    seen = set()
    deduped = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    if not deduped:
        raise SystemExit("--seeds produced an empty seed list")
    return deduped


def experiment_plan(mode: str, only: str = "") -> List[tuple[str, str, Mapping[str, str]]]:
    selected = {item.strip() for item in only.split(",") if item.strip()}
    items: List[tuple[str, str, Mapping[str, str]]] = []
    if mode in ("main", "all"):
        items.append(("main", "cote", suite.TABLE_VARIANTS["cote"]))
    if mode in ("table", "all"):
        items.extend(("table", name, env_updates) for name, env_updates in suite.TABLE_VARIANTS.items())
    if mode in ("ablation", "all"):
        items.extend(("ablation", name, env_updates) for name, env_updates in suite.ABLATIONS.items())
    if selected:
        items = [item for item in items if item[1] in selected]
        missing = selected.difference({item[1] for item in items})
        if missing:
            raise SystemExit(f"--only did not match experiments: {', '.join(sorted(missing))}")
    return items


def run_repeated_bidirectional(
    name: str,
    env_updates: Mapping[str, str],
    args: argparse.Namespace,
    output_dir: Path,
    episodes_per_seat: int,
    base_seed: int,
    max_chunks_per_seat: int,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for direction_index, seats in enumerate(suite.SEAT_DIRECTIONS):
        seat_key = seats.replace(",", "_")
        completed = 0
        chunk = 0
        state_root = output_dir / f"seats_{seat_key}"
        while completed < episodes_per_seat and chunk < max_chunks_per_seat:
            remaining = episodes_per_seat - completed
            env = suite.base_env(args, state_root)
            env.update(env_updates)
            env["COTE_SEED"] = str(base_seed * 100000 + direction_index * 1000 + chunk)
            summary = run_one_chunk(name, seats, remaining, chunk, args, env, output_dir)
            summary["experiment"] = name
            summary["seat_direction"] = seats
            summary["chunk_index"] = chunk
            summary["requested_remaining_before_chunk"] = remaining
            summary["public_env"] = suite.public_env(env)
            records.append(summary)
            completed += int(summary.get("episodes") or 0)
            (output_dir / f"{name}_{seat_key}_chunk{chunk:02d}.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "name": name,
                        "seat_direction": seats,
                        "chunk": chunk,
                        "seat_completed": completed,
                        "seat_target": episodes_per_seat,
                        "aggregate_so_far": suite.aggregate(records),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if int(summary.get("episodes") or 0) <= 0:
                break
            chunk += 1
        if completed < episodes_per_seat:
            raise RuntimeError(
                f"{name} seats={seats} reached {completed}/{episodes_per_seat} episodes "
                f"after {chunk} chunks; see {output_dir}"
            )
    return records


def run_one_chunk(
    name: str,
    seats: str,
    episodes: int,
    chunk: int,
    args: argparse.Namespace,
    env: Mapping[str, str],
    output_dir: Path,
) -> Dict[str, Any]:
    command = [
        sys.executable,
        str(suite.CLIENTS_DIR / "evaluate_cote.py"),
        "--policy",
        "cote",
        "--opponent-policy",
        "danzero",
        "--our-clients",
        seats,
        "--episodes",
        str(episodes),
        "--server-games",
        str(args.server_games),
        "--timeout",
        str(args.timeout),
        "--stop-on-game-result",
        "--keep-logs",
    ]
    if args.use_local_model:
        command.append("--use-local-model")
    print(f"RUN {name} seats={seats} chunk={chunk} remaining={episodes}", flush=True)
    try:
        proc = subprocess.run(
            command,
            cwd=str(suite.CLIENTS_DIR),
            env=dict(env),
            text=True,
            capture_output=True,
            timeout=max(float(args.timeout) + 90.0, 120.0),
        )
        summary = suite.parse_summary(proc.stdout)
        summary["returncode"] = proc.returncode
        summary["stderr_tail"] = proc.stderr[-1200:]
        return summary
    except subprocess.TimeoutExpired as exc:
        return {
            "parse_error": f"chunk subprocess timeout after {exc.timeout} seconds",
            "episodes": 0,
            "returncode": 124,
            "stdout_tail": (exc.stdout or "")[-1200:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-1200:] if isinstance(exc.stderr, str) else "",
            "chunk_output_dir": str(output_dir),
        }


@contextmanager
def temporary_env(updates: Mapping[str, str]) -> Iterable[None]:
    old_values = {key: os.environ.get(key) for key in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def summarize(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in payload.get("experiments", []):
        key = (str(row.get("group")), str(row.get("name")))
        grouped.setdefault(key, []).append(row)

    summary: List[Dict[str, Any]] = []
    for (group, name), rows in sorted(grouped.items()):
        aggs = [row.get("aggregate", {}) for row in rows]
        seed_rates = [float(agg.get("episode_win_rate", 0.0) or 0.0) for agg in aggs]
        total_episodes = sum(int(agg.get("episodes") or 0) for agg in aggs)
        total_wins = sum(int(agg.get("measured_wins") or 0) for agg in aggs)
        total_point_diff = sum(int(agg.get("point_diff_sum") or 0) for agg in aggs)
        rate_std = stdev(seed_rates) if len(seed_rates) > 1 else 0.0
        ci95 = 1.96 * rate_std / math.sqrt(len(seed_rates)) if seed_rates else 0.0
        summary.append(
            {
                "group": group,
                "name": name,
                "seeds": [row.get("seed") for row in rows],
                "episodes": total_episodes,
                "measured_wins": total_wins,
                "pooled_win_rate": round(total_wins / total_episodes, 4) if total_episodes else 0.0,
                "mean_seed_win_rate": round(mean(seed_rates), 4) if seed_rates else 0.0,
                "std_seed_win_rate": round(rate_std, 4),
                "ci95_seed_win_rate": round(ci95, 4),
                "point_diff_sum": total_point_diff,
                "average_point_diff": round(total_point_diff / total_episodes, 4) if total_episodes else None,
                "passed_60_pooled": (total_wins / total_episodes) >= 0.60 if total_episodes else False,
            }
        )
    return summary


def write_outputs(output_dir: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(payload)
    enriched = {**payload, "summary": summary}
    (output_dir / "long_reproduction.json").write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_dir / "long_reproduction_summary.csv", summary)
    (output_dir / "REPORT.md").write_text(report_markdown(enriched, output_dir), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "group",
        "name",
        "seeds",
        "episodes",
        "measured_wins",
        "pooled_win_rate",
        "mean_seed_win_rate",
        "std_seed_win_rate",
        "ci95_seed_win_rate",
        "average_point_diff",
        "passed_60_pooled",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = dict(row)
            values["seeds"] = ",".join(str(item) for item in row.get("seeds", []))
            writer.writerow({field: values.get(field) for field in fields})


def report_markdown(payload: Mapping[str, Any], output_dir: Path) -> str:
    lines = [
        "# Long Current-Platform Reproduction Report",
        "",
        "## Protocol",
        "",
        f"- Mode: `{payload.get('metadata', {}).get('mode')}`",
        f"- Seeds: `{payload.get('metadata', {}).get('seeds')}`",
        f"- Episodes per seed: `{payload.get('metadata', {}).get('episodes_per_seed')}`",
        f"- Topology init: `{payload.get('metadata', {}).get('topology_init')}`",
        f"- Local model enabled: `{payload.get('metadata', {}).get('use_local_model')}`",
        "- Duplicate Randomization: unavailable in this offline server, so this is a seat-swap approximation.",
        "",
        "## Summary",
        "",
        "| group | name | episodes | wins | pooled_win_rate | mean_seed_win_rate | std | 95% CI | avg_point_diff | passed_60 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("summary", []):
        lines.append(
            "| {group} | {name} | {episodes} | {wins} | {pooled:.2%} | {mean:.2%} | {std:.2%} | +/-{ci:.2%} | {diff} | {passed} |".format(
                group=row.get("group"),
                name=row.get("name"),
                episodes=row.get("episodes"),
                wins=row.get("measured_wins"),
                pooled=float(row.get("pooled_win_rate", 0.0) or 0.0),
                mean=float(row.get("mean_seed_win_rate", 0.0) or 0.0),
                std=float(row.get("std_seed_win_rate", 0.0) or 0.0),
                ci=float(row.get("ci95_seed_win_rate", 0.0) or 0.0),
                diff=row.get("average_point_diff"),
                passed=row.get("passed_60_pooled"),
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSON: `{output_dir / 'long_reproduction.json'}`",
            f"- CSV: `{output_dir / 'long_reproduction_summary.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
