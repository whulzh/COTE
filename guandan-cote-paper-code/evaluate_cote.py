# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from cote_agent import TARGET_WIN_RATE


CLIENTS_DIR = Path(__file__).resolve().parent
PLATFORM_ROOT = Path(os.environ.get("GUANDAN_PLATFORM_ROOT", CLIENTS_DIR.parent)).expanduser()
DANZERO_ROOT = Path(os.environ.get("DANZERO_ROOT", PLATFORM_ROOT.parent / "Danzero_plus")).expanduser()
WINDOWS_SERVER = PLATFORM_ROOT / "windows" / "guandan_offline_v1006.exe"
UBUNTU_SERVER = PLATFORM_ROOT / "ubuntu" / "guandan_offline_v1006"
DANZERO_DIR = DANZERO_ROOT / "wintest" / "danzero"
DANZERO_TORCH_DIR = DANZERO_ROOT / "wintest" / "torch"
DANZERO_COMPAT_DIR = CLIENTS_DIR / "danzero_compat"
DANZERO_NUMPY_ACTOR = CLIENTS_DIR / "danzero_numpy_actor.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate COTE or random teams on the offline Guandan platform.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Stop after this many episodeOver messages. This is the paper-aligned per-deal unit.",
    )
    parser.add_argument("--server-games", type=int, default=1, help="Argument passed to the offline platform.")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--target-win-rate", type=float, default=TARGET_WIN_RATE)
    parser.add_argument("--our-clients", default="1,3", help="Client ids in the measured team: 1,3 or 0,2.")
    parser.add_argument("--policy", choices=["cote", "random", "danzero"], default="cote", help="Policy used by the measured team.")
    parser.add_argument(
        "--opponent-policy",
        choices=["cote", "random", "danzero"],
        default="random",
        help="Policy used by the other team.",
    )
    parser.add_argument("--use-local-model", action="store_true", help="Enable local model calls during evaluation.")
    parser.add_argument(
        "--stop-on-game-result",
        action="store_true",
        help="Stop when the platform emits GAME_RESULT. Leave off for paper-style per-deal evaluation.",
    )
    parser.add_argument("--keep-logs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    our_clients = {int(item) for item in args.our_clients.split(",") if item.strip()}
    our_positions = {client_id_to_position(client_id) for client_id in our_clients}
    our_team = 0 if our_positions == {0, 2} else 1 if our_positions == {1, 3} else None
    if our_team is None:
        raise SystemExit("--our-clients must be one team: 1,3 or 2,4")

    log_dir = CLIENTS_DIR.parent / ".run_logs" / ("eval_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("COTE_STATE_PATH", str(log_dir / "state_client{client_id}.json"))
    if args.use_local_model:
        if not local_model_available(env):
            raise SystemExit(
                "--use-local-model requires LOCAL_MODEL_PATH or COTE_LOCAL_MODEL_PATH in the environment."
            )
        env["COTE_USE_LOCAL_MODEL"] = "1"
    else:
        env["COTE_USE_LOCAL_MODEL"] = "0"

    procs: List[subprocess.Popen[Any]] = []
    try:
        kill_existing_platform()
        server = start_server(args.server_games, log_dir, env)
        procs.append(server)
        time.sleep(1.0)
        procs.extend(start_policy_clients(args, our_clients, log_dir, env))

        summary = monitor(
            log_dir,
            our_team,
            args.episodes,
            args.timeout,
            args.target_win_rate,
            our_clients,
            args.stop_on_game_result,
        )
        summary["measured_policy"] = args.policy
        summary["opponent_policy"] = args.opponent_policy
        summary["measured_clients"] = sorted(our_clients)
        summary["measured_positions"] = sorted(our_positions)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        (log_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0 if summary["passed"] else 2
    finally:
        terminate_all(procs)
        kill_existing_platform()
        if not args.keep_logs:
            maybe_remove_logs(log_dir)


def client_id_to_position(client_id: int) -> int:
    """Map platform websocket client id (1..4) to game seat id (0..3)."""

    return client_id % 4


def local_model_available(env: Dict[str, str]) -> bool:
    if env.get("LOCAL_MODEL_PATH") or env.get("COTE_LOCAL_MODEL_PATH"):
        return True
    try:
        from cote_paper.config import LOCAL_MODEL_PATH
    except Exception:
        return False
    return bool(LOCAL_MODEL_PATH)


def start_server(server_games: int, log_dir: Path, env: Dict[str, str]) -> subprocess.Popen[Any]:
    server = WINDOWS_SERVER if os.name == "nt" else UBUNTU_SERVER
    if not server.exists():
        raise FileNotFoundError(server)
    return subprocess.Popen(
        [str(server), str(server_games)],
        cwd=str(server.parent),
        stdout=(log_dir / "server.out.log").open("w", encoding="utf-8", errors="replace"),
        stderr=(log_dir / "server.err.log").open("w", encoding="utf-8", errors="replace"),
        env=env,
    )


def start_client(client_id: int, use_cote: bool, log_dir: Path, env: Dict[str, str]) -> subprocess.Popen[Any]:
    script = CLIENTS_DIR / ("client_cote.py" if use_cote else "client_random_quiet.py")
    label = "cote" if use_cote else "random"
    return subprocess.Popen(
        [sys.executable, str(script), "--client-id", str(client_id)],
        cwd=str(CLIENTS_DIR),
        stdout=(log_dir / f"client{client_id}_{label}.out.log").open("w", encoding="utf-8", errors="replace"),
        stderr=(log_dir / f"client{client_id}_{label}.err.log").open("w", encoding="utf-8", errors="replace"),
        env=env,
    )


def start_policy_clients(
    args: argparse.Namespace,
    our_clients: set[int],
    log_dir: Path,
    env: Dict[str, str],
) -> List[subprocess.Popen[Any]]:
    if args.policy == "danzero":
        raise SystemExit("DanZero_plus is wired as the fixed baseline opponent; use --opponent-policy danzero.")

    if args.opponent_policy != "danzero":
        procs: List[subprocess.Popen[Any]] = []
        for client_id in range(1, 5):
            use_cote = (args.policy == "cote" and client_id in our_clients) or (
                args.opponent_policy == "cote" and client_id not in our_clients
            )
            procs.append(start_client(client_id, use_cote, log_dir, env))
            time.sleep(0.25)
        return procs

    if our_clients not in ({0, 2}, {1, 3}):
        raise SystemExit(
            "DanZero_plus mirrored evaluation supports measured seats --our-clients 0,2 or --our-clients 1,3."
        )

    danzero_ports = "6000,6002" if our_clients == {1, 3} else "6001,6003"
    procs = [start_danzero_actor(log_dir, env, danzero_ports)]
    time.sleep(1.0)
    if our_clients == {1, 3}:
        # Match Danzero_plus/wintest/torch/testvsdqn.sh seating:
        # DanZero client0, measured client1, DanZero client2, measured client3.
        launch_order = [("danzero", 0), (args.policy, 1), ("danzero", 2), (args.policy, 3)]
    else:
        # Mirrored seating for the same two teams.
        launch_order = [(args.policy, 0), ("danzero", 1), (args.policy, 2), ("danzero", 3)]
    for policy, client_id in launch_order:
        if policy == "danzero":
            procs.append(start_danzero_client(client_id, log_dir, env))
        else:
            procs.append(start_client(client_id, policy == "cote", log_dir, env))
        time.sleep(0.35)
    return procs


def start_danzero_actor(log_dir: Path, env: Dict[str, str], ports: str) -> subprocess.Popen[Any]:
    if not DANZERO_NUMPY_ACTOR.exists():
        raise FileNotFoundError(DANZERO_NUMPY_ACTOR)
    if not (DANZERO_DIR / "q_network.ckpt").exists():
        raise FileNotFoundError(DANZERO_DIR / "q_network.ckpt")
    return subprocess.Popen(
        [sys.executable, str(DANZERO_NUMPY_ACTOR), "--ports", ports, "--ckpt", str(DANZERO_DIR / "q_network.ckpt")],
        cwd=str(CLIENTS_DIR),
        stdout=(log_dir / "danzero_actor.out.log").open("w", encoding="utf-8", errors="replace"),
        stderr=(log_dir / "danzero_actor.err.log").open("w", encoding="utf-8", errors="replace"),
        env=env,
    )


def start_danzero_client(client_id: int, log_dir: Path, env: Dict[str, str]) -> subprocess.Popen[Any]:
    base_dir = DANZERO_DIR if client_id in {0, 2} else DANZERO_TORCH_DIR
    script = base_dir / f"client{client_id}.py"
    if not script.exists():
        raise FileNotFoundError(script)
    compat_env = env.copy()
    existing_path = compat_env.get("PYTHONPATH")
    compat_env["PYTHONPATH"] = str(DANZERO_COMPAT_DIR) + (os.pathsep + existing_path if existing_path else "")
    return subprocess.Popen(
        [sys.executable, str(script)],
        cwd=str(base_dir),
        stdout=(log_dir / f"client{client_id}_danzero.out.log").open("w", encoding="utf-8", errors="replace"),
        stderr=(log_dir / f"client{client_id}_danzero.err.log").open("w", encoding="utf-8", errors="replace"),
        env=compat_env,
    )


def monitor(
    log_dir: Path,
    our_team: int,
    target_episodes: int,
    timeout: float,
    target_win_rate: float,
    our_clients: Iterable[int],
    stop_on_game_result: bool = False,
) -> Dict[str, Any]:
    start = time.time()
    team_wins = [0, 0]
    game_final: Optional[List[int]] = None

    while time.time() - start < timeout:
        team_wins = [0, 0]
        episode_payloads = read_payloads(log_dir, "EPISODE_RESULT", our_clients)[:target_episodes]
        for payload in episode_payloads:
            winner = int(payload.get("winnerTeam", -1))
            if winner in (0, 1):
                team_wins[winner] += 1

        game_payloads = read_payloads(log_dir, "GAME_RESULT", our_clients)
        if game_payloads:
            final = game_payloads[-1].get("final")
            if isinstance(final, list) and len(final) >= 4:
                game_final = [int(value) for value in final[:4]]

        total = sum(team_wins)
        if total >= target_episodes or (stop_on_game_result and game_final is not None):
            break
        time.sleep(0.5)

    episode_payloads = read_payloads(log_dir, "EPISODE_RESULT", our_clients)[:target_episodes]
    team_wins = [0, 0]
    for payload in episode_payloads:
        winner = int(payload.get("winnerTeam", -1))
        if winner in (0, 1):
            team_wins[winner] += 1
    episode_total = sum(team_wins)
    point_diffs = [point_diff_from_order(payload.get("order"), our_team) for payload in episode_payloads]
    point_diffs = [value for value in point_diffs if value is not None]
    point_diff_sum = sum(point_diffs)
    average_point_diff = point_diff_sum / len(point_diffs) if point_diffs else None
    episode_win_rate = team_wins[our_team] / episode_total if episode_total else None
    match_wins = None
    match_total = None
    match_win_rate = None
    if game_final:
        team0_match_wins = (game_final[0] + game_final[2]) / 2.0
        team1_match_wins = (game_final[1] + game_final[3]) / 2.0
        match_wins_by_team = [team0_match_wins, team1_match_wins]
        match_wins = match_wins_by_team[our_team]
        match_total = team0_match_wins + team1_match_wins
        match_win_rate = match_wins / match_total if match_total else None
        opp_team = 1 - our_team
        opp_match_wins = match_wins_by_team[opp_team]
        source = "gameResult" if episode_total < target_episodes else "episodeOver"
    else:
        opp_match_wins = None
        source = "episodeOver"

    measured_win_rate = episode_win_rate if episode_win_rate is not None else 0.0
    cote_stats = latest_cote_stats(log_dir, set(our_clients))
    edge_counts = [stats.get("current_edge_count") for stats in cote_stats.values() if stats.get("current_edge_count") is not None]
    edge_retentions = [
        stats.get("current_edge_retention")
        for stats in cote_stats.values()
        if stats.get("current_edge_retention") is not None
    ]
    token_usage = aggregate_token_usage(cote_stats.values())
    model_control = aggregate_model_control(cote_stats.values())
    average_tokens_per_deal = token_usage["total_tokens"] / episode_total if episode_total else 0.0

    return {
        "source": source,
        "metric_note": "Per-deal episode win rate is reported from the current run; gameResult is platform-only auxiliary.",
        "platform_experiment_note": "Current platform statistics are seat-swap/random-opponent approximations unless a fixed strong opponent and mirrored deal control are supplied.",
        "episodes": episode_total,
        "team_wins": {"team0": team_wins[0], "team1": team_wins[1]},
        "our_team": our_team,
        "client_position_mapping": "platform client id maps to seat by client_id % 4",
        "point_diff_sum": point_diff_sum,
        "average_point_diff": round(average_point_diff, 4) if average_point_diff is not None else None,
        "point_diff_rule": "winner team advances 3/2/1 levels when partner finishes 2nd/3rd/4th; sign is from measured team's perspective.",
        "platform_match_wins": match_wins,
        "platform_opponent_match_wins": opp_match_wins,
        "platform_match_total": match_total,
        "platform_match_win_rate": round(match_win_rate, 4) if match_win_rate is not None else None,
        "episode_win_rate": round(episode_win_rate, 4) if episode_win_rate is not None else None,
        "target_win_rate": target_win_rate,
        "passed": measured_win_rate >= target_win_rate,
        "edge_retention": {
            "measured_client_stats": cote_stats,
            "average_edge_count": round(sum(edge_counts) / len(edge_counts), 4) if edge_counts else None,
            "average_edge_retention": round(sum(edge_retentions) / len(edge_retentions), 4) if edge_retentions else None,
        },
        "token_usage": {
            **token_usage,
            "average_total_tokens_per_deal": round(average_tokens_per_deal, 4),
        },
        "model_control": model_control,
        "log_dir": str(log_dir),
    }


def point_diff_from_order(order: Any, our_team: int) -> Optional[int]:
    if not isinstance(order, list) or len(order) < 4:
        return None
    try:
        finish_order = [int(pos) for pos in order[:4]]
    except (TypeError, ValueError):
        return None
    winner_team = finish_order[0] % 2
    partner_finish_index = next(
        (idx for idx, pos in enumerate(finish_order[1:], start=1) if pos % 2 == winner_team),
        3,
    )
    advance = {1: 3, 2: 2, 3: 1}.get(partner_finish_index, 1)
    return advance if winner_team == our_team else -advance


def latest_cote_stats(log_dir: Path, measured_clients: set[int]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for path in sorted(log_dir.glob("client*_cote.out.log")):
        match = re.search(r"client(\d+)_", path.name)
        if not match:
            continue
        client_id = int(match.group(1))
        if client_id not in measured_clients:
            continue
        payloads = read_payloads_from_paths([path], "COTE_STATS")
        if payloads:
            stats[str(client_id)] = payloads[-1]
    return stats


def aggregate_token_usage(stats_items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "successful_calls": 0,
        "failed_calls": 0,
    }
    for stats in stats_items:
        usage = stats.get("local_model_usage", {})
        if not isinstance(usage, dict):
            continue
        for key in totals:
            totals[key] += int(usage.get(key, 0) or 0)
    return totals


def aggregate_model_control(stats_items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    totals = {
        "decision_attempts": 0,
        "decision_successes": 0,
        "decision_failures": 0,
        "edge_attempts": 0,
        "edge_successes": 0,
        "edge_failures": 0,
    }
    for stats in stats_items:
        decision = stats.get("local_model_decision") if isinstance(stats, dict) else None
        edge = stats.get("edge_local_model") if isinstance(stats, dict) else None
        if isinstance(decision, dict):
            totals["decision_attempts"] += int(decision.get("attempts", 0) or 0)
            totals["decision_successes"] += int(decision.get("successes", 0) or 0)
            totals["decision_failures"] += int(decision.get("failures", 0) or 0)
        if isinstance(edge, dict):
            totals["edge_attempts"] += int(edge.get("attempts", 0) or 0)
            totals["edge_successes"] += int(edge.get("successes", 0) or 0)
            totals["edge_failures"] += int(edge.get("failures", 0) or 0)
    return totals


def read_payloads(log_dir: Path, prefix: str, preferred_clients: Iterable[int]) -> List[Dict[str, Any]]:
    paths: List[Path] = []
    for client_id in preferred_clients:
        paths.extend(sorted(log_dir.glob(f"client{client_id}_*.out.log")))
    if not paths:
        paths = sorted(log_dir.glob("client*.out.log"))[:1]
    else:
        paths = paths[:1]
    return read_payloads_from_paths(paths, prefix)


def read_payloads_from_paths(paths: Iterable[Path], prefix: str) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    pattern = re.compile(re.escape(prefix) + r"\s+(\{.*\})")
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            continue
        for match in pattern.finditer(text):
            try:
                payloads.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                continue
    return payloads


def terminate_all(procs: Iterable[subprocess.Popen[Any]]) -> None:
    for proc in procs:
        if proc.poll() is None:
            terminate_process_tree(proc)
    deadline = time.time() + 3.0
    for proc in procs:
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        if proc.poll() is None:
            terminate_process_tree(proc, force=True)


def terminate_process_tree(proc: subprocess.Popen[Any], force: bool = False) -> None:
    if os.name == "nt":
        flags = ["/T", "/PID", str(proc.pid)]
        if force:
            flags.insert(0, "/F")
        subprocess.run(["taskkill", *flags], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    if force:
        proc.kill()
    else:
        proc.terminate()


def kill_existing_platform() -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/IM", "guandan_offline_v1006.exe", "/T"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.run(["pkill", "-f", "guandan_offline_v1006"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def maybe_remove_logs(log_dir: Path) -> None:
    summary = log_dir / "summary.json"
    if summary.exists():
        keep_dir = log_dir.parent / (log_dir.name + "_summary")
        keep_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary, keep_dir / "summary.json")
    shutil.rmtree(log_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
