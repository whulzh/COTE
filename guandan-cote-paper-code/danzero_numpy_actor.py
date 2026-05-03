# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import pickle
import signal
import sys
import time
from multiprocessing import Event, Process
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import zmq


DANZERO_ROOT = Path(os.environ.get("DANZERO_ROOT", Path(__file__).resolve().parents[2] / "Danzero_plus")).expanduser()
DEFAULT_CKPT = DANZERO_ROOT / "wintest" / "danzero" / "q_network.ckpt"


class NumpyDQN:
    """DanZero_plus DQN inference path for the bundled q_network.ckpt."""

    def __init__(self, ckpt: Path) -> None:
        with ckpt.open("rb") as handle:
            weights = pickle.load(handle)
        if len(weights) != 12:
            raise ValueError(f"Expected 12 TensorFlow weight arrays, got {len(weights)}")
        self.weights: List[np.ndarray] = [np.asarray(item, dtype=np.float32) for item in weights]

    def forward(self, x_batch: Any) -> np.ndarray:
        x = np.asarray(x_batch, dtype=np.float32)
        for idx in range(0, 10, 2):
            x = np.tanh(x @ self.weights[idx] + self.weights[idx + 1])
        return x @ self.weights[10] + self.weights[11]

    def choose(self, state: dict[str, Any]) -> int:
        x_batch = state.get("x_batch")
        if x_batch is None:
            return 0
        values = self.forward(x_batch).reshape(-1)
        if values.size <= 0:
            return 0
        return int(np.argmax(values))


def _recv_object(payload: bytes) -> Any:
    return pickle.loads(payload)


def _send_object(obj: Any) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def run_actor(port: int, ckpt: Path, stop: Event) -> None:
    model = NumpyDQN(ckpt)
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.linger = 0
    socket.bind(f"tcp://*:{port}")
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    print(f"DANZERO_ACTOR_READY port={port}", flush=True)
    try:
        while not stop.is_set():
            events = dict(poller.poll(250))
            if socket not in events:
                continue
            try:
                state = _recv_object(socket.recv())
                action = model.choose(state)
                socket.send(_send_object(action))
            except Exception as exc:  # noqa: BLE001 - reply to keep the client alive.
                print(f"DANZERO_ACTOR_ERROR port={port} error={exc}", flush=True)
                socket.send(_send_object(0))
    finally:
        socket.close(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DanZero_plus DQN actors without TensorFlow 1.")
    parser.add_argument("--ports", default="6000,6002", help="Comma-separated REP ports used by DanZero clients.")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="Path to DanZero_plus q_network.ckpt.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    ports = [int(item) for item in args.ports.split(",") if item.strip()]
    stop = Event()

    def _stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    procs: List[Process] = []
    for port in ports:
        proc = Process(target=run_actor, args=(port, ckpt, stop), daemon=True)
        proc.start()
        procs.append(proc)
        time.sleep(0.2)
    try:
        while not stop.is_set() and all(proc.is_alive() for proc in procs):
            time.sleep(0.5)
    finally:
        stop.set()
        for proc in procs:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
