# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from typing import Any, Dict, Optional

from ws4py.client.threadedclient import WebSocketClient


class QuietRandomClient(WebSocketClient):
    def __init__(self, url: str, client_id: int):
        super().__init__(url)
        self.client_id = client_id
        self.random = random.Random(os.getpid())

    def opened(self) -> None:
        print(f"RANDOM_CLIENT_READY client_id={self.client_id}", flush=True)

    def closed(self, code: int, reason: Optional[str] = None) -> None:
        print(f"RANDOM_CLIENT_CLOSED code={code} reason={reason}", flush=True)

    def received_message(self, message: Any) -> None:
        try:
            msg = json.loads(str(message))
            log_result_messages(msg)
            if "actionList" in msg:
                max_index = int(msg.get("indexRange", len(msg.get("actionList") or []) - 1))
                self.send(json.dumps({"actIndex": self.random.randint(0, max(0, max_index))}))
        except Exception:
            traceback.print_exc()
            sys.exit(1)


def log_result_messages(msg: Dict[str, Any]) -> None:
    if msg.get("type") != "notify":
        return
    if msg.get("stage") == "episodeOver":
        order = msg.get("order") or []
        if order:
            print(
                "EPISODE_RESULT "
                + json.dumps({"order": order, "winnerTeam": int(order[0]) % 2, "curRank": msg.get("curRank")}),
                flush=True,
            )
    elif msg.get("stage") == "gameResult":
        print("GAME_RESULT " + json.dumps({"final": msg.get("final") or msg.get("victoryNum")}), flush=True)


def main(default_client_id: Optional[int] = None) -> None:
    parser = argparse.ArgumentParser(description="Run one quiet random Guandan client.")
    parser.add_argument("--client-id", type=int, default=default_client_id if default_client_id is not None else 1, choices=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    url = os.environ.get("GUANDAN_WS_URL", f"ws://127.0.0.1:23456/game/client{args.client_id}")
    ws = QuietRandomClient(url, args.client_id)
    try:
        ws.connect()
        ws.run_forever()
    except KeyboardInterrupt:
        ws.close()


if __name__ == "__main__":
    main()
