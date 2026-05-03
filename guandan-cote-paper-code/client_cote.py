# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Dict, Optional

from ws4py.client.threadedclient import WebSocketClient

from cote_agent import CoteAgent, metrics_reference


class CoteClient(WebSocketClient):
    def __init__(self, url: str, client_id: int):
        super().__init__(url)
        self.client_id = client_id
        self.agent = CoteAgent(client_id=client_id)

    def opened(self) -> None:
        print(
            "COTE_CLIENT_READY "
            + json.dumps({"client_id": self.client_id, **metrics_reference()}, ensure_ascii=False),
            flush=True,
        )

    def closed(self, code: int, reason: Optional[str] = None) -> None:
        print(f"COTE_CLIENT_CLOSED code={code} reason={reason}", flush=True)

    def received_message(self, message: Any) -> None:
        try:
            msg = json.loads(str(message))
            self.agent.observe(msg)
            log_result_messages(msg)
            if "actionList" in msg:
                act_index = self.agent.select_action(msg)
                self.send(json.dumps({"actIndex": int(act_index)}))
        except Exception as exc:  # noqa: BLE001 - keep the websocket alive only if a fallback is possible.
            print(f"COTE_CLIENT_ERROR {exc}", flush=True)
            traceback.print_exc()
            if "actionList" in locals().get("msg", {}):
                self.send(json.dumps({"actIndex": 0}))


def log_result_messages(msg: Dict[str, Any]) -> None:
    stage = msg.get("stage")
    msg_type = msg.get("type")
    if msg_type != "notify":
        return
    if stage == "episodeOver":
        order = msg.get("order") or []
        if order:
            payload = {
                "order": order,
                "winnerTeam": int(order[0]) % 2,
                "curRank": msg.get("curRank"),
                "restCards": msg.get("restCards"),
            }
            print("EPISODE_RESULT " + json.dumps(payload, ensure_ascii=False), flush=True)
    elif stage == "gameResult":
        payload = {"final": msg.get("final") or msg.get("victoryNum"), "draws": msg.get("draws")}
        print("GAME_RESULT " + json.dumps(payload, ensure_ascii=False), flush=True)


def build_url(client_id: int) -> str:
    default_url = f"ws://127.0.0.1:23456/game/client{client_id}"
    return os.environ.get("GUANDAN_WS_URL", default_url)


def main(default_client_id: Optional[int] = None) -> None:
    parser = argparse.ArgumentParser(description="Run one local-model COTE Guandan client.")
    parser.add_argument("--client-id", type=int, default=default_client_id if default_client_id is not None else 1, choices=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    ws = CoteClient(build_url(args.client_id), args.client_id)
    try:
        ws.connect()
        ws.run_forever()
    except KeyboardInterrupt:
        ws.close()
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
