# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest
from dataclasses import replace

from cote_paper.game import Candidate, GameContext
from cote_paper.nodes import NODE_KEYS, OUTPUT_NODE
from cote_paper.policy import PaperCOTEAgent


class FakeNodeModel:
    enabled = True
    last_error = ""

    def __init__(self) -> None:
        self.node_calls: list[str] = []
        self.choice_payload = None

    def think_node(self, node_key, context, candidates, history_tail, edge_messages, topology_summary):
        self.node_calls.append(node_key)
        return {
            "node": node_key,
            "summary": f"{node_key} thought",
            "candidateScores": {"0": 0.0, "1": 1.0},
            "confidence": 0.75,
        }

    def choose_action(self, context, candidates, history_tail, edge_prompts, topology_summary, node_thoughts=None):
        self.choice_payload = node_thoughts or {}
        return candidates[0].index

    def stats(self):
        return {}


def context() -> GameContext:
    return GameContext(
        my_pos=0,
        partner_pos=2,
        opponents=(1, 3),
        hand_cards=["S3", "S4", "S5"],
        public_info=[{"rest": 3}, {"rest": 5}, {"rest": 4}, {"rest": 6}],
        self_rank="2",
        oppo_rank="2",
        cur_rank="2",
        cur_pos=None,
        cur_action=None,
        greater_pos=None,
        greater_action=None,
        stage="play",
    )


def candidates() -> list[Candidate]:
    return [
        Candidate(0, ["Single", "3", ["S3"]], 10.0, {key: 0.0 for key in NODE_KEYS}, 0.5, "low"),
        Candidate(1, ["Single", "4", ["S4"]], 10.0, {key: 0.0 for key in NODE_KEYS}, 0.5, "high"),
    ]


class LocalNodeThinkingTest(unittest.TestCase):
    def test_select_action_runs_seven_thinkers_then_t8_decider(self) -> None:
        agent = PaperCOTEAgent(client_id=0)
        agent.hp = replace(
            agent.hp,
            disable_edge_messages=True,
            edge_local_model=False,
            use_local_model=True,
            node_local_model=True,
            local_model_min_actions=2,
            local_model_budget=0,
        )
        agent.local_model = FakeNodeModel()

        choice = agent.select_action(
            {
                "stage": "play",
                "myPos": 0,
                "handCards": ["S3", "S4", "S5"],
                "publicInfo": [{"rest": 3}, {"rest": 5}, {"rest": 4}, {"rest": 6}],
                "curRank": "2",
                "greaterPos": None,
                "greaterAction": None,
                "actionList": [
                    ["Single", "3", ["S3"]],
                    ["Single", "4", ["S4"]],
                ],
                "indexRange": 1,
            }
        )

        self.assertEqual(choice, 1)
        self.assertEqual(agent.local_model.node_calls, [key for key in NODE_KEYS if key != OUTPUT_NODE])
        self.assertIn("T1_board_parser", agent.local_model.choice_payload)
        self.assertEqual(agent.node_model_successes, 7)
        self.assertEqual(agent.local_model_decision_successes, 1)

    def test_local_node_thinking_calls_all_non_output_nodes(self) -> None:
        agent = PaperCOTEAgent(client_id=0)
        agent.local_model = FakeNodeModel()

        thoughts = agent._run_local_node_thinking(context(), candidates(), [])

        self.assertEqual(agent.local_model.node_calls, [key for key in NODE_KEYS if key != OUTPUT_NODE])
        self.assertEqual(sorted(thoughts), sorted(key for key in NODE_KEYS if key != OUTPUT_NODE))
        self.assertEqual(agent.node_model_successes, 7)
        self.assertEqual(agent.node_model_failures, 0)

    def test_local_node_scores_change_candidate_order(self) -> None:
        agent = PaperCOTEAgent(client_id=0)
        thoughts = {
            key: {"candidateScores": {"0": -1.0, "1": 1.0}, "confidence": 1.0}
            for key in NODE_KEYS
            if key != OUTPUT_NODE
        }

        reranked = agent._apply_local_node_scores(candidates(), thoughts)

        self.assertEqual(reranked[0].index, 1)
        self.assertGreater(reranked[0].base_score, reranked[1].base_score)

    def test_t8_receives_prior_node_thoughts(self) -> None:
        agent = PaperCOTEAgent(client_id=0)
        agent.local_model = FakeNodeModel()
        thoughts = {"T1_board_parser": {"summary": "parsed board"}}

        agent.local_model.choose_action(context(), candidates(), [], {}, {}, node_thoughts=thoughts)

        self.assertEqual(agent.local_model.choice_payload, thoughts)


if __name__ == "__main__":
    unittest.main()
