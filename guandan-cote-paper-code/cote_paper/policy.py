# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .belief import BeliefSnapshot, BeliefTracker
from .config import (
    LOCAL_MODEL_PATH,
    NODE_COUNT,
    TARGET_WIN_RATE,
    HyperParams,
)
from .el_bdpea import ELBDPEA
from .fitness import CoteFitnessEvaluator, edge_record_from_message
from .game import (
    ACTION_TYPE_BONUS,
    Candidate,
    GameContext,
    action_cards,
    action_name,
    action_rank,
    card_value,
    high_card_cost,
    is_bomb,
    is_pass,
    estimated_turns,
    normalize_actions,
    rank_value,
    residual_shape_score,
    softmax,
)
from .local_model import LocalCausalLM
from .nodes import NODE_KEYS, OUTPUT_NODE
from .prompts import PromptPopulation, decode_soft_prompt, edge_target_distribution, prompt_distribution
from .soft_prompt import SoftPromptTrainer
from .topology import CoteTopologyState, Edge, load_topology
from .trajectory import EdgeRecord, EpisodeRecord, TrajectoryLogger


class PaperCOTEAgent:
    """COTE agent with local-model inference and trainable edge prompts."""

    def __init__(self, client_id: Optional[int] = None, seed: Optional[int] = None) -> None:
        self.client_id = client_id
        self.my_pos = client_id % 4 if client_id else 0
        self.hp = HyperParams()
        self.seed = self._resolve_seed(seed, client_id)
        self.rng = random.Random(self.seed)
        self.debug = os.environ.get("COTE_DEBUG", "0") == "1"
        self.state_path = self._state_path(client_id)
        self.topology, self.population = self._load_or_init_state()
        self.population.ensure_edges(self.topology.all_edges(), self.rng, self.hp)
        self.belief = BeliefTracker(my_pos=self.my_pos)
        self.logger = TrajectoryLogger(client_id=client_id)
        self.evaluator = CoteFitnessEvaluator(self.hp, self.rng)
        self.evolver = ELBDPEA(self.hp, self.rng)
        self.local_model = LocalCausalLM()
        self.soft_prompt_trainer = SoftPromptTrainer(self.hp, self.local_model)
        self.local_model_used = 0
        self.local_model_decision_successes = 0
        self.local_model_decision_failures = 0
        self.edge_model_used = 0
        self.edge_model_successes = 0
        self.edge_model_failures = 0
        self.last_decision: Optional[Dict[str, Any]] = None
        self.last_context: Optional[GameContext] = None
        self.last_snapshot: Optional[BeliefSnapshot] = None

    def observe(self, msg: Mapping[str, Any]) -> None:
        if "myPos" in msg:
            self.my_pos = int(msg["myPos"])
            self.belief.my_pos = self.my_pos
        self.belief.observe(msg)
        self.logger.observe(msg)
        if msg.get("stage") == "episodeOver" and msg.get("type") == "notify":
            context = self.last_context or GameContext.from_message(dict(msg), self.my_pos)
            snapshot = self.belief.snapshot(context)
            completed = self.logger.mark_episode_over(msg, self.my_pos)
            if completed is not None:
                self._offline_update(completed, snapshot)

    def select_action(self, msg: Mapping[str, Any]) -> int:
        context = GameContext.from_message(dict(msg), self.my_pos)
        self.my_pos = context.my_pos
        self.last_context = context
        actions = normalize_actions(msg.get("actionList", []))
        max_index = int(msg.get("indexRange", len(actions) - 1))
        if not actions:
            return 0
        if len(actions) == 1 or max_index <= 0:
            return 0

        snapshot = self.belief.snapshot(context)
        self.last_snapshot = snapshot
        edge_records = self._run_edge_communication(context, snapshot)
        candidates, action_distribution = self._rank_candidates(context, actions, edge_records)
        chosen = self._choose_from_distribution(candidates, action_distribution)
        source = "cote_sample" if self.hp.sample_action else "cote_argmax"

        if self._should_ask_local_model(context, actions, candidates):
            edge_prompts = self.population.active_prompt_vectors()
            model_idx = self.local_model.choose_action(
                context,
                candidates[: min(8, len(candidates))],
                self.belief.history_tail() if self.hp.belief_channel else [],
                edge_prompts,
                self.topology.runtime_summary(),
            )
            self.local_model_used += 1
            if model_idx is not None and self._accept_local_model_choice(model_idx, candidates):
                chosen = model_idx
                source = "local_model_t8"
                self.local_model_decision_successes += 1
            else:
                self.local_model_decision_failures += 1
                if self.debug and self.local_model.last_error:
                    print(f"COTE_LOCAL_MODEL_ERROR {self.local_model.last_error}", flush=True)

        dropped = self._action_dropout(candidates)
        if dropped is not None:
            chosen = dropped
            source = "baseline_dropout"

        if not 0 <= chosen <= max_index:
            chosen = candidates[0].index if 0 <= candidates[0].index <= max_index else 0
            source = "repair"

        self.logger.record_decision(context, candidates, chosen, action_distribution, source, edge_records)
        self.last_decision = {
            "source": source,
            "actIndex": chosen,
            "myPos": context.my_pos,
            "partnerPos": context.partner_pos,
            "topScore": round(candidates[0].base_score, 3),
            "edgeRetention": round(self.topology.edge_retention, 4),
            "activeEdges": self.topology.retained_edge_count,
            "generation": self.topology.generation,
        }
        if self.debug:
            print("COTE_DECISION " + json.dumps(self.last_decision, ensure_ascii=False), flush=True)
        return chosen

    def _accept_local_model_choice(self, act_index: int, candidates: Sequence[Candidate]) -> bool:
        selected = next((item for item in candidates if item.index == act_index), None)
        if selected is None:
            return False
        max_rank_raw = os.environ.get("COTE_LOCAL_MODEL_MAX_RANK_OVERRIDE")
        if max_rank_raw:
            try:
                max_rank = int(max_rank_raw)
            except ValueError:
                max_rank = 99
            rank = next((idx for idx, item in enumerate(candidates) if item.index == act_index), 99)
            if rank > max_rank:
                self.local_model.last_error = f"local model choice rank {rank} exceeds COTE_LOCAL_MODEL_MAX_RANK_OVERRIDE={max_rank}"
                return False
        try:
            max_drop = float(os.environ.get("COTE_LOCAL_MODEL_MAX_SCORE_DROP", "35.0"))
        except ValueError:
            max_drop = 35.0
        top_score = candidates[0].base_score if candidates else selected.base_score
        if top_score - selected.base_score > max_drop:
            self.local_model.last_error = (
                f"local model choice rejected by score gate: drop={top_score - selected.base_score:.3f}, "
                f"limit={max_drop:.3f}"
            )
            return False
        return True

    def _run_edge_communication(self, context: GameContext, snapshot: BeliefSnapshot) -> List[EdgeRecord]:
        if self.hp.disable_edge_messages:
            return []
        records: List[EdgeRecord] = []
        for edge in self.topology.active_edges():
            src, dst = edge
            source_belief = snapshot.node_vectors.get(src, snapshot.public_vector)
            target_belief = snapshot.node_vectors.get(dst, snapshot.public_vector)
            genome = self.population.best(edge)
            vector = genome.vector
            raw_message = None
            if (
                self.hp.edge_local_model
                and self.local_model.enabled
                and self.edge_model_used < self.hp.edge_local_model_budget
            ):
                raw_message = self.local_model.generate_edge_message(edge, vector, source_belief, context)
                self.edge_model_used += 1
                if raw_message:
                    self.edge_model_successes += 1
                else:
                    self.edge_model_failures += 1
            if not raw_message:
                raw_message = self._deterministic_edge_message(edge, vector, source_belief, context)
            parsed = self._parse_message_distribution(raw_message, vector, context)
            weight = self.topology.weight(edge)
            after = self._belief_update(target_belief, parsed, weight)
            records.append(edge_record_from_message(edge, raw_message, parsed, target_belief, after, weight))
        return records

    def _rank_candidates(
        self,
        context: GameContext,
        actions: Sequence[List[Any]],
        edge_records: Sequence[EdgeRecord],
    ) -> tuple[List[Candidate], List[float]]:
        semantic = self._semantic_support(edge_records)
        scored: List[Candidate] = []
        raw_scores: List[float] = []
        for idx, action in enumerate(actions):
            node_scores = self._node_scores(context, action, semantic)
            node_scores = self._apply_ablation_to_node_scores(node_scores)
            score = self.hp.node_score_scale * self.topology.aggregate_node_scores(node_scores, OUTPUT_NODE)
            score += self.hp.semantic_score_scale * self._semantic_action_bonus(context, action, semantic)
            if self.hp.reward_channel:
                score += self.hp.rule_score_scale * self._rule_action_value(context, action)
            score += self._guard_bonus(context, action)
            score += self.hp.expert_score_scale * self._expert_action_value(context, action)
            raw_scores.append(score)
            scored.append(Candidate(idx, action, score, node_scores, 0.0, self._reason(context, action)))
        probs = softmax(raw_scores, temperature=22.0)
        with_probs = [
            Candidate(item.index, item.action, item.base_score, item.node_scores, probs[item.index], item.reason)
            for item in scored
        ]
        with_probs.sort(key=lambda item: (item.base_score, self.rng.random() * 0.001), reverse=True)
        return with_probs, probs

    def _guard_bonus(self, context: GameContext, action: List[Any]) -> float:
        cards = action_cards(action)
        if cards and len(cards) >= context.hand_size:
            return self.hp.finish_guard_bonus
        if is_pass(action) and context.partner_winning and not context.leading:
            return self.hp.pass_guard_bonus
        if (not is_pass(action)) and context.opponent_winning and context.min_opponent_rest <= 5:
            return self.hp.block_guard_bonus
        return 0.0

    def _expert_action_value(self, context: GameContext, action: List[Any]) -> float:
        cards = action_cards(action)
        if is_pass(action):
            if context.leading:
                return -500.0
            if context.partner_winning:
                return 260.0 + max(0, 8 - context.partner_rest) * 20.0
            if context.opponent_winning:
                return -360.0 - max(0, 6 - context.min_opponent_rest) * 45.0
            return 15.0

        remaining = max(0, context.hand_size - len(cards))
        if remaining == 0:
            return 1200.0

        name = action_name(action)
        turns_before = estimated_turns(context.hand_cards, context.cur_rank)
        rest_cards = list(context.hand_cards)
        for card in cards:
            try:
                rest_cards.remove(card)
            except ValueError:
                pass
        turns_after = estimated_turns(rest_cards, context.cur_rank)
        turn_gain = turns_before - turns_after
        value = 0.0
        value += turn_gain * 230.0
        value += min(len(cards), 8) * 22.0
        value -= high_card_cost(cards, context.cur_rank) * (0.8 if remaining <= 5 else 1.6)
        value -= rank_value(action_rank(action), context.cur_rank) * (1.0 if context.leading else 2.0)

        if remaining <= 2:
            value += 180.0
        elif remaining <= 5:
            value += 90.0

        if context.partner_winning and not context.leading:
            value -= 280.0
        if context.opponent_winning:
            value += 120.0 + max(0, 6 - context.min_opponent_rest) * 55.0
            if context.min_opponent_rest <= len(cards):
                value += 160.0

        if is_bomb(action) and remaining > 0:
            if context.opponent_winning and context.min_opponent_rest <= 5:
                value += 120.0
            elif remaining <= 3:
                value -= 40.0
            else:
                value -= 520.0

        if context.leading and name == "Single" and context.hand_size > 7:
            value -= 70.0
        if turn_gain <= 0 and remaining > 4:
            value -= 140.0
        return value

    def _action_dropout(self, candidates: Sequence[Candidate]) -> Optional[int]:
        rate = max(0.0, min(1.0, self.hp.action_dropout_rate))
        if rate <= 0.0 or len(candidates) <= 1:
            return None
        if self.rng.random() >= rate:
            return None
        rank_cap = min(len(candidates) - 1, 3)
        rank = 1 + int(self.rng.random() * rank_cap)
        return candidates[rank].index

    def _apply_ablation_to_node_scores(self, node_scores: Dict[str, float]) -> Dict[str, float]:
        if self.hp.belief_channel:
            return node_scores
        adjusted = dict(node_scores)
        for node in (
            "T2_history_tracker",
            "T3_card_counter",
            "T4_opponent_intent",
            "T5_teammate_intent",
        ):
            adjusted[node] = 0.0
        return adjusted

    def _node_scores(self, context: GameContext, action: List[Any], semantic: Mapping[str, float]) -> Dict[str, float]:
        if context.stage == "tribute":
            return self._tribute_scores(context, action, give_high=True)
        if context.stage == "back":
            return self._tribute_scores(context, action, give_high=False)

        cards = action_cards(action)
        action_type = action_name(action)
        is_bomb_action = is_bomb(action)
        finish = bool(cards) and len(cards) >= context.hand_size
        rank_val = rank_value(action_rank(action), context.cur_rank)
        card_cost = high_card_cost(cards, context.cur_rank)
        residual_delta = residual_shape_score(context.hand_cards, cards, context.cur_rank)
        urgent = context.opponent_winning and context.min_opponent_rest <= max(5, len(cards) + 1)
        pass_action = is_pass(action)

        lead = 0.0
        follow = 0.0
        partner = 0.0
        pressure = 0.0
        bomb = 0.0
        shape = residual_delta
        risk = 0.0

        if pass_action:
            if context.leading:
                lead -= 100.0
            follow += 12.0
            if context.partner_winning:
                partner += 85.0 + 40.0 * semantic.get("help_partner", 0.0)
            if urgent:
                pressure -= 95.0 + 50.0 * semantic.get("block_opponent", 0.0)
            if context.hand_size <= 5:
                risk -= 18.0
            return {
                "T1_board_parser": lead + follow,
                "T2_history_tracker": risk,
                "T3_card_counter": shape + bomb,
                "T4_opponent_intent": pressure,
                "T5_teammate_intent": partner,
                "T6_macro_evaluator": lead + pressure + risk,
                "T7_hand_value": shape,
                "T8_action_decider": lead + follow + partner + pressure + risk,
            }

        length_bonus = len(cards) * (18.0 if context.hand_size <= 8 else 10.5)
        natural_bonus = ACTION_TYPE_BONUS.get(action_type, 0.0)
        cheapness = 25.0 - 1.25 * rank_val - 0.5 * card_cost

        if finish:
            finish_signal = 1.0 + semantic.get("finish", 0.0)
            lead += 320.0 * finish_signal
            follow += 320.0 * finish_signal
            pressure += 200.0
            risk += 130.0

        if context.leading:
            lead += length_bonus + natural_bonus - 0.85 * rank_val
            lead += 16.0 * semantic.get("shed_cards", 0.0) * min(len(cards), 6)
            if action_type == "Single" and context.hand_size > 6:
                lead -= 18.0
            if is_bomb_action and not finish:
                bomb -= 220.0 + 90.0 * semantic.get("preserve_bomb", 0.0)
                lead -= 120.0
            if len(cards) >= 5:
                lead += 12.0
        else:
            follow += cheapness + len(cards) * 3.0 + min(20.0, natural_bonus / 2.0)
            if context.partner_winning and not finish:
                partner -= 125.0 + 45.0 * semantic.get("help_partner", 0.0)
                follow -= 75.0
            if context.opponent_winning:
                pressure += 22.0 + 40.0 * semantic.get("block_opponent", 0.0)
                if urgent:
                    pressure += 95.0
                    follow += 30.0
                if context.min_opponent_rest <= len(cards):
                    pressure += 120.0
            if is_bomb_action and not finish:
                bomb -= 190.0 + 70.0 * semantic.get("preserve_bomb", 0.0)
                if urgent or context.hand_size <= 6:
                    bomb += 120.0 + 55.0 * semantic.get("block_opponent", 0.0)

        if context.partner_rest <= 3 and not finish and not context.leading:
            partner += 25.0 if context.partner_winning else -10.0

        if card_cost > 25 and context.hand_size > 8 and not finish:
            risk -= min(35.0, card_cost * 0.65)
        if action_type in ("Straight", "ThreePair", "TwoTrips", "ThreeWithTwo") and context.hand_size > len(cards):
            shape += 15.0
        if action_type == "Bomb" and len(cards) >= 5:
            bomb += 12.0 if urgent else -10.0

        return {
            "T1_board_parser": lead + follow,
            "T2_history_tracker": risk,
            "T3_card_counter": shape + bomb,
            "T4_opponent_intent": pressure,
            "T5_teammate_intent": partner,
            "T6_macro_evaluator": lead + pressure + risk,
            "T7_hand_value": shape + natural_bonus - card_cost,
            "T8_action_decider": lead + follow + partner + pressure + bomb + shape + risk,
        }

    def _tribute_scores(self, context: GameContext, action: List[Any], give_high: bool) -> Dict[str, float]:
        cards = action_cards(action)
        if not cards and isinstance(action, list):
            cards = [item for item in action if isinstance(item, str) and len(item) >= 2]
        value = max((card_value(card, context.cur_rank) for card in cards), default=0.0)
        score = value if give_high else -value
        return {
            "T1_board_parser": 0.0,
            "T2_history_tracker": 0.0,
            "T3_card_counter": score,
            "T4_opponent_intent": 0.0,
            "T5_teammate_intent": score,
            "T6_macro_evaluator": score * 10.0,
            "T7_hand_value": score * 4.0,
            "T8_action_decider": score * 10.0,
        }

    def _reason(self, context: GameContext, action: List[Any]) -> str:
        if is_pass(action):
            return "pass; partner control" if context.partner_winning else "pass"
        cards = action_cards(action)
        if len(cards) >= context.hand_size:
            return "finish hand"
        if is_bomb(action):
            return "bomb pressure" if context.min_opponent_rest <= 5 else "bomb reserved unless necessary"
        if context.leading:
            return f"lead {action_name(action)} shedding {len(cards)}"
        return f"cheap follow with {action_name(action)}"

    def _semantic_support(self, edge_records: Sequence[EdgeRecord]) -> Dict[str, float]:
        from .prompts import SEMANTIC_AXES

        support = {axis: 0.0 for axis in SEMANTIC_AXES}
        total = 0.0
        for record in edge_records:
            if record.edge[1] != OUTPUT_NODE:
                continue
            for idx, axis in enumerate(SEMANTIC_AXES):
                if idx < len(record.parsed_distribution):
                    support[axis] += record.weight * record.parsed_distribution[idx]
            total += record.weight
        if total > 1e-9:
            for axis in support:
                support[axis] /= total
        return support

    def _semantic_action_bonus(self, context: GameContext, action: List[Any], semantic: Mapping[str, float]) -> float:
        cards = action_cards(action)
        if is_pass(action):
            return 30.0 * semantic.get("help_partner", 0.0) if context.partner_winning else -20.0 * semantic.get("block_opponent", 0.0)
        bonus = 0.0
        if cards and len(cards) >= context.hand_size:
            bonus += 80.0 * semantic.get("finish", 0.0)
        if context.opponent_winning:
            bonus += 45.0 * semantic.get("block_opponent", 0.0)
        if is_bomb(action) and context.hand_size > len(cards) and not (context.opponent_winning and context.min_opponent_rest <= 5):
            bonus -= 55.0 * semantic.get("preserve_bomb", 0.0)
        bonus += min(len(cards), 6) * 5.0 * semantic.get("shed_cards", 0.0)
        return bonus

    def _rule_action_value(self, context: GameContext, action: List[Any]) -> float:
        """Domain value used by T6/T7 before T8 aggregation.

        The paper's T6/T7 are still Guandan-specific evaluators; this term keeps
        the learned communication scaffold from overpowering basic card-play
        rules when the prompt population is still near random initialization.
        """

        cards = action_cards(action)
        if is_pass(action):
            if context.partner_winning:
                return 115.0
            if context.leading:
                return -240.0
            if context.opponent_winning and context.min_opponent_rest <= 6:
                return -180.0
            return 8.0

        remaining = max(0, context.hand_size - len(cards))
        rank_val = rank_value(action_rank(action), context.cur_rank)
        card_cost = high_card_cost(cards, context.cur_rank)
        name = action_name(action)
        bomb = is_bomb(action)
        turns_before = estimated_turns(context.hand_cards, context.cur_rank)
        rest_cards = list(context.hand_cards)
        for card in cards:
            try:
                rest_cards.remove(card)
            except ValueError:
                pass
        turns_after = estimated_turns(rest_cards, context.cur_rank)
        turn_gain = turns_before - turns_after

        if remaining == 0:
            return 900.0

        value = 0.0
        value += turn_gain * 135.0
        value += min(len(cards), 8) * (12.0 if context.leading else 8.0)
        value += ACTION_TYPE_BONUS.get(name, 0.0) * 0.8
        value -= rank_val * (1.6 if context.leading else 2.8)
        value -= card_cost * (0.45 if remaining <= 6 else 1.15)

        if remaining <= 2:
            value += 150.0
        elif remaining <= 5:
            value += 70.0

        if context.partner_winning and not context.leading:
            value -= 220.0
        if context.opponent_winning:
            value += 70.0
            if context.min_opponent_rest <= 5:
                value += 140.0
                value += min(len(cards), 6) * 18.0

        if bomb and remaining > 0:
            if context.opponent_winning and context.min_opponent_rest <= 5:
                value += 40.0
            elif remaining <= 3:
                value -= 30.0
            else:
                value -= 460.0

        if context.leading and name == "Single" and context.hand_size > 8:
            value -= 45.0
        if not context.leading and len(cards) >= 5 and not bomb:
            value += 20.0
        if turn_gain <= 0 and remaining > 4:
            value -= 90.0
        return value

    def _deterministic_edge_message(
        self,
        edge: Edge,
        vector: Sequence[float],
        source_belief: Sequence[float],
        context: GameContext,
    ) -> str:
        prompt = decode_soft_prompt(vector, edge)
        dist = self._contextual_prompt_distribution(vector, context)
        return json.dumps(
            {
                "prompt": prompt,
                "edge": f"{edge[0]}->{edge[1]}",
                "finish": round(dist[0], 3),
                "block_opponent": round(dist[1], 3),
                "help_partner": round(dist[2], 3),
                "preserve_bomb": round(dist[3], 3),
                "shed_cards": round(dist[4], 3),
                "low_ambiguity": round(dist[5], 3),
                "belief_head": [round(value, 3) for value in source_belief[:5]],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _parse_message_distribution(self, raw_message: str, vector: Sequence[float], context: GameContext) -> List[float]:
        dist = self._contextual_prompt_distribution(vector, context)
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return dist
        keys = ["finish", "block_opponent", "help_partner", "preserve_bomb", "shed_cards", "low_ambiguity"]
        values = []
        for idx, key in enumerate(keys):
            try:
                values.append(float(payload.get(key, dist[idx])))
            except (TypeError, ValueError):
                values.append(dist[idx])
        total = sum(max(0.0, value) for value in values)
        if total <= 1e-9:
            return dist
        return [max(0.0, value) / total for value in values]

    def _contextual_prompt_distribution(self, vector: Sequence[float], context: GameContext) -> List[float]:
        dist = prompt_distribution(vector)
        values = list(dist)
        if context.hand_size <= 5:
            values[0] += 0.40
        if self.hp.belief_channel:
            if context.opponent_winning or context.min_opponent_rest <= 5:
                values[1] += 0.45
            if context.partner_winning or context.partner_rest <= 3:
                values[2] += 0.35
        if context.leading:
            values[4] += 0.25
        if context.hand_size > 10:
            values[3] += 0.15
        total = sum(values)
        return [value / total for value in values]

    def _belief_update(self, base: Sequence[float], parsed: Sequence[float], weight: float) -> List[float]:
        size = min(len(base), len(parsed))
        if size <= 0:
            return list(base)
        alpha = max(0.0, min(0.85, 0.10 + weight))
        base_head = list(base[:size])
        parsed_head = list(parsed[:size])
        total_base = sum(base_head) or 1.0
        total_parsed = sum(parsed_head) or 1.0
        return [
            (1.0 - alpha) * (base_head[idx] / total_base) + alpha * (parsed_head[idx] / total_parsed)
            for idx in range(size)
        ]

    def _choose_from_distribution(self, candidates: Sequence[Candidate], distribution: Sequence[float]) -> int:
        if not candidates:
            return 0
        if not self.hp.sample_action:
            return candidates[0].index
        roll = self.rng.random()
        cumulative = 0.0
        for idx, prob in enumerate(distribution):
            cumulative += prob
            if roll <= cumulative:
                return idx
        return candidates[0].index

    def _should_ask_local_model(
        self,
        context: GameContext,
        actions: Sequence[List[Any]],
        candidates: Sequence[Candidate],
    ) -> bool:
        if (
            not self.hp.use_local_model
            or not self.local_model.enabled
            or self.local_model_used >= self.hp.local_model_budget
        ):
            return False
        if len(actions) < self.hp.local_model_min_actions:
            return False
        high_leverage = (
            context.leading
            or context.hand_size <= 8
            or context.min_opponent_rest <= 6
            or any(is_bomb(action) for action in actions)
        )
        if not high_leverage:
            return False
        if len(candidates) >= 2 and candidates[0].base_score - candidates[1].base_score > 35.0:
            return False
        return True

    def _offline_update(self, episode: EpisodeRecord, snapshot: BeliefSnapshot) -> None:
        prompt_vectors = self.population.active_prompt_vectors()
        breakdown = self.evaluator.evaluate_episode(episode, self.topology, prompt_vectors, snapshot)
        if self.hp.evolve:
            mode = self.hp.opt_mode
            prompt_step = self.hp.prompt_evolve
            topology_step = self.hp.topology_update
            alternating_step = False
            if mode == "prompt_only":
                prompt_step = True
                topology_step = False
            elif mode == "topo_only":
                prompt_step = False
                topology_step = True
            elif mode == "alternating":
                alternating_step = True
                prompt_step = self.topology.generation % 2 == 0
                topology_step = not prompt_step
            reachable = {node: snapshot.node_vectors.get(node, snapshot.public_vector) for node in NODE_KEYS}
            if prompt_step:
                self.evolver.evolve(self.population, self.topology, self.evaluator, breakdown, reachable)
                self.soft_prompt_trainer.train_population(self.population, self.topology, breakdown)
                if alternating_step and not topology_step:
                    self.topology.generation += 1
            if topology_step:
                gradients = self.evaluator.topology_gradients(self.topology, breakdown)
                self.topology.apply_gradients(gradients, self.hp.eta_w, self.hp.theta_grad)
                if self.hp.topology_prune:
                    self.topology.prune(self.hp.edge_threshold)
        self._save_state(breakdown)
        stats_payload = {
            "client_id": self.client_id,
            "my_pos": self.my_pos,
            "seed": self.seed,
            "generation": self.topology.generation,
            "fitness_total": round(breakdown.total, 6),
            "topo_fitness": round(breakdown.topo_fitness, 6),
            "prompt_fitness_sum": round(breakdown.prompt_fitness_sum, 6),
            "action_reward": round(breakdown.action_reward, 6),
            "message_reward": round(breakdown.message_reward, 6),
            "decision_reward": round(breakdown.decision_reward, 6),
            "gt_calibration_loss": round(breakdown.gt_calibration_loss, 6),
            "ablation": {
                "reward_channel": self.hp.reward_channel,
                "error_channel": self.hp.error_channel,
                "belief_channel": self.hp.belief_channel,
                "disable_edge_messages": self.hp.disable_edge_messages,
                "prompt_evolve": self.hp.prompt_evolve,
                "topology_update": self.hp.topology_update,
                "topology_prune": self.hp.topology_prune,
                "opt_mode": self.hp.opt_mode,
                "node_score_scale": self.hp.node_score_scale,
                "semantic_score_scale": self.hp.semantic_score_scale,
                "rule_score_scale": self.hp.rule_score_scale,
                "finish_guard_bonus": self.hp.finish_guard_bonus,
                "pass_guard_bonus": self.hp.pass_guard_bonus,
                "block_guard_bonus": self.hp.block_guard_bonus,
                "expert_score_scale": self.hp.expert_score_scale,
                "action_dropout_rate": self.hp.action_dropout_rate,
            },
            "local_model_usage": self.local_model.stats(),
            "local_model_last_error": (self.local_model.last_error or "")[:300],
            "local_model_decision": {
                "attempts": self.local_model_used,
                "successes": self.local_model_decision_successes,
                "failures": self.local_model_decision_failures,
            },
            "edge_local_model": {
                "attempts": self.edge_model_used,
                "successes": self.edge_model_successes,
                "failures": self.edge_model_failures,
            },
            "soft_prompt_training": {
                "updates": self.soft_prompt_trainer.last_update_count,
                "losses": self.soft_prompt_trainer.last_losses,
                "last_error": self.soft_prompt_trainer.last_error,
            },
            **self.topology.runtime_summary(),
        }
        print("COTE_STATS " + json.dumps(stats_payload, ensure_ascii=False), flush=True)
        if self.debug:
            print(
                "COTE_FITNESS "
                + json.dumps(
                    {
                        "total": round(breakdown.total, 4),
                        "topo": round(breakdown.topo_fitness, 4),
                        "decision": round(breakdown.decision_reward, 4),
                        "gtCalib": round(breakdown.gt_calibration_loss, 4),
                        **self.topology.runtime_summary(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    def _load_or_init_state(self) -> tuple[CoteTopologyState, PromptPopulation]:
        topology = load_topology(str(self.state_path)) if self.state_path else None
        population: Optional[PromptPopulation] = None
        if self.state_path and self.state_path.exists():
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(payload.get("prompts"), dict):
                    population = PromptPopulation.from_json(payload["prompts"], self.hp)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                population = None
        if topology is None:
            topology = CoteTopologyState.from_env()
        if population is None:
            population = PromptPopulation.bootstrap(topology.all_edges(), self.rng, self.hp)
        return topology, population

    def _save_state(self, breakdown: Optional[Any] = None) -> None:
        if not self.state_path:
            return
        payload: Dict[str, Any] = {
            "topology": self.topology.to_json(),
            "prompts": self.population.to_json(),
            "metrics": metrics_reference(),
            "last_fitness": {
                "total": getattr(breakdown, "total", None),
                "topo_fitness": getattr(breakdown, "topo_fitness", None),
                "decision_reward": getattr(breakdown, "decision_reward", None),
                "gt_calibration_loss": getattr(breakdown, "gt_calibration_loss", None),
            }
            if breakdown is not None
            else None,
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _state_path(self, client_id: Optional[int]) -> Optional[Path]:
        if os.environ.get("COTE_DISABLE_STATE", "0") == "1":
            return None
        raw = os.environ.get("COTE_STATE_PATH")
        if raw:
            formatted = raw.format(client_id=client_id or 0)
            return Path(formatted)
        clients_dir = Path(__file__).resolve().parents[1]
        return clients_dir / f".cote_state_client{client_id or 0}.json"

    @staticmethod
    def _resolve_seed(seed: Optional[int], client_id: Optional[int]) -> int:
        if seed is not None:
            return int(seed)
        raw = os.environ.get("COTE_SEED")
        if raw:
            try:
                return int(raw) + 1009 * int(client_id or 0)
            except ValueError:
                pass
        return os.getpid() + int(time.time())


def metrics_reference() -> Dict[str, Any]:
    return {
        "target_win_rate": TARGET_WIN_RATE,
        "node_count": NODE_COUNT,
        "node_design": NODE_KEYS,
        "algorithm": "COTE + EL-BDPEA with local soft-prompt training",
        "model_backend": "local Transformers causal LM",
        "default_model_path": LOCAL_MODEL_PATH,
        "training_notes": [
            "Edge prompts are trained as continuous vectors with PyTorch autograd.",
            "When LOCAL_MODEL_PATH is set and COTE_SOFT_PROMPT_LM_LOSS=1, the soft prompt is injected into local model embeddings for language-model loss.",
        ],
        "ablation_env": {
            "reward_channel": "COTE_REWARD_CHANNEL",
            "error_channel": "COTE_ERROR_CHANNEL",
            "belief_channel": "COTE_BELIEF_CHANNEL",
        },
    }
