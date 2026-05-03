# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Sequence

from .config import LOCAL_MODEL_DEVICE_MAP, LOCAL_MODEL_DTYPE, LOCAL_MODEL_MAX_NEW_TOKENS, LOCAL_MODEL_PATH
from .game import Candidate, GameContext, parse_json_object
from .prompts import decode_soft_prompt
from .topology import Edge


class LocalCausalLM:
    """Lazy Transformers backend for local COTE inference and prompt training."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        dtype: Optional[str] = None,
        device_map: Optional[str] = None,
    ) -> None:
        self.model_path = model_path if model_path is not None else os.environ.get("LOCAL_MODEL_PATH", LOCAL_MODEL_PATH)
        self.dtype = dtype or os.environ.get("LOCAL_MODEL_DTYPE", LOCAL_MODEL_DTYPE)
        self.device_map = device_map or os.environ.get("LOCAL_MODEL_DEVICE_MAP", LOCAL_MODEL_DEVICE_MAP)
        self.tokenizer: Any = None
        self.model: Any = None
        self.torch: Any = None
        self.last_error: Optional[str] = None
        self.successful_calls = 0
        self.failed_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def enabled(self) -> bool:
        return bool(self.model_path)

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    @property
    def hidden_size(self) -> int:
        if not self.loaded:
            return 0
        config = getattr(self.model, "config", None)
        return int(getattr(config, "hidden_size", 0) or getattr(config, "n_embd", 0) or 0)

    def ensure_loaded(self) -> bool:
        if self.loaded:
            return True
        if not self.enabled:
            self.last_error = "LOCAL_MODEL_PATH is not set"
            return False
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # noqa: BLE001 - optional local runtime dependency.
            self.last_error = f"local model dependencies are unavailable: {exc}"
            return False
        try:
            torch_dtype = self._torch_dtype(torch)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                device_map=self.device_map,
                trust_remote_code=True,
                local_files_only=True,
            )
            self.model.eval()
            self.torch = torch
            return True
        except Exception as exc:  # noqa: BLE001 - keep card-play fallback available.
            self.last_error = str(exc)
            self.tokenizer = None
            self.model = None
            return False

    def choose_action(
        self,
        context: GameContext,
        candidates: Sequence[Candidate],
        history_tail: Sequence[Dict[str, Any]],
        edge_prompts: Dict[Edge, Sequence[float]],
        topology_summary: Dict[str, Any],
        node_thoughts: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        if not self.enabled or not candidates:
            return None
        system = (
            "You are node T8_action_decider in a COTE Guandan decision network. "
            "Candidates are already sorted by topology and prompt score. Return JSON only: "
            "{\"actIndex\":number,\"confidence\":number,\"reason\":\"short\"}."
        )
        payload = self._decision_payload(context, candidates, history_tail, edge_prompts, topology_summary, node_thoughts)
        parsed = self.generate_json(system, payload, max_new_tokens=int(os.environ.get("COTE_LOCAL_DECISION_MAX_TOKENS", "256")))
        if parsed is None:
            return None
        try:
            idx = int(parsed.get("actIndex"))
        except (TypeError, ValueError):
            self.last_error = f"invalid actIndex in {parsed}"
            return None
        allowed = {candidate.index for candidate in candidates}
        if idx not in allowed:
            if 0 <= idx < len(candidates):
                return candidates[idx].index
            self.last_error = f"local model returned non-candidate actIndex={idx}"
            return None
        return idx

    def think_node(
        self,
        node_key: str,
        context: GameContext,
        candidates: Sequence[Candidate],
        history_tail: Sequence[Dict[str, Any]],
        edge_messages: Sequence[Any],
        topology_summary: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled or not candidates:
            return None
        system = (
            f"You are node {node_key} in a COTE Guandan multi-agent decision network. "
            "Think only from this node's role. Return JSON only: "
            "{\"node\":string,\"summary\":\"short\",\"candidateScores\":{\"actIndex\":number},\"confidence\":number}. "
            "Scores should be in [-1,1], where higher means this node prefers the candidate."
        )
        payload = json.dumps(
            {
                "node": node_key,
                "roleHint": self._node_role_hint(node_key),
                "state": self._state_payload(context),
                "recentHistory": list(history_tail)[-10:],
                "edgeMessages": list(edge_messages)[-12:],
                "topology": topology_summary,
                "candidates": [
                    {
                        "actIndex": item.index,
                        "candidateRank": rank,
                        "action": item.action,
                        "score": round(item.base_score, 3),
                        "probability": round(item.probability, 4),
                        "reason": item.reason,
                    }
                    for rank, item in enumerate(candidates)
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        parsed = self.generate_node_thought_json(
            node_key,
            system,
            payload,
            max_new_tokens=int(os.environ.get("COTE_NODE_LOCAL_MAX_TOKENS", "256")),
            temperature=0.1,
        )
        if parsed is None:
            return None
        parsed["node"] = str(parsed.get("node") or node_key)
        return parsed

    def generate_node_thought_json(
        self,
        node_key: str,
        system: str,
        payload: str,
        max_new_tokens: int,
        temperature: float = 0.1,
    ) -> Optional[Dict[str, Any]]:
        text = self.generate_text(system, payload, max_new_tokens=max_new_tokens, temperature=temperature)
        if text is None:
            return None
        try:
            parsed = parse_json_object(text)
            self.successful_calls += 1
            return parsed
        except Exception as exc:  # noqa: BLE001 - malformed node JSON can often be repaired.
            repaired = repair_node_thought_json(text, node_key)
            if repaired is not None:
                self.successful_calls += 1
                self.last_error = ""
                return repaired
            self.failed_calls += 1
            self.last_error = f"{exc}; content={text[:300]!r}"
            return None

    def generate_edge_message(
        self,
        edge: Edge,
        vector: Sequence[float],
        source_belief: Sequence[float],
        context: GameContext,
    ) -> Optional[str]:
        if not self.enabled:
            return None
        prompt = decode_soft_prompt(vector, edge, compact=False)
        system = (
            "You are an internal COTE communication node. Return JSON only. "
            "The JSON must contain exactly these numeric keys: "
            "finish, block_opponent, help_partner, preserve_bomb, shed_cards, low_ambiguity. "
            "Each value must be between 0 and 1."
        )
        payload = json.dumps(
            {
                "edge": f"{edge[0]}->{edge[1]}",
                "softPromptProjection": prompt,
                "sourceBelief": [round(value, 4) for value in source_belief],
                "state": self._state_payload(context),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        parsed = self.generate_json(
            system,
            payload,
            max_new_tokens=int(os.environ.get("COTE_EDGE_LOCAL_MAX_TOKENS", "192")),
            temperature=0.1,
            soft_prompt_vector=vector,
            soft_prompt_text=prompt,
        )
        if parsed is None:
            return None
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))

    def generate_json(
        self,
        system: str,
        payload: str,
        max_new_tokens: int = LOCAL_MODEL_MAX_NEW_TOKENS,
        temperature: float = 0.1,
        soft_prompt_vector: Optional[Sequence[float]] = None,
        soft_prompt_text: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        text = self.generate_text(
            system,
            payload,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            soft_prompt_vector=soft_prompt_vector,
            soft_prompt_text=soft_prompt_text,
        )
        if text is None:
            return None
        try:
            parsed = parse_json_object(text)
            self.successful_calls += 1
            return parsed
        except Exception as exc:  # noqa: BLE001 - preserve fallback behavior.
            self.failed_calls += 1
            self.last_error = f"{exc}; content={text[:300]!r}"
            return None

    def generate_text(
        self,
        system: str,
        payload: str,
        max_new_tokens: int = LOCAL_MODEL_MAX_NEW_TOKENS,
        temperature: float = 0.1,
        soft_prompt_vector: Optional[Sequence[float]] = None,
        soft_prompt_text: Optional[str] = None,
    ) -> Optional[str]:
        if not self.ensure_loaded():
            self.failed_calls += 1
            return None
        prompt = self._format_chat(system, payload)
        try:
            encoded = self.tokenizer(prompt, return_tensors="pt")
            encoded = {key: value.to(self._device()) for key, value in encoded.items()}
            input_len = int(encoded["input_ids"].shape[-1])
            soft_len = 0
            if soft_prompt_vector is None:
                output = self.model.generate(
                    **encoded,
                    do_sample=temperature > 0.0,
                    temperature=max(temperature, 1e-5),
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                generated_ids = output[0][input_len:]
                self.prompt_tokens += input_len
                self.completion_tokens += max(0, int(generated_ids.shape[-1]))
                return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            else:
                embeds = self.model.get_input_embeddings()(encoded["input_ids"])
                soft = self.soft_prompt_embeddings(
                    soft_prompt_vector,
                    embeds.dtype,
                    embeds.device,
                    anchor_text=soft_prompt_text,
                )
                soft_len = soft.shape[0]
                inputs_embeds = self.torch.cat([soft.unsqueeze(0), embeds], dim=1)
                attention_mask = self.torch.cat(
                    [
                        self.torch.ones((1, soft_len), dtype=encoded["attention_mask"].dtype, device=embeds.device),
                        encoded["attention_mask"],
                    ],
                    dim=1,
                )
                output = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=temperature > 0.0,
                    temperature=max(temperature, 1e-5),
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=int(os.environ.get("COTE_EDGE_LOCAL_MIN_TOKENS", "8")),
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                generated_ids = output[0]
                self.prompt_tokens += input_len + soft_len
                self.completion_tokens += max(0, int(generated_ids.shape[-1]))
                return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        except Exception as exc:  # noqa: BLE001 - local model failure should fall back.
            self.failed_calls += 1
            self.last_error = str(exc)
            return None

    def nll_with_soft_prompt(self, vector_tensor: Any, system: str, payload: str, target_text: str) -> Any:
        if not self.ensure_loaded():
            return None
        torch = self.torch
        prompt = self._format_chat(system, payload)
        full_text = prompt + target_text
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self._device())
        full = self.tokenizer(full_text, return_tensors="pt")
        input_ids = full["input_ids"].to(self._device())
        attention_mask = full["attention_mask"].to(self._device())
        labels = input_ids.clone()
        labels[:, : prompt_ids.shape[-1]] = -100
        base_embeds = self.model.get_input_embeddings()(input_ids)
        soft = self.soft_prompt_embeddings(
            vector_tensor,
            base_embeds.dtype,
            base_embeds.device,
            anchor_text=system,
        )
        inputs_embeds = torch.cat([soft.unsqueeze(0), base_embeds], dim=1)
        soft_mask = torch.ones((1, soft.shape[0]), dtype=attention_mask.dtype, device=attention_mask.device)
        soft_labels = torch.full((1, soft.shape[0]), -100, dtype=labels.dtype, device=labels.device)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.cat([soft_mask, attention_mask], dim=1),
            labels=torch.cat([soft_labels, labels], dim=1),
        )
        return outputs.loss

    def soft_prompt_embeddings(
        self,
        vector: Any,
        dtype: Any,
        device: Any,
        anchor_text: Optional[str] = None,
    ) -> Any:
        if self.torch is None:
            import torch
        else:
            torch = self.torch
        if not hasattr(vector, "to"):
            tensor = torch.tensor(list(vector), dtype=dtype, device=device)
        else:
            tensor = vector.to(device=device, dtype=dtype)
        hidden = max(1, self.hidden_size or int(tensor.numel()))
        tokens = max(1, int(os.environ.get("COTE_SOFT_PROMPT_TOKENS", "4")))

        if anchor_text and self.tokenizer is not None and self.model is not None:
            ids = self.tokenizer(
                anchor_text,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(device)
            with torch.no_grad():
                anchor = self.model.get_input_embeddings()(ids)[0]
            if anchor.shape[0] >= tokens:
                base = anchor[:tokens]
            else:
                repeat = anchor.mean(dim=0, keepdim=True).repeat(tokens - anchor.shape[0], 1)
                base = torch.cat([anchor, repeat], dim=0)
        else:
            embed_weight = self.model.get_input_embeddings().weight
            base = embed_weight.mean(dim=0, keepdim=True).repeat(tokens, 1)

        bias = tensor.repeat((hidden + tensor.numel() - 1) // max(1, tensor.numel()))[:hidden]
        bias = torch.tanh(bias).unsqueeze(0).repeat(tokens, 1)
        scale = float(os.environ.get("COTE_SOFT_PROMPT_BIAS_SCALE", "0.05"))
        return base.to(dtype=dtype, device=device) + scale * bias

    def stats(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
        }

    def _decision_payload(
        self,
        context: GameContext,
        candidates: Sequence[Candidate],
        history_tail: Sequence[Dict[str, Any]],
        edge_prompts: Dict[Edge, Sequence[float]],
        topology_summary: Dict[str, Any],
        node_thoughts: Optional[Dict[str, Any]] = None,
    ) -> str:
        inbound_prompts = {
            f"{edge[0]}->{edge[1]}": decode_soft_prompt(vector, edge)
            for edge, vector in edge_prompts.items()
            if edge[1] == "T8_action_decider"
        }
        return json.dumps(
            {
                "state": self._state_payload(context),
                "recentHistory": list(history_tail)[-10:],
                "topology": topology_summary,
                "inboundT8SoftPrompts": inbound_prompts,
                "nodeThoughts": node_thoughts or {},
                "candidates": [
                    {
                        "actIndex": item.index,
                        "candidateRank": rank,
                        "action": item.action,
                        "score": round(item.base_score, 3),
                        "probability": round(item.probability, 4),
                        "nodes": {key: round(value, 2) for key, value in item.node_scores.items()},
                        "reason": item.reason,
                    }
                    for rank, item in enumerate(candidates)
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _node_role_hint(node_key: str) -> str:
        hints = {
            "T1_board_parser": "Parse current trick, legal move context, leading/following constraints, and board pressure.",
            "T2_history_tracker": "Use recent action history to detect tempo, pass patterns, and control transfer.",
            "T3_card_counter": "Estimate remaining ranks, bombs, hidden-card risk, and card depletion.",
            "T4_opponent_intent": "Infer opponent threat, near-finish risk, and blocking urgency.",
            "T5_teammate_intent": "Infer partner control, partner finishing chances, and cooperative restraint.",
            "T6_macro_evaluator": "Judge long-horizon strategy, level advancement, tempo, and risk tradeoffs.",
            "T7_hand_value": "Evaluate hand shape, shedding value, bomb preservation, and residual hand quality.",
            "T8_action_decider": "Integrate all node outputs and choose the final legal action.",
        }
        return hints.get(node_key, "Evaluate candidates from this COTE node's role.")

    def _state_payload(self, context: GameContext) -> Dict[str, Any]:
        return {
            "seat": context.my_pos,
            "partner": context.partner_pos,
            "opponents": list(context.opponents),
            "stage": context.stage,
            "handSize": context.hand_size,
            "handCards": context.hand_cards,
            "selfRank": context.self_rank,
            "oppoRank": context.oppo_rank,
            "curRank": context.cur_rank,
            "greaterPos": context.greater_pos,
            "greaterAction": context.greater_action,
            "leading": context.leading,
            "partnerWinning": context.partner_winning,
            "opponentWinning": context.opponent_winning,
            "minOpponentRest": context.min_opponent_rest,
        }

    def _format_chat(self, system: str, payload: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": payload}]
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return f"System:\n{system}\n\nUser:\n{payload}\n\nAssistant:\n"

    def _device(self) -> Any:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return "cpu"

    def _torch_dtype(self, torch: Any) -> Any:
        key = str(self.dtype or "auto").lower()
        if key in {"auto", ""}:
            return "auto"
        if key in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if key in {"fp16", "float16", "half"}:
            return torch.float16
        if key in {"fp32", "float32"}:
            return torch.float32
        return "auto"


def repair_node_thought_json(text: str, node_key: str) -> Optional[Dict[str, Any]]:
    """Repair common malformed node-thinking JSON produced by small local LMs.

    The node prompt asks for a `candidateScores` mapping, but some models emit
    pseudo-JSON fragments such as `"actIndex":0:0.85` or `"actIndex":75,0.85`.
    Those fragments carry the information we need, so strict node thinking can
    accept them after deterministic repair instead of discarding the whole call.
    """

    scores: Dict[str, float] = {}
    for match in re.finditer(
        r'"?actIndex"?\s*[:=]\s*"?(-?\d+)"?\s*[:,]\s*(-?\d+(?:\.\d+)?)',
        text,
    ):
        scores[str(int(match.group(1)))] = max(-1.0, min(1.0, float(match.group(2))))
    if not scores:
        return None
    return {
        "node": _extract_string_field(text, "node") or node_key,
        "summary": _extract_string_field(text, "summary") or "repaired node thought",
        "candidateScores": scores,
        "confidence": _extract_float_field(text, "confidence", 0.7),
        "repaired": True,
    }


def _extract_string_field(text: str, key: str) -> Optional[str]:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', text)
    return match.group(1) if match else None


def _extract_float_field(text: str, key: str, default: float) -> float:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    if not match:
        return default
    try:
        return max(0.0, min(1.0, float(match.group(1))))
    except ValueError:
        return default
