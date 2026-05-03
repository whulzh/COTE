# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import re
from typing import Iterable, List, Mapping, Sequence


SEMANTIC_AXES = (
    "finish",
    "block_opponent",
    "help_partner",
    "preserve_bomb",
    "shed_cards",
    "low_ambiguity",
)

_KEYWORDS = {
    "finish": ("finish", "win", "go out", "empty hand", "complete", "走完", "冲刺"),
    "block_opponent": ("block", "opponent", "deny", "stop", "pressure", "压制", "拦截"),
    "help_partner": ("partner", "teammate", "assist", "cover", "help", "配合", "队友"),
    "preserve_bomb": ("bomb", "preserve", "save", "reserve", "炸", "保留"),
    "shed_cards": ("shed", "reduce", "shape", "lead", "cards", "出牌", "牌型"),
    "low_ambiguity": ("clear", "unambiguous", "certain", "low ambiguity", "明确", "低歧义"),
}


def normalize_distribution(values: Iterable[float], size: int | None = None) -> List[float]:
    raw = [max(0.0, float(value)) for value in values]
    if size is not None:
        raw = raw[:size]
        raw.extend([0.0] * max(0, size - len(raw)))
    total = sum(raw)
    if total <= 1e-12:
        n = size or len(raw) or len(SEMANTIC_AXES)
        return [1.0 / n for _ in range(n)]
    return [value / total for value in raw]


def parse_message_distribution(message: str | Mapping[str, object] | Sequence[float]) -> List[float]:
    """Strict `q_phi` semantic parser.

    JSON microcode is treated as the primary interface. Free text falls back to
    deterministic keyword evidence so malformed local-model output still yields
    a calibrated distribution instead of silently using the prompt prior.
    """

    if isinstance(message, Mapping):
        return _distribution_from_mapping(message)
    if isinstance(message, (list, tuple)):
        return normalize_distribution([float(value) for value in message], len(SEMANTIC_AXES))

    text = str(message or "")
    payload = _extract_json_payload(text)
    if isinstance(payload, Mapping):
        return _distribution_from_mapping(payload)
    return _keyword_distribution(text)


def _extract_json_payload(text: str) -> Mapping[str, object] | None:
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, Mapping) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            payload = json.loads(text[start : end + 1])
            return payload if isinstance(payload, Mapping) else None
        except json.JSONDecodeError:
            return None
    return None


def _distribution_from_mapping(payload: Mapping[str, object]) -> List[float]:
    values: List[float] = []
    for axis in SEMANTIC_AXES:
        try:
            values.append(float(payload.get(axis, 0.0)))
        except (TypeError, ValueError):
            values.append(0.0)
    return normalize_distribution(values, len(SEMANTIC_AXES))


def _keyword_distribution(text: str) -> List[float]:
    lowered = text.lower()
    scores = [0.05 for _ in SEMANTIC_AXES]
    for idx, axis in enumerate(SEMANTIC_AXES):
        for keyword in _KEYWORDS[axis]:
            pattern = re.escape(keyword.lower())
            if re.search(pattern, lowered):
                scores[idx] += 1.0
    return normalize_distribution(scores, len(SEMANTIC_AXES))


def kl_divergence(target: Sequence[float], predicted: Sequence[float]) -> float:
    size = max(len(target), len(predicted))
    p = normalize_distribution(target, size)
    q = normalize_distribution(predicted, size)
    return sum(pi * math.log(max(pi, 1e-12) / max(qi, 1e-12)) for pi, qi in zip(p, q))


def entropy(distribution: Sequence[float]) -> float:
    dist = normalize_distribution(distribution)
    return -sum(value * math.log(max(value, 1e-12)) for value in dist)


def clarity_score(distribution: Sequence[float]) -> float:
    dist = normalize_distribution(distribution)
    if not dist:
        return 0.0
    max_entropy = math.log(len(dist))
    return 1.0 - entropy(dist) / max_entropy if max_entropy > 0 else 1.0
