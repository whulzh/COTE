# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional

from .config import HyperParams
from .fitness import EdgeMetrics, FitnessBreakdown
from .game import normalize_distribution
from .prompts import PromptGenome, PromptPopulation, clip_prompt, edge_target_distribution, prompt_distribution
from .topology import CoteTopologyState, Edge


class SoftPromptTrainer:
    """Autograd updater for edge-level soft prompt vectors."""

    def __init__(self, hp: Optional[HyperParams] = None, local_model: Any = None) -> None:
        self.hp = hp or HyperParams()
        self.local_model = local_model
        self.last_update_count = 0
        self.last_error: Optional[str] = None
        self.last_losses: Dict[str, float] = {}

    def train_population(
        self,
        population: PromptPopulation,
        topology: CoteTopologyState,
        breakdown: FitnessBreakdown,
    ) -> int:
        if not self.hp.soft_prompt_train:
            self.last_update_count = 0
            return 0
        torch = self._torch()
        if torch is None:
            self.last_update_count = 0
            return 0

        updates = 0
        self.last_losses = {}
        for edge in topology.active_edges():
            items = population.populations.get(edge)
            if not items:
                continue
            best = max(items, key=lambda item: item.fitness.mean).clone()
            metrics = breakdown.edge_metrics.get(edge, EdgeMetrics())
            trained = self._train_one(torch, best, metrics)
            if trained is None:
                continue
            trained.fitness.mean = breakdown.total
            trained.fitness.std = 0.0
            population.inject_gradient_individual(trained, self.hp)
            updates += 1
        self.last_update_count = updates
        return updates

    def _train_one(self, torch: Any, genome: PromptGenome, metrics: EdgeMetrics) -> Optional[PromptGenome]:
        vector = torch.tensor(genome.vector, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([vector], lr=max(self.hp.eta_p, 1e-8))
        last_loss = None
        for _ in range(max(1, self.hp.soft_prompt_steps)):
            optimizer.zero_grad()
            loss = self._semantic_loss(torch, vector, genome.edge, metrics)
            lm_loss = self._local_model_loss(vector, genome.edge)
            if lm_loss is not None:
                loss = loss + self.hp.soft_prompt_lm_loss_weight * lm_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                vector.clamp_(-self.hp.prompt_bound, self.hp.prompt_bound)
            last_loss = float(loss.detach().cpu().item())
        if last_loss is not None:
            self.last_losses[f"{genome.edge[0]}->{genome.edge[1]}"] = last_loss
        trained = genome.clone()
        trained.vector = [clip_prompt(float(value), self.hp.prompt_bound) for value in vector.detach().cpu().tolist()]
        trained.generation += 1
        return trained

    def _semantic_loss(self, torch: Any, vector: Any, edge: Edge, metrics: EdgeMetrics) -> Any:
        target = torch.tensor(edge_target_distribution(edge), dtype=vector.dtype, device=vector.device)
        logits = vector[: len(target)]
        if logits.numel() < target.numel():
            logits = torch.cat([logits, torch.zeros(target.numel() - logits.numel(), dtype=vector.dtype, device=vector.device)])
        log_probs = torch.log_softmax(logits / 0.8, dim=0)
        semantic_kl = torch.sum(target * (torch.log(target.clamp_min(1e-8)) - log_probs))

        dist = torch.softmax(logits / 0.8, dim=0)
        clarity_reward = -float(self.hp.lambda_c) * dist.max()
        error_penalty = float(metrics.edge_error if self.hp.error_channel else 0.0) * semantic_kl
        info_reward = -float(metrics.info_gain if self.hp.belief_channel else 0.0) * torch.sum(target * dist)
        norm = float(self.hp.lambda_p + self.hp.lambda_l) * torch.mean(vector * vector)
        return semantic_kl + error_penalty + info_reward + clarity_reward + norm

    def _local_model_loss(self, vector: Any, edge: Edge) -> Any:
        if not self.hp.soft_prompt_lm_loss or self.local_model is None:
            return None
        target = normalize_distribution(edge_target_distribution(edge))
        payload = json.dumps(
            {
                "edge": f"{edge[0]}->{edge[1]}",
                "targetDistribution": [round(value, 4) for value in target],
            },
            separators=(",", ":"),
        )
        target_text = json.dumps(
            {
                "finish": round(target[0], 4),
                "block_opponent": round(target[1], 4),
                "help_partner": round(target[2], 4),
                "preserve_bomb": round(target[3], 4),
                "shed_cards": round(target[4], 4),
                "low_ambiguity": round(target[5], 4),
            },
            separators=(",", ":"),
        )
        try:
            return self.local_model.nll_with_soft_prompt(
                vector,
                "Train the injected edge soft prompt to emit the target COTE message distribution.",
                payload,
                target_text,
            )
        except Exception as exc:  # noqa: BLE001 - semantic prompt training still runs.
            self.last_error = str(exc)
            return None

    def _torch(self) -> Any:
        try:
            import torch

            return torch
        except Exception as exc:  # noqa: BLE001 - allow import in light environments.
            self.last_error = f"torch is unavailable: {exc}"
            return None


def prompt_training_delta(before: list[float], after: list[float]) -> float:
    size = min(len(before), len(after))
    if size <= 0:
        return 0.0
    return math.sqrt(sum((float(after[idx]) - float(before[idx])) ** 2 for idx in range(size)) / size)
