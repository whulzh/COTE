# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Mapping, Sequence

from .config import HyperParams
from .nodes import NODE_KEYS
from .prompts import PromptGenome, PromptPopulation, clip_prompt
from .topology import CoteTopologyState, Edge


def project_inbound_simplex(topology: CoteTopologyState, min_active_edges: int = 1) -> None:
    """Project each inbound column of W onto a nonnegative simplex.

    Algorithm 2's projection step keeps communication mass normalized per
    receiver node. If pruning removes all inbound edges, strict mode restores
    a uniform minimal inbound set so later gradients still have a valid domain.
    """

    min_active_edges = max(1, int(min_active_edges))
    for dst in NODE_KEYS:
        inbound = [(src, dst) for src in NODE_KEYS if src != dst]
        positive = [edge for edge in inbound if topology.weight(edge) > 0.0]
        if not positive:
            for edge in inbound[:min_active_edges]:
                topology.set_weight(edge, 1.0)
            positive = [edge for edge in inbound if topology.weight(edge) > 0.0]
        total = sum(max(0.0, topology.weight(edge)) for edge in positive)
        if total <= 1e-12:
            uniform = 1.0 / len(positive)
            for edge in positive:
                topology.set_weight(edge, uniform)
            continue
        for edge in positive:
            topology.set_weight(edge, max(0.0, topology.weight(edge)) / total)


def prompt_gradient_step(
    population: PromptPopulation,
    prompt_gradients: Mapping[Edge, Sequence[float]],
    hp: HyperParams,
) -> None:
    """Lamarckian injection for equation 18 prompt updates."""

    for edge, gradient in prompt_gradients.items():
        try:
            best = population.best(edge)
        except KeyError:
            continue
        vector = list(best.vector)
        for idx, grad in enumerate(gradient[: len(vector)]):
            vector[idx] = clip_prompt(vector[idx] + hp.eta_p * float(grad), hp.prompt_bound)
        injected = PromptGenome(
            edge=edge,
            vector=vector,
            reachable_belief=list(best.reachable_belief),
            noise=list(best.noise),
            sigma=best.sigma,
            fitness=best.fitness,
            generation=best.generation + 1,
        )
        population.inject_gradient_individual(injected, hp)


def apply_bidirectional_gradient_refinement(
    topology: CoteTopologyState,
    population: PromptPopulation,
    topology_gradients: Mapping[Edge, float],
    prompt_gradients: Mapping[Edge, Sequence[float]],
    hp: HyperParams,
) -> None:
    topology.apply_gradients(topology_gradients, hp.eta_w, hp.theta_grad)
    prompt_gradient_step(population, prompt_gradients, hp)
    project_inbound_simplex(topology)
