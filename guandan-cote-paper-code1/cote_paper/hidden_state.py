# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass
class HiddenStateParticle:
    """One hidden-card assignment hypothesis.

    `owner_by_card` maps a stable card id/string to the player position that
    owns it in this hypothesis. The particle weight is the posterior mass.
    """

    owner_by_card: Dict[str, int]
    weight: float = 1.0
    metadata: Dict[str, object] = field(default_factory=dict)

    def clone(self) -> "HiddenStateParticle":
        return HiddenStateParticle(dict(self.owner_by_card), float(self.weight), dict(self.metadata))


class ParticleBeliefState:
    """Discrete posterior over hidden Guandan states.

    This is the strict-mode replacement for the previous scalar belief proxy:
    observed public actions condition the particle set, while downstream
    modules can read marginals, entropy, and GT calibration targets.
    """

    def __init__(self, particles: Sequence[HiddenStateParticle]) -> None:
        self.particles = [particle.clone() for particle in particles]
        self.normalize()

    def normalize(self) -> None:
        total = sum(max(0.0, particle.weight) for particle in self.particles)
        if not self.particles:
            return
        if total <= 0.0:
            uniform = 1.0 / len(self.particles)
            for particle in self.particles:
                particle.weight = uniform
            return
        for particle in self.particles:
            particle.weight = max(0.0, particle.weight) / total

    def condition_on_play(self, pos: int, cards: Iterable[str]) -> None:
        observed = [str(card) for card in cards]
        self.particles = [
            particle
            for particle in self.particles
            if all(particle.owner_by_card.get(card) == int(pos) for card in observed)
        ]
        self.normalize()

    def update_likelihood(self, likelihoods: Mapping[int, float]) -> None:
        for idx, particle in enumerate(self.particles):
            particle.weight *= max(0.0, float(likelihoods.get(idx, 1.0)))
        self.normalize()

    def marginal_owner_probability(self, card: str, pos: int) -> float:
        return sum(
            particle.weight
            for particle in self.particles
            if particle.owner_by_card.get(str(card)) == int(pos)
        )

    def owner_distribution(self, card: str) -> Dict[int, float]:
        distribution: Dict[int, float] = {}
        for particle in self.particles:
            owner = particle.owner_by_card.get(str(card))
            if owner is None:
                continue
            distribution[int(owner)] = distribution.get(int(owner), 0.0) + particle.weight
        return distribution

    def entropy(self) -> float:
        return -sum(
            particle.weight * math.log(max(particle.weight, 1e-12))
            for particle in self.particles
            if particle.weight > 0.0
        )

    def effective_sample_size(self) -> float:
        denom = sum(particle.weight * particle.weight for particle in self.particles)
        return 1.0 / denom if denom > 1e-12 else 0.0

    def resample(self, count: int, rng: Optional[random.Random] = None) -> "ParticleBeliefState":
        if count <= 0 or not self.particles:
            return ParticleBeliefState([])
        rng = rng or random.Random()
        cumulative: List[float] = []
        running = 0.0
        for particle in self.particles:
            running += particle.weight
            cumulative.append(running)
        sampled: List[HiddenStateParticle] = []
        for _ in range(count):
            roll = rng.random() * max(running, 1e-12)
            idx = 0
            while idx < len(cumulative) - 1 and cumulative[idx] < roll:
                idx += 1
            clone = self.particles[idx].clone()
            clone.weight = 1.0
            sampled.append(clone)
        return ParticleBeliefState(sampled)
