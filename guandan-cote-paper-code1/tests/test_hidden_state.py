from cote_paper.hidden_state import HiddenStateParticle, ParticleBeliefState


def test_particle_belief_normalizes_weights():
    particles = [
        HiddenStateParticle(owner_by_card={"S3": 1}, weight=2.0),
        HiddenStateParticle(owner_by_card={"S3": 2}, weight=1.0),
    ]
    belief = ParticleBeliefState(particles)
    belief.normalize()
    assert round(sum(p.weight for p in belief.particles), 6) == 1.0
    assert round(belief.marginal_owner_probability("S3", 1), 6) == round(2.0 / 3.0, 6)


def test_observed_card_eliminates_inconsistent_particle():
    particles = [
        HiddenStateParticle(owner_by_card={"S3": 1}, weight=0.5),
        HiddenStateParticle(owner_by_card={"S3": 2}, weight=0.5),
    ]
    belief = ParticleBeliefState(particles)
    belief.condition_on_play(pos=1, cards=["S3"])
    assert len(belief.particles) == 1
    assert belief.particles[0].owner_by_card["S3"] == 1


def test_condition_keeps_normalized_fallback_when_all_particles_removed():
    particles = [HiddenStateParticle(owner_by_card={"S3": 2}, weight=1.0)]
    belief = ParticleBeliefState(particles)
    belief.condition_on_play(pos=1, cards=["S3"])
    assert belief.particles == []
    assert belief.entropy() == 0.0
