from cote_paper.joint_fitness import StrictCoteFitnessEvaluator
from cote_paper.policy import PaperCOTEAgent
from cote_paper.strict_el_bdpea import StrictELBDPEA


def test_strict_repro_agent_uses_strict_algorithm_modules(monkeypatch):
    monkeypatch.setenv("COTE_STRICT_REPRO", "1")
    monkeypatch.setenv("COTE_DISABLE_STATE", "1")
    agent = PaperCOTEAgent(client_id=1, seed=123)
    assert isinstance(agent.evaluator, StrictCoteFitnessEvaluator)
    assert isinstance(agent.evolver, StrictELBDPEA)
    assert agent.topology.edge_retention == 56 / 64
    for _, dst in agent.topology.all_edges():
        inbound_sum = sum(agent.topology.weight((src, dst)) for src, _ in agent.topology.inbound_edges(dst, threshold=0.0))
        assert round(inbound_sum, 6) == 1.0
