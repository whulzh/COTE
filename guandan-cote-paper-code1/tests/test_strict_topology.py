from cote_paper.topology import CoteTopologyState
from cote_paper.nodes import NODE_KEYS


def test_dense_initial_strict_inbound_normalizes_non_self_edges():
    topology = CoteTopologyState.dense_initial(normalize_inbound=True)
    for dst in NODE_KEYS:
        inbound_sum = sum(topology.weight((src, dst)) for src in NODE_KEYS if src != dst)
        assert round(inbound_sum, 6) == 1.0


def test_prune_can_renormalize_inbound(monkeypatch):
    monkeypatch.setenv("COTE_NORMALIZE_TOPOLOGY", "1")
    topology = CoteTopologyState.dense_initial(normalize_inbound=True)
    topology.set_weight((NODE_KEYS[0], NODE_KEYS[1]), 0.001)
    topology.prune(0.01)
    inbound_sum = sum(topology.weight((src, NODE_KEYS[1])) for src in NODE_KEYS if src != NODE_KEYS[1])
    assert round(inbound_sum, 6) == 1.0
