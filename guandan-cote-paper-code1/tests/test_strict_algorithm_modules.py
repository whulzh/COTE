import random

from cote_paper.config import HyperParams
from cote_paper.gradient_update import project_inbound_simplex
from cote_paper.joint_fitness import StrictCoteFitnessEvaluator
from cote_paper.message_semantics import parse_message_distribution
from cote_paper.model_pool import StrictModelPool
from cote_paper.strict_el_bdpea import StrictELBDPEA
from cote_paper.topology import CoteTopologyState
from cote_paper.trajectory import EdgeRecord, DecisionRecord, EpisodeRecord


def test_project_inbound_simplex_sets_nonnegative_unit_inbound():
    topology = CoteTopologyState.dense_initial(normalize_inbound=True)
    first_dst = next(iter({dst for _, dst in topology.all_edges()}))
    for edge in topology.inbound_edges(first_dst):
        topology.set_weight(edge, -1.0)
    project_inbound_simplex(topology, min_active_edges=1)
    inbound_sum = sum(topology.weight(edge) for edge in topology.inbound_edges(first_dst, threshold=0.0))
    assert round(inbound_sum, 6) == 1.0


def test_strict_fitness_reparses_edge_messages_with_q_phi():
    hp = HyperParams(strict_repro=True)
    evaluator = StrictCoteFitnessEvaluator(hp, random.Random(1))
    edge = ("T4_opponent_intent", "T8_action_decider")
    record = EdgeRecord(
        edge=edge,
        raw_message="block opponent and preserve bomb",
        parsed_distribution=[1, 0, 0, 0, 0, 0],
        before_belief=[0.5, 0.5],
        after_belief=[0.4, 0.6],
        weight=0.5,
        info_gain=0.0,
        clarity=0.0,
        edge_error=0.0,
    )
    repaired = evaluator.strict_edge_record(record)
    assert repaired.parsed_distribution == parse_message_distribution(record.raw_message)
    assert repaired.edge_error > 0.0


def test_strict_el_bdpea_uses_candidate_replay_budget():
    hp = HyperParams(strict_candidate_replays=7)
    evolver = StrictELBDPEA(hp, random.Random(1))
    assert evolver.candidate_replay_count() == 7


def test_strict_model_pool_defaults_to_shared_backend():
    pool = StrictModelPool(shared_backend=object())
    assert pool.node_backend("T1_board_parser") is pool.shared_backend
