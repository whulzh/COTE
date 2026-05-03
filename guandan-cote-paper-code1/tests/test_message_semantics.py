from cote_paper.message_semantics import parse_message_distribution, kl_divergence, SEMANTIC_AXES


def test_json_microcode_message_distribution():
    msg = (
        '{"finish":0.1,"block_opponent":0.7,"help_partner":0.1,'
        '"preserve_bomb":0.05,"shed_cards":0.03,"low_ambiguity":0.02}'
    )
    dist = parse_message_distribution(msg)
    assert len(dist) == len(SEMANTIC_AXES)
    assert round(sum(dist), 6) == 1.0
    assert dist[1] > dist[0]


def test_text_message_distribution_fallback():
    dist = parse_message_distribution("block opponent and preserve bomb")
    assert dist[1] > 0.25
    assert dist[3] > 0.20


def test_kl_zero_for_equal_distribution():
    assert kl_divergence([0.5, 0.5], [0.5, 0.5]) < 1e-8
