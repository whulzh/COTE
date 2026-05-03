from cote_paper.config import HyperParams


def test_strict_repro_config_defaults_off(monkeypatch):
    monkeypatch.delenv("COTE_STRICT_REPRO", raising=False)
    hp = HyperParams()
    assert hp.strict_repro is False


def test_strict_repro_config_enables_theory_paths(monkeypatch):
    monkeypatch.setenv("COTE_STRICT_REPRO", "1")
    monkeypatch.setenv("COTE_NORMALIZE_TOPOLOGY", "1")
    hp = HyperParams()
    assert hp.strict_repro is True
    assert hp.strict_replay_particles >= 256
    assert hp.strict_message_samples >= 1
    assert hp.strict_candidate_replays >= 1
