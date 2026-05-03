import pytest

from run_danzero_mirrored_evaluation import parse_args, validate_duplicate_randomization


def test_duplicate_randomization_requires_exact_deal_replay(monkeypatch):
    monkeypatch.delenv("COTE_EXACT_DEAL_REPLAY", raising=False)
    args = parse_args(["--duplicate-randomization"])
    with pytest.raises(RuntimeError, match="exact deal replay"):
        validate_duplicate_randomization(args)


def test_duplicate_randomization_guard_passes_when_server_declares_support(monkeypatch):
    monkeypatch.setenv("COTE_EXACT_DEAL_REPLAY", "1")
    args = parse_args(["--duplicate-randomization"])
    validate_duplicate_randomization(args)
