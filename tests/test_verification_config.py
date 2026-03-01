import os
import pytest

from src.config.verification_config import VerificationConfig, VerificationConfigError


def test_default_config_is_valid():
    cfg = VerificationConfig()
    assert cfg.verified_confidence_threshold == 0.70
    assert cfg.rejected_confidence_threshold == 0.30
    assert cfg.low_confidence_range == (0.30, 0.70)


def test_from_env_overrides_and_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIFIED_CONFIDENCE_THRESHOLD", "0.8")
    monkeypatch.setenv("REJECTED_CONFIDENCE_THRESHOLD", "0.2")
    monkeypatch.setenv("LOW_CONFIDENCE_LO", "0.25")
    monkeypatch.setenv("LOW_CONFIDENCE_HI", "0.85")
    monkeypatch.setenv("MIN_ENTAILING_SOURCES_FOR_VERIFIED", "3")

    cfg = VerificationConfig.from_env()
    assert cfg.verified_confidence_threshold == pytest.approx(0.8)
    assert cfg.rejected_confidence_threshold == pytest.approx(0.2)
    assert cfg.low_confidence_range == (0.25, 0.85)
    assert cfg.min_entailing_sources_for_verified == 3


def test_invalid_ranges_raise():
    # rejected >= low -> invalid
    with pytest.raises(VerificationConfigError):
        VerificationConfig(rejected_confidence_threshold=0.4, low_confidence_range=(0.3, 0.7))

    # mmr lambda out of range
    with pytest.raises(VerificationConfigError):
        VerificationConfig(mmr_lambda=1.5)

    # top_k invalid
    with pytest.raises(VerificationConfigError):
        VerificationConfig(top_k_rerank=0)
