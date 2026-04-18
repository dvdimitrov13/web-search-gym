"""Smoke tests for BrowseComp loader pieces that don't need network."""

from __future__ import annotations


def test_decrypt_roundtrips():
    """decrypt(encrypt(x, k), k) == x — sanity check on our XOR scheme.

    We don't ship an encrypt function, so we simulate one inline and verify
    decrypt recovers the plaintext. If someone breaks the SHA256 / XOR
    contract, BrowseComp rows will silently decrypt to garbage.
    """
    import base64

    from bench.browsecomp import decrypt, derive_key

    plaintext = "What is the capital of France?"
    password = "canary-xyz-123"
    key = derive_key(password, len(plaintext.encode()))
    encrypted = bytes(a ^ b for a, b in zip(plaintext.encode(), key))
    ct_b64 = base64.b64encode(encrypted).decode()
    assert decrypt(ct_b64, password) == plaintext


def test_derive_key_deterministic():
    from bench.browsecomp import derive_key

    k1 = derive_key("pw", 64)
    k2 = derive_key("pw", 64)
    assert k1 == k2
    assert len(k1) == 64


def test_answer_to_browsecomp_text_format():
    from bench.browsecomp import answer_to_browsecomp_text
    from core.types import Answer

    a = Answer(explanation="it's Paris", exact_answer="Paris", confidence=95)
    text = answer_to_browsecomp_text(a)
    assert "Explanation: it's Paris" in text
    assert "Exact Answer: Paris" in text
    assert "Confidence: 95" in text


def test_splits_yaml_valid():
    """splits.yaml parses and has the three expected splits."""
    from pathlib import Path
    import yaml

    path = Path(__file__).resolve().parent.parent / "bench" / "splits.yaml"
    data = yaml.safe_load(path.read_text())
    assert set(data.keys()) == {"dev", "smoke", "full"}
    assert isinstance(data["dev"]["indices"], list) and len(data["dev"]["indices"]) == 10
    assert isinstance(data["smoke"]["indices"], list) and len(data["smoke"]["indices"]) == 50
    assert data["full"]["indices"] is None
