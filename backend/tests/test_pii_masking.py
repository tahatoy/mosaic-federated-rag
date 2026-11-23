"""Kişisel veri maskeleme v2 birim testleri (ağır bağımlılık yüklemeden, ast ile)."""
import ast, os, re, types
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "ai_engine.py").read_text(encoding="utf-8")
_keep = [n for n in ast.parse(SRC).body if (isinstance(n, ast.FunctionDef) and (n.name == "_filter_pii" or n.name.startswith("_pii_")))
         or (isinstance(n, ast.Assign) and any(getattr(t, "id", "").startswith("_PII_") for t in n.targets))]
_ns = {"re": re, "os": os, "logger": types.SimpleNamespace(info=lambda *a, **k: None)}
exec(compile(ast.Module(body=_keep, type_ignores=[]), "ai_engine", "exec"), _ns)
mask = _ns["_filter_pii"]

def test_tc_checksum_and_separators():
    assert "10000000146" not in mask("Kimlik 10000000146 kayıtlı", {})       # geçerli TC
    assert "12345678901" in mask("Ref 12345678901", {})                      # geçersiz sağlama -> maskelenmez

def test_iban_mod97_case_insensitive():
    out = mask("IBAN: tr33 0006 1005 1978 6457 8413 26", {})
    assert "[IBAN_1]" in out

def test_card_luhn_layout():
    assert "[CREDIT_CARD_1]" in mask("Kart 4111-1111-1111-1111 bloke", {})
    assert "[CREDIT_CARD" not in mask("Aktif 8.104.023 25.518.056 TL", {})

def test_phone_variants():
    for p in ["0532 123 45 67", "0(532)1234567", "+90-532-123-4567", "0532.123.45.67"]:
        assert "[PHONE_1]" in mask(f"Tel {p} arayın", {}), p

def test_names_and_false_positives():
    assert "[NAME_1]" in mask("Sayın Burak Aksoy, başvurunuz alındı.", {})
    for s in ["Sermaye Piyasası Kurulu yeni tebliğ yayımladı.", "Ramazan Bayramı tatili", "Kredi tutarı 1.250.000 TL"]:
        assert mask(s, {}) == s, s

def test_mapping_reuse():
    m = {}
    out = mask("burak@ornek.com ve burak@ornek.com", m)
    assert out.count("[EMAIL_1]") == 2 and len(m) == 1
