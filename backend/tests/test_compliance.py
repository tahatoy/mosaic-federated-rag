"""KVKK/BDDK teknik tedbir birim testleri: iz kaydı bütünlüğü, saklama-imha, erişim yetki matrisi."""
import json, sys
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compliance.audit import AuditLog
from compliance import access

def test_chain_and_tamper(tmp_path):
    a = AuditLog(tmp_path, "t", "k" * 32)
    for i in range(4): a.record("http", "api:x", path="/query", i=i)
    assert a.verify() == (True, 4, None)
    fp = next(tmp_path.glob("audit-*.jsonl")); L = fp.read_text().splitlines()
    e = json.loads(L[1]); e["actor"] = "degisti"; L[1] = json.dumps(e); fp.write_text("\n".join(L) + "\n")
    ok, n, bad = a.verify(); assert not ok and "imza" in bad

def test_deletion_breaks_chain(tmp_path):
    a = AuditLog(tmp_path, "t", "k" * 32)
    for i in range(4): a.record("http", "x", i=i)
    fp = next(tmp_path.glob("audit-*.jsonl")); L = fp.read_text().splitlines(); del L[1]; fp.write_text("\n".join(L) + "\n")
    ok, _, bad = a.verify(); assert not ok and "zincir" in bad

def test_no_raw_text_only_digest(tmp_path):
    a = AuditLog(tmp_path, "t", "k" * 32); a.record("rag_sorgu", "x", sorgu_ozet=a.digest("Ahmet Yılmaz 10000000146"))
    assert "10000000146" not in next(tmp_path.glob("audit-*.jsonl")).read_text()

def test_retention_purge(tmp_path):
    a = AuditLog(tmp_path, "t", "k" * 32, retention_days=30)
    old = tmp_path / f"audit-{(datetime.now() - timedelta(days=40)):%Y%m%d}.jsonl"; old.write_text("{}\n")
    assert old.name in a.purge() and not old.exists()
    assert "audit_purge" in "".join(p.read_text() for p in tmp_path.glob("audit-*.jsonl"))

def test_role_matrix(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEYS_FILE", str(tmp_path / "k.json"))
    k = access.create_key("a1", "analist"); assert access.authenticate(k) == ("a1", "analist")
    assert "sha256" in json.loads((tmp_path / "k.json").read_text())["a1"] and k not in (tmp_path / "k.json").read_text()
    assert access.authenticate("msk_yanlis") is None
    assert access.is_allowed("analist", "/query") and not access.is_allowed("analist", "/erase") and access.is_allowed("yonetici", "/erase")
