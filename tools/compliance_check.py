"""KVKK/BDDK teknik tedbir doğrulaması (çalışan dağıtıma karşı). Çıktı: ai/experiments/results/kvkk_bddk_dogrulama.json"""
import json, os, re, ssl, subprocess, sys, socket
from pathlib import Path
import httpx
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "backend"))
from compliance.audit import AuditLog
R = ROOT / "ai/experiments/results"; ORCH = "http://127.0.0.1:8004"
B = {"branch_01": os.environ["BRANCH1_RAG_URL"], "branch_02": os.environ["BRANCH2_RAG_URL"]}
CA, CC, CK = "backend/certs/ca.crt", "backend/certs/hq_client.crt", "backend/certs/hq_client.key"
def _mtls(verify, cert):
    """httpx 0.28 uyumlu mTLS bağlamı: verify=<yol> iken cert= yok sayıldığı için sertifikayı SSLContext'e yükle."""
    import ssl as _ssl
    if verify is False and not cert:
        return False
    if verify is False:
        ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_CLIENT); ctx.check_hostname = False; ctx.verify_mode = _ssl.CERT_NONE
    else:
        ctx = _ssl.create_default_context(cafile=verify) if isinstance(verify, str) else _ssl.create_default_context()
    if cert and cert[0]:
        ctx.load_cert_chain(*cert)
    return ctx

keys = json.load(open(ROOT / "logs/.test_keys"))
out = {"kontroller": []}
def chk(ad, dayanak, gecti, ayrinti):
    out["kontroller"].append({"kontrol": ad, "dayanak": dayanak, "gecti": bool(gecti), "ayrinti": ayrinti}); print(("[OK]" if gecti else "[FAIL]"), ad, "|", ayrinti)

# 1-3 Kimlik doğrulama / yetkilendirme (orkestratör)
q = {"query": "Mevduat faizi nasıl hesaplanır?", "branch_id": 1}
r0 = httpx.post(f"{ORCH}/query", json=q, timeout=120); chk("Anahtarsız sorgu reddedilir", "BDDK BS Yön. kimlik doğrulama; KVKK Rehberi erişim", r0.status_code == 401, r0.status_code)
r1 = httpx.post(f"{ORCH}/query", json=q, headers={"X-API-Key": keys["analist"]}, timeout=180); chk("Analist anahtarıyla sorgu", "yetki matrisi", r1.status_code == 200, r1.status_code)
r2 = httpx.post(f"{ORCH}/erase", json={"contains": ["x"]}, headers={"X-API-Key": keys["analist"]}, timeout=60); chk("Analist imha yapamaz (en az yetki)", "KVKK Rehberi yetki matrisi", r2.status_code == 403, r2.status_code)
r3 = httpx.post(f"{ORCH}/query", json=q, headers={"X-API-Key": "msk_sahte"}, timeout=60); chk("Geçersiz anahtar reddedilir", "kimlik doğrulama", r3.status_code == 401, r3.status_code)
# 4-6 Aktarımda şifreleme / mTLS
for name, url in B.items():
    if not url.startswith("https://"):
        chk(f"{name}: mTLS/kaynakta maskeleme", "—", False, "UYGULANMADI: VM1 erişim bilgisi yok, eski yapılandırmada kontrol grubu olarak bırakıldı"); continue
    try: httpx.get(url.replace("https://", "http://") + "/health", timeout=5); plain = "yanıt verdi"
    except Exception as e: plain = f"reddedildi ({type(e).__name__})"
    chk(f"{name}: düz HTTP kabul edilmez", "aktarımda şifreleme", "reddedildi" in plain, plain)
    try: httpx.get(url + "/health", verify=CA, timeout=5); nocert = "yanıt verdi"
    except Exception as e: nocert = f"reddedildi ({type(e).__name__})"
    chk(f"{name}: istemci sertifikasız TLS reddedilir (mTLS)", "BDDK karşılıklı kimlik doğrulama", "reddedildi" in nocert, nocert)
    rr = httpx.get(url + "/health", verify=_mtls(CA, (CC, CK)), timeout=5)
    host = url.split("//")[1].split(":")[0]; port = int(url.rsplit(":", 1)[1])
    ctx = ssl.create_default_context(cafile=CA); ctx.load_cert_chain(CC, CK)
    with socket.create_connection((host, port), 5) as so, ctx.wrap_socket(so, server_hostname=host) as ss: ver, cip = ss.version(), ss.cipher()[0]
    chk(f"{name}: mTLS ile erişim ve protokol", "TLS 1.2+", rr.status_code == 200 and ver in ("TLSv1.2", "TLSv1.3"), f"{rr.status_code} {ver} {cip}")
# 7 Ağ ayrımı: iç servisler yalnızca loopback
lan = os.getenv("HQ_LAN_IP", "10.0.0.10")
for port in (8000, 8003, 8098, 8099):
    try: socket.create_connection((lan, port), 2).close(); st = "açık"
    except OSError: st = "kapalı"
    chk(f"HQ iç servis {port} LAN'a kapalı", "BDDK ağ güvenliği/segmentasyon", st == "kapalı", f"{lan}:{port} {st}")
# 8 Kaynakta maskeleme (kanarya, şube cevabında açık sır)
can = json.load(open(R / "h2_canaries.json")); leak = 0; tot = 0; name_leak = 0
with httpx.Client(verify=_mtls(CA, (CC, CK)), timeout=60) as cl:
    for c in can:
        bu = B["branch_0%d" % c["branch"]]
        if not bu.startswith("https://"): continue  # kontrol grubu şube hariç
        ch = cl.post(f"{bu}/query", json={"query": c["ad"] + " müşteri dosyası kredi başvurusu", "top_k": 5, "use_reranker": False}).json()["chunks"]
        t = re.sub(r"\D", "", " ".join(x["text"] for x in ch)); tt = " ".join(x["text"] for x in ch)
        for s in c["sırlar"]:
            tot += 1; leak += int((re.sub(r"\D", "", s) in t) if not s.startswith("TR") else s in tt.replace(" ", ""))
        name_leak += int(c["ad"] in tt)
chk("Şube→HQ hattında açık sayısal sır yok (kaynakta maskeleme)", "KVKK veri minimizasyonu / maskeleme", leak == 0, f"{leak}/{tot} sır açık; ad açık {name_leak}/{tot//4} (yalnızca mTLS+maskelemeli şube)")
# 9 İz kaydı: bütünlük + ham veri içermeme
a = AuditLog(directory=ROOT / "logs/audit"); ok, n, bad = a.verify()
chk("HQ iz kaydı zinciri doğrulanır", "BDDK iz kayıt mekanizması (bütünlük)", ok and n > 0, f"{n} kayıt, bozuk={bad}")
raw = (ROOT / "logs/audit").glob("audit-*.jsonl"); txt = "".join(p.read_text(encoding="utf-8") for p in raw)
secrets_in_audit = sum(1 for c in can for s in c["sırlar"] if s in txt) + sum(1 for c in can if c["ad"] in txt)
chk("İz kaydında kişisel veri yok", "KVKK veri minimizasyonu", secrets_in_audit == 0, f"{secrets_in_audit} eşleşme")
logtxt = "".join((ROOT / "logs" / f).read_text(encoding="utf-8", errors="ignore") for f in ("orch.log", "hq.log", "router.log") if (ROOT / "logs" / f).exists())
lk = sum(1 for c in can for s in c["sırlar"] if s in logtxt)
chk("HQ uygulama loglarında kanarya sırrı yok", "KVKK Rehberi: loglarda kişisel veri", lk == 0, f"{lk} eşleşme")
# 10 İmha (ilgili kişi silme talebi) uçtan uca
victim = next(c for c in can if B["branch_0%d" % c["branch"]].startswith("https://")); tc = victim["sırlar"][0]; vb = B["branch_0%d" % victim["branch"]]
re_ = httpx.post(f"{ORCH}/erase", json={"contains": [tc]}, headers={"X-API-Key": keys["yonetici"]}, timeout=120)
with httpx.Client(verify=_mtls(CA, (CC, CK)), timeout=60) as cl:
    after = cl.post(f"{vb}/query", json={"query": victim["ad"] + " müşteri dosyası kredi başvurusu", "top_k": 5, "use_reranker": False}).json()["chunks"]
chk("İmha talebi kaydı şubeden siler", "KVKK md.7 silme/yok etme", re_.status_code == 200 and not any(x["id"] == victim["id"] for x in after), f"{re_.status_code} {json.dumps(re_.json(), ensure_ascii=False)[:200]}")
ok2, n2, _ = a.verify(); chk("İmha işlemi iz kaydına işlendi ve zincir geçerli", "hesap verebilirlik", ok2 and "imha_talebi" in "".join(x.read_text(encoding="utf-8") for x in (ROOT / "logs/audit").glob("audit-*.jsonl")), f"{n2} kayıt")
# 11 Veri yerelliği: dış ağ çağrısı yapan LLM/API yok
ext = subprocess.run(["grep", "-rlE", "api\\.openai\\.com|anthropic\\.com|generativelanguage|api\\.cohere", "backend", "--include=*.py"], capture_output=True, text=True).stdout.strip()
chk("Dış yapay zekâ API çağrısı yok (veri yurt içinde, şirket içinde işlenir)", "BDDK yurt içi birincil sistem / KVKK md.9", ext == "", ext or "yok")
out["ozet"] = {"gecen": sum(c["gecti"] for c in out["kontroller"]), "toplam": len(out["kontroller"])}
(R / "kvkk_bddk_dogrulama.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"); print(out["ozet"])
