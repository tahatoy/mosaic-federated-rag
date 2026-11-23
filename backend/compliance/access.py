"""
Erişim denetimi (KVKK Rehberi: yetki matrisi, en az yetki; BDDK: kimlik doğrulama ve yetkilendirme).
API anahtarları düz metin değil SHA-256 özetiyle saklanır: config/api_keys.json
  {"anahtar_kimligi": {"sha256": "<özet>", "rol": "analist|denetci|yonetici", "aktif": true}}
Yetki matrisi aşağıdaki ROLE_PERMS ile tanımlıdır. AUTH_REQUIRED=true ise /health dışındaki uçlar anahtar ister.
Şube RAG servisleri ayrıca mTLS ile korunur (yalnızca kurum CA'sının imzaladığı istemci sertifikaları bağlanabilir).
"""
import hashlib, hmac, json, os, secrets
from pathlib import Path

ROLE_PERMS = {
    "analist":  {"/query", "/query/stream"},
    "denetci":  {"/query", "/query/stream", "/audit/verify"},
    "yonetici": {"/query", "/query/stream", "/audit/verify", "/erase", "/reset", "/reset_hq", "/add_chunks", "/upload"},
}
PUBLIC = {"/health", "/docs", "/openapi.json"}


def _keyfile():
    return Path(os.getenv("API_KEYS_FILE", "config/api_keys.json"))


def load_keys():
    f = _keyfile()
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def create_key(key_id: str, role: str) -> str:
    """Yeni anahtar üretir, yalnızca özetini kaydeder; düz anahtarı bir kez döndürür."""
    assert role in ROLE_PERMS, role
    raw = "msk_" + secrets.token_urlsafe(32)
    keys = load_keys(); keys[key_id] = {"sha256": hashlib.sha256(raw.encode()).hexdigest(), "rol": role, "aktif": True}
    f = _keyfile(); f.parent.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(keys, indent=1), encoding="utf-8")
    try: os.chmod(f, 0o600)
    except OSError: pass
    return raw


def authenticate(raw_key: str):
    """(anahtar_kimliği, rol) veya None. Sabit zamanlı karşılaştırma."""
    if not raw_key: return None
    h = hashlib.sha256(raw_key.encode()).hexdigest()
    for kid, k in load_keys().items():
        if k.get("aktif", True) and hmac.compare_digest(h, k["sha256"]): return kid, k["rol"]
    return None


def is_allowed(role: str, path: str) -> bool:
    return path in PUBLIC or path in ROLE_PERMS.get(role, set())


def auth_required() -> bool:
    return os.getenv("AUTH_REQUIRED", "false").lower() == "true"
