"""
Değiştirilemezliği doğrulanabilir iz kaydı (BDDK BS Yönetmeliği: iz kayıt mekanizması; KVKK Rehberi: işlem kayıtları).
- Her kayıt bir önceki kaydın özetini (prev) içerir; HMAC-SHA256 ile imzalanır -> silme/değiştirme/ekleme tespit edilir.
- Ham sorgu/metin YAZILMAZ: yalnızca maskeli sorgunun özeti (sha256), belge kimlikleri, istemci kimliği, sonuç.
- Günlük dosya: <dir>/audit-YYYYMMDD.jsonl ; saklama süresi AUDIT_RETENTION_DAYS (varsayılan 730 gün) sonunda imha.
"""
import fcntl, hashlib, hmac, json, os, sys, threading, uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

_TZ = timezone(timedelta(hours=3))  # Türkiye saati


class AuditLog:
    def __init__(self, directory=None, node=None, key=None, retention_days=None):
        self.dir = Path(directory or os.getenv("AUDIT_DIR", "logs/audit")); self.dir.mkdir(parents=True, exist_ok=True)
        svc = sys.argv[sys.argv.index("--service") + 1] if "--service" in sys.argv[:-1] else None
        self.node = node or os.getenv("BRANCH_ID") or os.getenv("SERVICE_TYPE") or svc or "node"
        k = key or os.getenv("AUDIT_HMAC_KEY")
        if not k:  # anahtar yoksa düğüme özel kalıcı anahtar üret (dosya izni 600)
            kf = self.dir / ".audit_key"
            if not kf.exists():
                kf.write_text(uuid.uuid4().hex + uuid.uuid4().hex); os.chmod(kf, 0o600)
            k = kf.read_text().strip()
        self.key = k.encode()
        self.retention = int(retention_days or os.getenv("AUDIT_RETENTION_DAYS", "730"))
        self._lock = threading.Lock(); self._state = self.dir / ".last_hash"
        self._last = self._state.read_text().strip() if self._state.exists() else "GENESIS"

    @staticmethod
    def digest(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    def _sign(self, entry: dict) -> str:
        body = json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hmac.new(self.key, body.encode("utf-8"), hashlib.sha256).hexdigest()

    def record(self, event: str, actor: str, **fields) -> str:
        """Aynı dizine birden çok süreç (yönlendirici, HQ RAG, orkestratör) yazar: tek doğrusal zincir için
        dosya kilidi (fcntl) altında son özet diskten okunur, kayıt eklenir, özet güncellenir."""
        now = datetime.now(_TZ)
        with self._lock, open(self.dir / ".lock", "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                prev = self._state.read_text().strip() if self._state.exists() else "GENESIS"
                entry = {"ts": now.isoformat(timespec="milliseconds"), "node": self.node, "event": event, "actor": actor,
                         "rid": fields.pop("rid", uuid.uuid4().hex[:16]), "prev": prev or "GENESIS", **fields}
                entry["mac"] = self._sign(entry)
                with open(self.dir / f"audit-{now:%Y%m%d}.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
                tmp = self._state.with_suffix(".tmp"); tmp.write_text(entry["mac"]); os.replace(tmp, self._state)
                self._last = entry["mac"]
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)
        return entry["rid"]

    def verify(self):
        """Zinciri baştan doğrular. Dönüş: (geçerli_mi, kayıt_sayısı, ilk_bozuk_konum)."""
        prev, n = None, 0
        for fp in sorted(self.dir.glob("audit-*.jsonl")):
            for i, line in enumerate(open(fp, encoding="utf-8"), 1):
                e = json.loads(line); mac = e.pop("mac")
                if prev is not None and e["prev"] != prev: return False, n, f"{fp.name}:{i} (zincir kopuk)"
                if not hmac.compare_digest(mac, self._sign(e)): return False, n, f"{fp.name}:{i} (imza geçersiz)"
                prev, n = mac, n + 1
        return True, n, None

    def purge(self):
        """Saklama süresi dolan günlük dosyaları imha eder; imhanın kendisi de iz kaydına yazılır."""
        limit = datetime.now(_TZ).date() - timedelta(days=self.retention); removed = []
        for fp in self.dir.glob("audit-*.jsonl"):
            if datetime.strptime(fp.stem[6:], "%Y%m%d").date() < limit:
                fp.unlink(); removed.append(fp.name)
        if removed: self.record("audit_purge", "system", files=removed, retention_days=self.retention)
        return removed


_INSTANCE = None
def get_audit() -> AuditLog:
    global _INSTANCE
    if _INSTANCE is None: _INSTANCE = AuditLog()
    return _INSTANCE
