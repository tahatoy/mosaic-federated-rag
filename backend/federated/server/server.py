"""
Faz 5.1 — Flower Federated Learning Sunucu (HQ)
=================================================
FedAvg varsayılan strateji + FedProx isteğe bağlı.
Her round sonunda MLflow'a toplam / şube metriklerini loglar.

Portlar:
  9080  ->  gRPC (Flower standart)
  9081  ->  Prometheus metrics (isteğe bağlı)
"""

import os
import logging
logging.basicConfig(level=logging.INFO)
import random
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import Metrics, Parameters
from flwr.server.client_proxy import ClientProxy
from loguru import logger
from fastapi import FastAPI, Request
import uvicorn

random.seed(42)
np.random.seed(42)

# ─── ORTAM DEĞİŞKENLERİ ───────────────────────────────────────────────────────
NUM_ROUNDS         = int(os.getenv("FL_ROUNDS",           "5"))
MIN_CLIENTS        = int(os.getenv("FL_MIN_CLIENTS",      os.getenv("FL_MIN_BRANCHES", "1")))
MIN_EVAL_CLIENTS   = int(os.getenv("FL_MIN_EVAL_CLIENTS", os.getenv("FL_MIN_BRANCHES", "1")))
FRACTION_FIT       = float(os.getenv("FL_FRACTION_FIT",   "1.0"))
FRACTION_EVAL      = float(os.getenv("FL_FRACTION_EVAL",  "1.0"))
SERVER_ADDRESS     = os.getenv("FL_SERVER_ADDRESS",        "0.0.0.0:9080")
MLFLOW_URI         = os.getenv("MLFLOW_TRACKING_URI",      "http://localhost:5000")
USE_FEDPROX        = os.getenv("FL_USE_FEDPROX",           "false").lower() == "true"
FEDPROX_MU        = float(os.getenv("FL_FEDPROX_MU",      "0.01"))
CERT_DIR           = Path(os.getenv("CERT_DIR", "backend/certs"))

# ─── MLflow ───────────────────────────────────────────────────────────────────

def _init_mlflow(experiment_name: str = "federated-learning") -> Optional[str]:
    import urllib.parse
    import socket
    try:
        parsed = urllib.parse.urlparse(MLFLOW_URI)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5000
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        result = sock.connect_ex((host, port))
        sock.close()
        if result != 0:
            logger.debug("MLflow sunucusu aktif değil, MLflow entegrasyonu atlanıyor.")
            return None
    except Exception as e:
        logger.debug(f"MLflow port kontrolü başarısız: {e}")
        return None

    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(experiment_name)
        # Check if run already active
        if mlflow.active_run():
            mlflow.end_run()
        run = mlflow.start_run(run_name=f"fl-server-{int(time.time())}")
        logger.info(f"MLflow run: {run.info.run_id}")
        return run.info.run_id
    except (ImportError, Exception) as e:
        logger.debug(f"MLflow başlatılamadı (isteğe bağlı): {e}")
        return None


def _log_round(run_id: Optional[str], rnd: int, metrics: dict):
    if run_id is None:
        return
    try:
        import mlflow
        with mlflow.start_run(run_id=run_id):
            for k, v in metrics.items():
                mlflow.log_metric(k, v, step=rnd)
    except (ImportError, Exception) as e:
        logger.warning(f"MLflow log hatası: {e}")


# ─── TOPLAMA FONKSİYONLARI ────────────────────────────────────────────────────

def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Ağırlıklı ortalama ile fit metriklerini toplar."""
    if not metrics:
        return {}
    total_examples = sum(n for n, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: Dict[str, float] = {}
    keys = metrics[0][1].keys()
    for key in keys:
        # Hepsi numeric mi kontrol et
        all_numeric = all(isinstance(m[key], (int, float)) for n, m in metrics)
        if all_numeric:
            # Ağırlıklı ortalama
            aggregated[key] = sum(n * m[key] for n, m in metrics) / total_examples
        # String/non-numeric keys atlanır
    return aggregated


def evaluate_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Tek yönlü ortalama — şube başına accuracy/F1."""
    if not metrics:
        return {}
    total_examples = sum(n for n, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: Dict[str, float] = {}
    keys = metrics[0][1].keys()
    for key in keys:
        # Hepsi numeric mi kontrol et
        all_numeric = all(isinstance(m[key], (int, float)) for n, m in metrics)
        if all_numeric:
            # Ağırlıklı ortalama
            aggregated[key] = sum(n * m[key] for n, m in metrics) / total_examples
        # String/non-numeric keys atlanır
    return aggregated


# ─── ÖZEL STRATEJİ ────────────────────────────────────────────────────────────

class MosaicStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg üzerine:
      - Her round sonunda MLflow'a metrik logu
      - Opsiyonel FedProx proximal term
      - 0 örnekli şubeleri filtreleme (ZeroDivisionError koruması)
    """

    def __init__(self, mlflow_run_id: Optional[str], **kwargs):
        super().__init__(**kwargs)
        self.mlflow_run_id = mlflow_run_id

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Metrics]:
        # KRİTİK: 0 örnek bildiren şubeleri filtrele (division by zero önler)
        valid_results = [res for res in results if res[1].num_examples > 0]

        if not valid_results:
            logger.warning(f"[Round {server_round}] Hiçbir geçerli sonuç (num_examples > 0) alınamadı. Atlanıyor.")
            return None, {}

        agg_params, agg_metrics = super().aggregate_fit(server_round, valid_results, failures)
        if agg_metrics:
            logger.info(f"[Round {server_round}] fit metrikleri: {agg_metrics}")
            _log_round(self.mlflow_run_id, server_round, {f"fit/{k}": v for k, v in agg_metrics.items()})
        return agg_params, agg_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures,
    ) -> Tuple[Optional[float], Metrics]:
        # KRİTİK: 0 örnek bildiren şubeleri filtrele
        valid_results = [res for res in results if res[1].num_examples > 0]

        if not valid_results:
            return None, {}

        loss, agg_metrics = super().aggregate_evaluate(server_round, valid_results, failures)
        if agg_metrics:
            logger.info(f"[Round {server_round}] eval metrikleri: {agg_metrics}")
            _log_round(self.mlflow_run_id, server_round, {f"eval/{k}": v for k, v in agg_metrics.items()})
        return loss, agg_metrics

    def configure_fit(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """FedProx µ değerini her istemciye fit config üzerinden ilet (Li et al. 2020)."""
        pairs = super().configure_fit(server_round, parameters, client_manager)
        if not USE_FEDPROX:
            return pairs
        updated = []
        for proxy, fit_ins in pairs:
            new_config = {**fit_ins.config, "fedprox_mu": float(FEDPROX_MU)}
            updated.append((proxy, fl.common.FitIns(fit_ins.parameters, new_config)))
        logger.info(
            f"[Round {server_round}] FedProx µ={FEDPROX_MU} "
            f"→ {len(updated)} istemciye iletildi"
        )
        return updated


import threading

# ─── SIMPLE THREAD-BASED SCHEDULER ────────────────────────────────────────────

class RoundScheduler:
    def __init__(self):
        self.running = False
        self._thread = None

    def start(self):
        if self.running: return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("[FL] Thread-based scheduler başlatıldı.")

    def _loop(self):
        llm_time_str = os.getenv("FL_SCHEDULE_LLM", "23:00")
        embed_time_str = os.getenv("FL_SCHEDULE_EMBEDDING", "22:00")

        logger.info(f"[FL] Scheduler aktif — LLM: {llm_time_str}, Embedding: {embed_time_str}")

        while self.running:
            now = time.strftime("%H:%M")
            if now == llm_time_str:
                logger.info(f"[FL] Planlı LLM round saati geldi: {now}")
                start_llm_round()
                time.sleep(61) # Aynı dakika içinde tekrar tetiklenmesin
            elif now == embed_time_str:
                logger.info(f"[FL] Planlı Embedding round saati geldi: {now}")
                start_embedding_round()
                time.sleep(61)
            time.sleep(30)

scheduler = RoundScheduler()
app_admin = FastAPI(title="Flower Server Admin")

def get_connected_branch_count() -> int:
    import httpx
    count = 0
    urls = []

    # Environment variables
    b1_url = os.getenv("BRANCH1_RAG_URL")
    b2_url = os.getenv("BRANCH2_RAG_URL")

    if b1_url:
        urls.append(b1_url)
    else:
        urls.append(f"http://localhost:{os.getenv('BRANCH1_RAG_PORT', '8001')}")

    if b2_url:
        urls.append(b2_url)
    else:
        urls.append(f"http://localhost:{os.getenv('BRANCH2_RAG_PORT', '8002')}")

    # Şube RAG servisleri mTLS ister: kurum CA'sı + genel merkez istemci sertifikasıyla yokla
    import ssl
    verify = True
    ca = os.getenv("RAG_CA_CERT_PATH", "")
    cert, key = os.getenv("RAG_CLIENT_CERT", ""), os.getenv("RAG_CLIENT_KEY", "")
    try:
        verify = ssl.create_default_context(cafile=ca) if ca and os.path.exists(ca) else ssl.create_default_context()
        if cert and key and os.path.exists(cert) and os.path.exists(key):
            verify.load_cert_chain(cert, key)
    except Exception as e:
        logger.warning(f"[FL] mTLS bağlamı kurulamadı: {e}")

    for url in urls:
        try:
            clean_url = url.rstrip("/")
            # Use a quick 1-second timeout
            with httpx.Client(timeout=1.0, verify=verify, trust_env=False) as client:
                resp = client.get(f"{clean_url}/health")
                if resp.status_code == 200:
                    count += 1
        except Exception:
            pass

    return count

import threading as _threading
_ROUND_LOCK = _threading.Lock()


def run_fl_round(round_type: str = "llm"):
    """Farklı model tipleri için FL round başlatır (aynı anda tek tur)."""
    if not _ROUND_LOCK.acquire(blocking=False):
        logger.warning(f"[FL] {round_type.upper()} isteği yok sayıldı: devam eden bir federe tur var (istemci bekleniyor olabilir).")
        return
    logger.info(f"[FL] {round_type.upper()} round başlatılıyor...")
    try:
        run_fl_server(round_type=round_type)
    except Exception as e:
        logger.error(f"[FL] Round hatası: {e}")
    finally:
        _ROUND_LOCK.release()

def start_llm_round():
    connected = get_connected_branch_count()
    min_branches = int(os.getenv("FL_MIN_BRANCHES", "1"))
    if connected < min_branches:
        logger.warning(f"[FL] LLM round atlandı: {connected}/{min_branches} branch bağlı")
        return
    logger.info("[FL] LLM round tetiklendi")
    run_fl_round(round_type="llm")

def start_embedding_round():
    connected = get_connected_branch_count()
    min_branches = int(os.getenv("FL_MIN_BRANCHES", "1"))
    if connected < min_branches:
        logger.warning(f"[FL] Embedding round atlandı: {connected}/{min_branches} branch bağlı")
        return
    logger.info("[FL] Embedding round tetiklendi")
    run_fl_round(round_type="embedding")

# ─── ADMIN ENDPOINT'LER ────────────────────────────────────────────────────────

@app_admin.post("/admin/trigger/llm")
async def trigger_llm(req: Request):
    if req.client.host != "127.0.0.1":
        return {"error": "Unauthorized"}
    import threading
    threading.Thread(target=start_llm_round, daemon=True).start()
    return {"status": "triggered", "type": "llm"}

@app_admin.post("/admin/trigger/embedding")
async def trigger_embedding(req: Request):
    if req.client.host != "127.0.0.1":
        return {"error": "Unauthorized"}
    import threading
    threading.Thread(target=start_embedding_round, daemon=True).start()
    return {"status": "triggered", "type": "embedding"}

@app_admin.get("/admin/status")
async def status():
    return {
        "scheduler_running": scheduler.running,
        "connected_branches": get_connected_branch_count()
    }

def get_tls_config():
    ca_cert     = CERT_DIR / "ca.crt"
    server_cert = CERT_DIR / "server.crt"
    server_key  = CERT_DIR / "server.key"

    if not all([ca_cert.exists(), server_cert.exists(), server_key.exists()]):
        logger.warning("[WARN]  TLS sertifikaları eksik! Şifresiz bağlantı (SADECE TEST).")
        return None

    return (
        ca_cert.read_bytes(),
        server_cert.read_bytes(),
        server_key.read_bytes()
    )

def run_fl_server(round_type: str = "llm"):
    """Flower Server'ı başlatır ve round'ları yürütür."""
    import signal
    import threading

    orig_signal = signal.signal
    is_bg_thread = threading.current_thread() != threading.main_thread()
    if is_bg_thread:
        # Mock signal.signal to avoid ValueError in non-main thread
        def dummy_signal(sig, handler):
            logger.debug(f"Mocked signal.signal({sig}, {handler}) called in background thread")
            return None
        signal.signal = dummy_signal

    try:
        run_id = _init_mlflow(f"federated-learning-{round_type}")

        # Dinleme adresini doğrula
        bind_addr = SERVER_ADDRESS
        if ":" not in bind_addr: bind_addr = f"0.0.0.0:{bind_addr}"

        if round_type == "embedding":
            host, port_str = bind_addr.split(":")
            bind_addr = f"{host}:{int(port_str) + 2}"

        strategy = MosaicStrategy(
            mlflow_run_id                    = run_id,
            fraction_fit                     = FRACTION_FIT,
            fraction_evaluate                = FRACTION_EVAL,
            min_fit_clients                  = MIN_CLIENTS,
            min_evaluate_clients             = MIN_EVAL_CLIENTS,
            min_available_clients            = MIN_CLIENTS,
            fit_metrics_aggregation_fn       = fit_metrics_aggregation,
            evaluate_metrics_aggregation_fn  = evaluate_metrics_aggregation,
        )

        logger.info(
            f"Flower sunucu başlatılıyor ({round_type.upper()}) | addr={bind_addr} "
            f"rounds={NUM_ROUNDS} min_clients={MIN_CLIENTS}"
        )

        fl.server.start_server(
            server_address=bind_addr,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
        )

        logger.info(f"Tüm {round_type.upper()} federated round'lar tamamlandı.")

        if run_id:
            try:
                import mlflow
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_param("num_rounds",   NUM_ROUNDS)
                    mlflow.log_param("min_clients",  MIN_CLIENTS)
                    mlflow.log_param("use_fedprox",  USE_FEDPROX)
                    mlflow.log_param("fedprox_mu",   FEDPROX_MU)
            except Exception:
                pass
    except Exception as e:
        logger.exception(f"Flower Server kritik hata ile kapandı: {e}")
    finally:
        if is_bg_thread:
            signal.signal = orig_signal

def main():
    # Scheduler Kurulumu
    scheduler.start()

    # Admin API'yi ayrı bir thread'de başlat
    if os.getenv("FL_ADMIN_API", "true").lower() == "true":
        import threading
        def run_admin():
            admin_host = os.getenv("FL_ADMIN_HOST", "127.0.0.1")
            uvicorn.run(app_admin, host=admin_host, port=9081, log_level="warning")

        global _admin_thread_started
        if '_admin_thread_started' not in globals():
            threading.Thread(target=run_admin, daemon=True).start()
            logger.info(f"[FL] Admin API {os.getenv('FL_ADMIN_HOST', '127.0.0.1')}:9081 üzerinden başlatıldı.")
            _admin_thread_started = True

if __name__ == "__main__":
    DAEMON_MODE = os.getenv("FL_DAEMON", "true").lower() == "true"

    if DAEMON_MODE:
        logger.info("Daemon mode açık — Planlanan saatlerde veya admin tetiklendiğinde FL çalışacak.")
        main()
        try:
            while True:
                time.sleep(30)
        except KeyboardInterrupt:
            logger.info("\nSunucu durdurma komutu alındı (Ctrl+C).")
            sys.exit(0)
    else:
        # Normal mode: 1 seferlik çalıştır
        main()
        run_fl_server()
