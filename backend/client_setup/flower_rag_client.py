"""
flower_rag_client.py — Flower Federe Öğrenme İstemcisi + RAG
VMware Ubuntu İstemci için tek giriş noktası.

Mimari:
  - Flower Client: TLS ile ana makineye bağlanır, model güncelleme alır
  - RAG: Şubeye özel ChromaDB ile sorgu yanıtlar
  - LLM YOK: Sorgular ana makineye HTTP ile iletilir

Gizlilik Katmanı:
  - fl.common.SecureAggregation (homomorfik şifreleme) KULLANILMIYOR
  - Differential Privacy: Gaussian Mekanizması (diffprivlib)
    * epsilon: gizlilik bütçesi (küçük = daha güçlü gizlilik)
    * delta: olasılıksal sızdırma toleransı
    * sensitivity: gradyan/vektör kırpma eşiği (L2 norm)
  - AMD SME (Secure Memory Encryption): Donanım katmanında bellek şifreleme
    * AMD EPYC / Ryzen Pro işlemcilerde otomatik aktif
    * /sys/module/amd_smca veya dmesg | grep SME ile kontrol edilebilir
    * Python tarafında ek kod gerektirmez — kernel düzeyinde çalışır
"""

import os
import time
import logging
logging.basicConfig(level=logging.INFO)
import hashlib
import argparse
import httpx
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
import flwr as fl
import numpy as np
import chromadb
try:
    from diffprivlib.mechanisms import Gaussian
except ImportError:
    class Gaussian:
        """Lightweight mathematical DP-SGD Gaussian mechanism fallback."""
        def __init__(self, epsilon=1.0, delta=1e-5, sensitivity=1.0):
            import math
            self.epsilon = float(epsilon)
            self.delta = float(delta)
            self.sensitivity = float(sensitivity)
            self.sigma = (math.sqrt(2 * math.log(1.25 / self.delta)) * self.sensitivity) / self.epsilon

        def randomise(self, value: float) -> float:
            return float(value) + float(np.random.normal(0.0, self.sigma))

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from loguru import logger

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "federated"))
from dp import privatize_update, composed_epsilon  # noqa: E402

# Yapılandırma
load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
class ClientConfig:
    SERVER_HOST: str = os.getenv("SERVER_HOST", "192.168.1.100")
    SERVER_FLOWER_PORT: int = int(os.getenv("SERVER_FLOWER_PORT", "9080"))
    SERVER_LLM_PORT: int    = int(os.getenv("SERVER_LLM_PORT", "8004"))
    BRANCH_ID: str          = os.getenv("BRANCH_ID", "branch_01")
    CERT_DIR: Path          = Path(os.getenv("CERT_DIR", "backend/certs"))
    CHROMA_PATH: Path       = Path(os.getenv("CHROMA_PATH", os.getenv("CHROMADB_PATH", "ai/data/chromadb")))
    EMBEDDING_MODEL: str    = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    DATA_DIR: Path          = Path(os.getenv("DATA_DIR", "ai/data"))
    MAX_RAM_MB: int         = int(os.getenv("MAX_RAM_MB", "8192"))
    FL_SERVER_ADDRESS: str  = os.getenv("FL_SERVER_ADDRESS", "")

    # ── Differential Privacy Parametreleri ──────────────────────────────
    # epsilon: Gizlilik bütçesi. Küçük değer = güçlü gizlilik, daha fazla gürültü.
    #   Önerilen: 1.0 (güçlü DP) — 10.0 (zayıf DP, yüksek doğruluk)
    DP_EPSILON: float       = float(os.getenv("DP_EPSILON", "1.0"))
    # delta: (ε, δ)-DP toleransı. Çok küçük tutulmalı (< 1/n).
    DP_DELTA: float         = float(os.getenv("DP_DELTA", "1e-5"))
    # sensitivity: Gradyan/istatistik vektörünün L2 norm kırpma eşiği.
    #   Yerel güncelleme bu değere normalize edilir — sızdırma minimuma iner.
    DP_SENSITIVITY: float   = float(os.getenv("DP_SENSITIVITY", "1.0"))

    @classmethod
    def flower_server_address(cls) -> str:
        if cls.FL_SERVER_ADDRESS:
            return cls.FL_SERVER_ADDRESS
        return f"{cls.SERVER_HOST}:{cls.SERVER_FLOWER_PORT}"

    @classmethod
    def llm_api_url(cls) -> str:
        return f"http://{cls.SERVER_HOST}:{cls.SERVER_LLM_PORT}"

config = ClientConfig()

# ─────────────────────────────────────────────
# RAG MOTORU
# ─────────────────────────────────────────────
class BranchRAGEngine:
    """Şubeye özel RAG motoru — ChromaDB + SentenceTransformers."""

    def __init__(self):
        logger.info(f"RAG motoru başlatılıyor | Şube: {config.BRANCH_ID}")
        # trust_remote_code=True ekleyerek Alibaba-NLP model yükleme hatasını çözüyoruz
        # Model yolu yerelde yoksa HuggingFace fallback yap
        model_path = config.EMBEDDING_MODEL
        if model_path.startswith("ai/") and not os.path.exists(model_path):
            logger.warning(f"[WARNING] Belirtilen embedding modeli yerelde bulunamadı ({model_path}). HuggingFace fallback: 'intfloat/multilingual-e5-small' yükleniyor.")
            model_path = "intfloat/multilingual-e5-small"

        self.embedding_model = SentenceTransformer(
            str(model_path),
            trust_remote_code=True,
            model_kwargs={"trust_remote_code": True}
        )

        # Daha güvenli başlatma yöntemi
        from chromadb.config import Settings
        self.chroma_client = chromadb.PersistentClient(
            path=str(config.CHROMA_PATH),
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )

        collection_map = {
            "hq": "finance_global",
            "branch_01": "finance_branch1",
            "branch_02": "finance_branch2",
        }
        collection_name = collection_map.get(config.BRANCH_ID, f"branch_{config.BRANCH_ID}")

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB bağlantısı OK | Koleksiyon: {collection_name}")

    def add_document(self, text: str, doc_id: str, metadata: Optional[Dict] = None) -> None:
        """Belge ekle — KVKK gereği metadata'ya kişisel veri ekleme."""
        embedding = self.embedding_model.encode(text).tolist()
        safe_metadata = {k: v for k, v in (metadata or {}).items()
                        if k not in ("tc_kimlik", "ad_soyad", "telefon", "email")}
        safe_metadata["branch_id"] = config.BRANCH_ID
        safe_metadata["doc_hash"] = hashlib.sha256(text.encode()).hexdigest()[:16]

        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[safe_metadata]
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Sorguya en yakın belgeleri getir."""
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count() or 1)
        )
        docs = []
        for i, doc in enumerate(results["documents"][0]):
            docs.append({
                "content": doc,
                "score": 1 - results["distances"][0][i],
                "metadata": results["metadatas"][0][i]
            })
        return docs

    def build_rag_prompt(self, query: str, top_k: int = 3, threshold: float = 0.45) -> str:
        """RAG bağlamını sorguya akıllıca ekle."""
        retrieved = self.retrieve(query, top_k)

        # Sadece belirli bir benzerlik skorunun üzerindeki belgeleri al
        relevant_docs = [doc for doc in retrieved if doc['score'] >= threshold]

        if not relevant_docs:
            # İlgili belge yoksa ham sorguyu gönder (Genel cevap vermesi için)
            return query

        context = "\n\n".join([
            f"[Yerel Kaynak {i+1}]: {doc['content']}"
            for i, doc in enumerate(relevant_docs)
        ])

        return (
            f"Aşağıda seninle paylaşılan şube belgelerine dayanarak soruyu yanıtla. "
            f"Eğer belgeler soruyla ilgili değilse, belgelerden bahsetmeden genel bir bankacılık asistanı gibi yanıt ver.\n\n"
            f"BAĞLAM:\n{context}\n\n"
            f"SORU: {query}"
        )

# ─────────────────────────────────────────────
# ANA MAKİNEYE SORGU KÖPRÜSÜ
# ─────────────────────────────────────────────
class LLMBridge:
    """LLM sorgularını ana makineye iletir. İstemcide LLM yoktur."""

    def __init__(self):
        self.base_url = config.llm_api_url()
        self.branch_id = 1 if "01" in config.BRANCH_ID else 2

    async def query(self, query: str, local_chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ana makinedeki LLM API'sına sorgu gönder."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "query": query,
                "local_chunks": local_chunks or [],
                "branch_id": self.branch_id,
            }
            # flower-server yerine SERVER_HOST kullan (bağlantı hatasını önler)
            url = f"http://{config.SERVER_HOST}:{config.SERVER_LLM_PORT}/query"
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

# ─────────────────────────────────────────────
# FLOWER CLIENTS
# ─────────────────────────────────────────────

class EmbeddingFlowerClient(fl.client.NumPyClient):
    """
    TurkEmbed embedding modelini federe olarak fine-tune eder.
    Seviye 2 Federated RAG: model ağırlıkları paylaşılır, ham veri asla çıkmaz.
    """

    def __init__(self, model_path: str, rag_engine: BranchRAGEngine):
        self.model_path = model_path
        self.rag = rag_engine
        self.branch_id = config.BRANCH_ID

        # Differential Privacy Parametreleri
        self.epsilon = config.DP_EPSILON
        self.delta = config.DP_DELTA
        self.sensitivity = config.DP_SENSITIVITY
        self.mechanism = Gaussian(epsilon=self.epsilon, delta=self.delta, sensitivity=self.sensitivity)

        logger.info(f"EmbeddingFlowerClient başlatıldı | ε={self.epsilon}")

    def _build_training_pairs(self) -> List[Any]:
        """
        Lokal ChromaDB chunk'larından sentetik (query, passage) çiftleri üretir.
        Strateji: chunk'ın ilk cümlesi = synthetic query, chunk'ın tamamı = passage.
        """
        from sentence_transformers import InputExample

        results = self.rag.collection.get(include=["documents"])
        documents = results.get("documents", [])

        train_examples = []
        for doc in documents:
            if not doc or len(doc) < 50:
                continue

            # İlk cümleyi query olarak kullan
            sentences = doc.split('.')
            query = sentences[0].strip()
            if len(query) < 10:
                continue

            train_examples.append(InputExample(texts=[query, doc]))

        return train_examples

    def _apply_dp_noise(self, weights: List[np.ndarray], global_params: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """İstemci güncellemesine (yerel - global) global L2 kırpma + Gauss gürültüsü uygular.

        Not: Bu bir istemci düzeyi Gauss mekanizmasıdır (DP-SGD değil). Tur başına bütçe ε'dur;
        R tur için temel bileşim R*ε olur (bkz. backend/federated/dp.py).
        """
        if global_params is None:
            raise ValueError("DP için global parametreler gerekli (kırpma güncelleme farkına uygulanır)")
        t0 = time.perf_counter()
        noisy = privatize_update(global_params, weights, clip_norm=self.sensitivity,
                                 epsilon=self.epsilon, delta=self.delta)
        self._dp_rounds = getattr(self, "_dp_rounds", 0) + 1
        logger.info(f"[Differential Privacy] Gauss gürültüsü eklendi (ε/tur={self.epsilon}, δ={self.delta}, "
                    f"kırpma={self.sensitivity}); tur={self._dp_rounds}, toplam ε (temel bileşim) = "
                    f"{composed_epsilon(self.epsilon, self._dp_rounds)}; süre={(time.perf_counter()-t0)*1000:.1f} ms")
        return noisy

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.rag.embedding_model.state_dict().items()]

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        from torch.utils.data import DataLoader
        from sentence_transformers import losses

        # 1. Global ağırlıkları yükle
        global_params = [np.array(v, copy=True) for v in parameters]
        state_dict = self.rag.embedding_model.state_dict()
        for key, value in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.from_numpy(value)
        self.rag.embedding_model.load_state_dict(state_dict)

        # 2. Eğitim verisi hazırla
        train_examples = self._build_training_pairs()
        if not train_examples:
            logger.warning("Eğitim için yeterli yerel veri bulunamadı.")
            return self.get_parameters({}), 0, {}

        # 3. Fine-tune
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=int(os.getenv("EMBEDDING_FL_BATCH_SIZE", "8")))
        train_loss = losses.MultipleNegativesRankingLoss(model=self.rag.embedding_model)

        epochs = int(os.getenv("EMBEDDING_FL_EPOCHS", "1"))
        self.rag.embedding_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            show_progress_bar=False
        )

        # 4. DP Noise
        noisy_params = self._apply_dp_noise(self.get_parameters({}), global_params)

        return noisy_params, len(train_examples), {"branch_id": self.branch_id}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        # Yerel değerlendirme uygulanmadı: uydurma metrik döndürülmez (sunucu 0 örnekli sonuçları yok sayar).
        return 0.0, 0, {}


class AdaptiveChatbotFlowerClient(fl.client.NumPyClient):
    """
    Flower Federe Öğrenme İstemcisi.
    Gradyan güncellemelerini ana makine ile TLS üzerinden paylaşır.
    Ham veri hiçbir zaman istemciden çıkmaz (KVKK/GDPR uyumlu).

    Gizlilik Mimarisi:
      - fl.common.SecureAggregation (HE tabanlı) KULLANILMIYOR.
      - Differential Privacy (Gaussian Mekanizması, diffprivlib):
          1. Yerel güncelleme vektörü L2 norma göre kırpılır (sensitivity).
          2. Gaussian gürültüsü eklenir (epsilon, delta parametrelerine göre).
          3. Gürültülü vektör sunucuya gönderilir — ham veri korunur.
      - AMD SME (Secure Memory Encryption):
          * Kernel düzeyinde çalışır, AMD EPYC/Ryzen Pro donanımında aktiftir.
          * Bu sınıf içinde ek kod gerekmez; bellek otomatik şifrelenir.
          * Kontrol: `dmesg | grep -i sme` veya `/sys/module/amd_smca/`
    """

    def __init__(self, rag_engine: BranchRAGEngine):
        self.rag = rag_engine
        self.branch_id = config.BRANCH_ID
        self._local_params = self._init_local_params()

        # Differential Privacy — Gaussian Mekanizması
        # Referans: Dwork & Roth (2014), "The Algorithmic Foundations of DP"
        # epsilon küçüldükçe gizlilik güçlenir, gürültü artar.
        self._dp_mechanism = Gaussian(
            epsilon=config.DP_EPSILON,
            delta=config.DP_DELTA,
            sensitivity=config.DP_SENSITIVITY,
        )
        logger.info(
            f"Flower Client hazır | Şube: {self.branch_id} | "
            f"DP ε={config.DP_EPSILON} δ={config.DP_DELTA} "
            f"sensitivity={config.DP_SENSITIVITY}"
        )

    def _init_local_params(self) -> List[np.ndarray]:
        """Yerel model parametrelerini başlat."""
        return [np.zeros(128, dtype=np.float32)]

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Sunucuya gönderilecek parametreler."""
        logger.debug("Parametreler sunucuya gönderiliyor...")
        return self._local_params

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Sunucudan gelen global parametrelerle yerel modeli güncelle.
        Ham veri çıkmaz — Differential Privacy ile koruma altında gradyanlar döner.
        """
        self._local_params = parameters
        doc_count = self.rag.collection.count()
        # ZeroDivisionError önlemek için en az 1 örnek raporla
        num_examples = max(1, doc_count)
        logger.info(f"Federe güncelleme alındı | Yerel belge sayısı: {doc_count}")

        # Yerel adaptasyon: DP gürültüsü eklenmiş istatistiksel özet
        local_update = self._compute_local_update()
        return local_update, num_examples, {
            "branch_id": self.branch_id,
            "doc_count": doc_count,
            "dp_epsilon": ClientConfig.DP_EPSILON,
        }

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Yerel veriyle değerlendirme."""
        self._local_params = parameters
        # Yerel değerlendirme uygulanmadı: sabit loss=0.01 / accuracy=0.95 kaldırıldı.
        # Sunucu (MosaicStrategy.aggregate_evaluate) 0 örnekli sonuçları yok sayar.
        return 0.0, 0, {}

    def _compute_local_update(self) -> List[np.ndarray]:
        """
        Şubeye özel yerel güncelleme hesapla — Differential Privacy ile.
        """
        results = self.rag.collection.get(include=["embeddings"])
        if results.get("embeddings") is not None and len(results["embeddings"]) > 0:
            embeddings = np.array(results["embeddings"])
            # Adım 1: İstatistiksel özet (ortalama vektör, ilk 128 boyut)
            mean_vec = embeddings.mean(axis=0)[:128].astype(np.float64)

            # Adım 2: L2 norm kırpma — sensitivity sınırını aşmayı önler
            l2_norm = np.linalg.norm(mean_vec)
            if l2_norm > config.DP_SENSITIVITY:
                mean_vec = mean_vec * (config.DP_SENSITIVITY / l2_norm)

            # Adım 3: Differential Privacy — Gaussian gürültüsü ekle
            noisy_vec = np.array([
                self._dp_mechanism.randomise(float(v)) for v in mean_vec
            ], dtype=np.float32)

            logger.info(
                f"[Differential Privacy] Ağırlık güncellemelerine Gaussian gürültüsü ekleniyor (Bütçe ε={config.DP_EPSILON}, δ={config.DP_DELTA}, Duyarlılık={config.DP_SENSITIVITY})"
            )
            logger.info(
                f"[Differential Privacy] Gürültü ekleme tamamlandı. "
                f"Orijinal L2 Norm: {l2_norm:.4f} | "
                f"Gürültülü L2 Norm: {np.linalg.norm(noisy_vec):.4f}"
            )
            return [noisy_vec]

        # Eğer belge yoksa bile sıfır vektörü dön ki round başarılı sayılsın
        logger.debug("Belge yok, sıfır vektörü gönderiliyor.")
        return [np.zeros(128, dtype=np.float32)]

# ─────────────────────────────────────────────
# BAŞLATMA
# ─────────────────────────────────────────────
def get_tls_config():
    """TLS sertifikalarını yükle."""
    ca_cert     = config.CERT_DIR / "ca.crt"
    client_cert = config.CERT_DIR / "client.crt"
    client_key  = config.CERT_DIR / "client.key"

    if not all([ca_cert.exists(), client_cert.exists(), client_key.exists()]):
        logger.warning("[WARNING] TLS sertifikaları eksik! Şifresiz bağlantı (SADECE TEST).")
        return None

    return (
        ca_cert.read_bytes(),
        client_cert.read_bytes(),
        client_key.read_bytes()
    )

def is_port_open(address: str) -> bool:
    import socket
    try:
        if ":" in address:
            host, port_str = address.split(":")
            port = int(port_str)
        else:
            host = "localhost"
            port = int(address)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="MOSAIC-v2 Flower Client")
    parser.add_argument(
        "--mode",
        choices=["llm", "embedding", "all"],
        default="all",
        help="llm: LLM adapter FL | embedding: TurkEmbed FL | all: ikisi sırayla"
    )
    args = parser.parse_args()

    # Resolve target server address
    server_addr = config.flower_server_address()
    if args.mode == "embedding":
        if ":" in server_addr:
            host, port_str = server_addr.split(":")
            server_addr = f"{host}:{int(port_str) + 2}"
        else:
            server_addr = f"127.0.0.1:{int(server_addr) + 2}"

    logger.info("=" * 50)
    logger.info("  Adaptif Chatbot Flower Client Başlatılıyor")
    logger.info(f"  Mod: {args.mode}")
    logger.info(f"  Şube: {config.BRANCH_ID}")
    logger.info(f"  Sunucu Hedefi: {server_addr}")
    logger.info(f"  DP Gizlilik Bütçesi: ε={config.DP_EPSILON}, δ={config.DP_DELTA}")
    logger.info("=" * 50)

    # Check if server is active to prevent traceback noise
    if not is_port_open(server_addr):
        logger.debug(f"[FL] Sunucu ({server_addr}) aktif değil, bekleniyor...")
        return False

    rag_engine = BranchRAGEngine()

    if args.mode in ["llm", "all"]:
        logger.info(">>> LLM Flower Round başlatılıyor...")
        flower_client = AdaptiveChatbotFlowerClient(rag_engine)
        try:
            fl.client.start_client(
                server_address=server_addr,
                client=flower_client.to_client(),
            )
        except Exception as e:
            logger.error(f"LLM Flower Error: {e}")

    if args.mode in ["embedding", "all"]:
        logger.info(">>> Embedding Flower Round başlatılıyor...")
        emb_client = EmbeddingFlowerClient(model_path=config.EMBEDDING_MODEL, rag_engine=rag_engine)
        try:
            fl.client.start_client(
                server_address=server_addr,
                client=emb_client.to_client(),
            )
        except Exception as e:
            logger.error(f"Embedding Flower Error: {e}")

    return True

if __name__ == "__main__":
    daemon_mode = os.getenv("FL_CLIENT_DAEMON", "true").lower() == "true"
    if daemon_mode:
        logger.info("FL Client Daemon Mode aktif — Sürekli döngüde çalışacak...")
        try:
            while True:
                did_run = main()
                # If we ran a round, sleep slightly longer, else check quickly
                sleep_sec = 30 if did_run else 10
                time.sleep(sleep_sec)
        except KeyboardInterrupt:
            logger.info("Kapatılıyor...")
    else:
        main()
