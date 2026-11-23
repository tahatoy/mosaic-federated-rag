"""
Federe öğrenme denetim verisi (arayüzdeki "Federe Öğrenme & KVKK Denetim" penceresi).

Bu modül YALNIZCA depodaki ölçülmüş deney çıktılarını ve tezde raporlanan değerleri döndürür;
sabit kodlanmış/uydurma kayıp eğrisi, "FL öncesi/sonrası" skoru veya bağlı istemci sayısı üretmez.
Kaynaklar:
  * ai/experiments/experiment_log.json  -> fazlar.faz5 (FedAvg / FedProx yönlendirici simülasyonu, Çizelge 4.3)
  * ai/experiments/results/h2_dp_analysis_v2.json -> istemci Gauss mekanizmasının ağırlıklara etkisi (Bölüm 4.7.2)
  * Ablasyon (Çizelge 4.11): federe FedProx yönlendirici 0.8509 / merkezi 0.9004 Macro F1
"""
import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ai" / "experiments"


def _load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class FederatedProofEngine:
    """Ölçülmüş federe öğrenme ve diferansiyel gizlilik bulgularını tek pakette sunar."""

    def get_full_audit_payload(self) -> Dict[str, Any]:
        faz5 = _load(RESULTS / "experiment_log.json").get("fazlar", {}).get("faz5", {})
        bul = faz5.get("bulgular", {})
        dp = _load(RESULTS / "results" / "h2_dp_analysis_v2.json")

        def strat(name):
            b = bul.get(name, {})
            return {"accuracy": b.get("final_accuracy"), "f1_macro": b.get("final_f1_macro"),
                    "elapsed_s": b.get("elapsed_s")}

        return {
            "status": "ready",
            "source_note": "Değerler depodaki deney çıktılarından okunur (experiment_log.json faz5, h2_dp_analysis_v2.json).",
            "router_fl": {
                "title": "BERT yönlendirici — FedAvg / FedProx (Çizelge 4.3)",
                "setup": "2 heterojen şube, 5 tur, tur başına 2 yerel dönem, μ = "
                         f"{faz5.get('konfig', {}).get('fedprox_mu', 0.01)}; Flower değil, sıfırdan yazılmış simülasyon; "
                         "yönlendirici eğitiminde DP gürültüsü yok, yalnızca gradyan kırpma (L2 ≤ 1.0).",
                "FedAvg": strat("FedAvg"),
                "FedProx": strat("FedProx"),
            },
            "router_ablation": {
                "title": "Tam sistem ablasyonu (Çizelge 4.11)",
                "federated_fedprox_f1": 0.8509,
                "central_bert_f1": 0.9004,
                "note": "0.0495 Macro F1 farkı (göreli %5.5) verinin merkezde toplanmamasının başarım maliyetidir.",
            },
            "dp_embedding": {
                "title": "Gömme istemcisi Gauss mekanizması (Denklem 3.4, Bölüm 4.7.2)",
                "epsilon": dp.get("epsilon"), "delta": dp.get("delta"), "C": dp.get("C"), "sigma": dp.get("sigma"),
                "param_count": dp.get("parametre_sayisi"),
                "weight_l2": dp.get("agirlik_global_L2_normu"),
                "noise_l2": dp.get("beklenen_gurultu_global_L2_normu"),
                "cosine_global": dp.get("global_kosinus_w_vs_gurultulu"),
                "cosine_tensor_median": dp.get("tensor_kosinus_medyan"),
                "note": "Bu ayarlarla eklenen gürültü ağırlıkları kullanılamaz hale getirmektedir; "
                        "gizlilik-fayda dengesinin ayarlanması gerekir.",
            },
            "layers": [
                {"name": "Katman A — Dil modeli LoRA federasyonu",
                 "status": "Tasarım düzeyinde; Gemma-3-4B LoRA adaptörü bu çalışmada merkezi olarak eğitildi."},
                {"name": "Katman B — Gömme modeli federasyonu",
                 "status": "Flower istemcisinde güncelleme kırpma + Gauss gürültüsü uygulanır (σ yukarıda)."},
            ],
            "channel": {
                "flower": "gRPC (9080); karşılıklı TLS yapılandırıldı, sertifikasız istemcinin reddi ayrıca test edilmedi.",
                "rag": "Çapraz şube RAG kanalı mTLS (TLS 1.3); sertifikasız ve düz HTTP bağlantılar reddedildi.",
            },
        }


# Global Singleton
proof_engine = FederatedProofEngine()
