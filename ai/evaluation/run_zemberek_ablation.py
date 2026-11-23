"""
H4 — Zemberek Tokenizasyon Offline Ablasyon (Faz 12)
=====================================================
Servis gerektirmeden Zemberek morfolojik lemmatizasyonunun
retrieval kalitesine etkisini ölçer.

Yöntem:
  Her (query, passage) çifti için:
    baseline : cosine_sim(embed("query: "+ raw_query),    embed("passage: "+ passage))
    zemberek : cosine_sim(embed("query: "+ lemma(query)), embed("passage: "+ passage))
  Δ = zemberek_sim - baseline_sim  → pozitif = Zemberek yardımcı oldu

  Ayrıca OOV analizi: HybridTokenizer.count_oov() ile UNK token oranı karşılaştırması.

Çıktı:
  experiments/ablation/zemberek_ablation_results.json
  experiments/ablation/ablation_table.md  (H4 satırı eklenir)

Kullanım:
  python evaluation/run_zemberek_ablation.py
  python evaluation/run_zemberek_ablation.py --model intfloat/multilingual-e5-small
  python evaluation/run_zemberek_ablation.py --data data/test/ablation_test.jsonl
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
LOCAL_MODEL_PATHS = [
    str(PROJECT_ROOT / "models" / "embeddings" / "turkembed_finetuned"),
    str(PROJECT_ROOT / "models" / "embeddings" / "e5small_finetuned"),
]


# ─── Sabit Test QA Setı ───────────────────────────────────────────────────────
# Servis gerektirmeden çalışan 15 Türkçe finans (query, passage) çifti

FALLBACK_QA_PAIRS = [
    {
        "query":   "Faiz oranı nedir?",
        "passage": "Faiz oranı, borç alınan para için ödenen yıllık ücretin anaparaya oranıdır. Merkez bankası politika faizini belirleyerek piyasa faizlerini etkiler.",
        "label":   1,
    },
    {
        "query":   "Vadeli mevduat hesabı nasıl açılır?",
        "passage": "Vadeli mevduat, belirli bir vade süresi için bankaya yatırılan para karşılığında faiz kazandıran hesaptır. Vade sonuna kadar çekilemez.",
        "label":   1,
    },
    {
        "query":   "DCF analizi nasıl yapılır?",
        "passage": "Discounted Cash Flow (DCF) analizinde gelecekteki nakit akışları iskonto oranıyla bugünkü değerine indirgenerek şirket değeri hesaplanır.",
        "label":   2,
    },
    {
        "query":   "Konut kredisi faiz oranları ne kadar?",
        "passage": "Konut kredisi faiz oranları bankadan bankaya değişmekte olup Ziraat Bankası %1.89, Garanti BBVA %1.95 oranında 120 ay vadeye kadar kredi sunmaktadır.",
        "label":   1,
    },
    {
        "query":   "Portföy çeşitlendirme stratejisi nedir?",
        "passage": "Portföy çeşitlendirme, farklı varlık sınıflarına (hisse senedi, tahvil, emtia, döviz) yatırım yaparak sistematik olmayan riski minimize etme stratejisidir.",
        "label":   2,
    },
    {
        "query":   "BIST 100 endeksi ne anlama gelir?",
        "passage": "BIST 100 endeksi, Borsa İstanbul'da işlem gören en büyük 100 şirketin piyasa değeri ağırlıklı performansını gösteren temel Türk hisse senedi endeksidir.",
        "label":   1,
    },
    {
        "query":   "Eurobond ile devlet tahvili arasındaki fark nedir?",
        "passage": "Eurobond, Türkiye'nin yabancı para biriminde (dolar/euro) ihraç ettiği borçlanma senedidir. Devlet tahvili ise Türk Lirası cinsinden iç borçlanma aracıdır.",
        "label":   2,
    },
    {
        "query":   "Enflasyon oranı nereye etki eder?",
        "passage": "Enflasyon; satın alma gücünü düşürür, nominal faiz oranlarını yükseltir, döviz kurunu zayıflatır ve sabit getirili varlıkların reel getirisini olumsuz etkiler.",
        "label":   1,
    },
    {
        "query":   "Mevduat faiz karşılaştırması nasıl yapılır?",
        "passage": "Mevduat faiz karşılaştırmasında vade süresi, brüt/net faiz oranı, stopaj kesintisi (%15) ve efektif yıllık getiri birlikte değerlendirilmelidir.",
        "label":   1,
    },
    {
        "query":   "Bilanço analizi hangi kalemleri içerir?",
        "passage": "Bilanço analizinde dönen varlıklar, duran varlıklar, kısa vadeli yükümlülükler, uzun vadeli borçlar ve özkaynak kalemleri incelenerek likidite ve kârlılık oranları hesaplanır.",
        "label":   2,
    },
    {
        "query":   "Repo işlemi nasıl çalışır?",
        "passage": "Repo (geri satın alma anlaşması), kısa vadeli fon sağlamak amacıyla menkul kıymetlerin belirli tarihte geri alım taahhüdüyle satılmasıdır. Merkez bankası repo ile likiditeyi yönetir.",
        "label":   1,
    },
    {
        "query":   "Döviz kuru nasıl belirlenir?",
        "passage": "Döviz kuru; arz ve talep dengesi, merkez bankası müdahaleleri, faiz farkı, enflasyon beklentisi ve cari denge tarafından belirlenir. Türkiye'de dalgalı kur rejimi uygulanmaktadır.",
        "label":   1,
    },
    {
        "query":   "Hisse senedi değerleme yöntemleri nelerdir?",
        "passage": "Hisse senedi değerleme yöntemleri; DCF (indirgenmiş nakit akışı), F/K oranı (fiyat/kazanç), FD/FAVÖK çarpanı ve net varlık değeri yöntemlerini kapsamaktadır.",
        "label":   2,
    },
    {
        "query":   "Kredi risk analizi nasıl yapılır?",
        "passage": "Kredi risk analizi; borçlunun finansal tablolarındaki borç/özkaynak oranı, nakit akış yeterliliği, teminat kalitesi ve sektör riski değerlendirilerek gerçekleştirilir.",
        "label":   2,
    },
    {
        "query":   "Altın yatırımı enflasyona karşı koruma sağlar mı?",
        "passage": "Altın, tarihsel olarak enflasyona ve döviz değer kayıplarına karşı koruyucu özelliğiyle bilinen emtia varlığıdır. Portföyde %5-10 oranında tutulması önerilmektedir.",
        "label":   1,
    },
]


# ─── Yardımcı fonksiyonlar ────────────────────────────────────────────────────

def _load_qa_pairs(data_path: str | None) -> list[dict]:
    """Test veri setini yükle. Yoksa hardcode fallback kullan."""
    if data_path:
        p = Path(data_path)
        if p.exists():
            rows = []
            try:
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        if "query" in item and "passage" in item:
                            rows.append(item)
                if rows:
                    logger.info(f"QA çiftleri yüklendi: {p} ({len(rows)} örnek)")
                    return rows
            except Exception as e:
                logger.warning(f"Veri dosyası okunamadı: {e} — fallback kullanılıyor")
    logger.info(f"Fallback QA seti kullanılıyor ({len(FALLBACK_QA_PAIRS)} çift)")
    return FALLBACK_QA_PAIRS


def _load_embed_model(model_name: str):
    """Embedding modelini yerel önce, sonra HF'den yukle."""
    from sentence_transformers import SentenceTransformer
    import torch

    # Önce yerel kontrol
    for local in LOCAL_MODEL_PATHS:
        if Path(local).exists():
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                m = SentenceTransformer(local, device=device)
                logger.info(f"Yerel embedding modeli yüklendi: {local} ({device})")
                return m
            except Exception as e:
                logger.debug(f"Yerel model yüklenemedi: {local} — {e}")

    # HuggingFace'den yükle
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"HuggingFace'den yükleniyor: {model_name} ({device})")
    return SentenceTransformer(model_name, device=device)


def _load_tokenizers():
    """HybridTokenizer çiftini yükle. Zemberek yoksa SystemExit."""
    try:
        from tokenizer.hybrid_tokenizer import HybridTokenizer
        logger.info("HybridTokenizer (BERT-only baseline) yükleniyor...")
        # NOT: tokenizer.hybrid_tokenizer.HybridTokenizer parametreleri farklı olabilir.
        # Eğer bpe_model_path gerekiyorsa burada ayarlanmalı.
        # Varsayılan olarak BertHybridTokenizer benzeri davranış için use_zemberek kullanılır.
        tok_baseline = HybridTokenizer(bpe_model_path=None, use_zemberek=False)
        logger.info("HybridTokenizer (Zemberek+BERT) yükleniyor — ilk init yavaş olabilir...")
        tok_zemberek = HybridTokenizer(bpe_model_path=None, use_zemberek=True)
        logger.info("Her iki tokenizer yüklendi.")
        return tok_baseline, tok_zemberek
    except ImportError as e:
        logger.error(f"HybridTokenizer import hatası: {e}")
        logger.error("Çözüm: pip install zemberek-python veya JAVA_HOME ayarla")
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Tokenizer başlatma hatası: {e}")
        if "TurkishMorphology" in str(e) or "java" in str(e).lower():
            logger.error(
                "Zemberek JVM başlatılamadı.\n"
                "  1. Java kurulu mu? java -version\n"
                "  2. JAVA_HOME ayarlı mı?\n"
                "  3. zemberek-python kurulu mu? pip install zemberek-python"
            )
        raise SystemExit(1)


# ─── Ablasyon çekirdeği ────────────────────────────────────────────────────────

def run_ablation(
    qa_pairs: list[dict],
    embed_model,
    tok_baseline,
    tok_zemberek,
) -> dict:
    """
    Her (query, passage) çifti için Zemberek vs baseline cosine similarity karşılaştırması.

    tok_baseline  : HybridTokenizer(use_lemma=False)  → BERT-only
    tok_zemberek  : HybridTokenizer(use_lemma=True)   → Zemberek+BERT
    """
    per_sample = []

    for i, item in enumerate(qa_pairs, 1):
        query   = item["query"].strip()
        passage = item["passage"].strip()
        label   = item.get("label", 1)

        # Zemberek lemmatizasyonu
        try:
            lemma_query = tok_zemberek.lemmatize_text(query)
        except Exception as e:
            logger.warning(f"[{i}] Lemmatizasyon hatası: {e} — ham sorgu kullanılıyor")
            lemma_query = query

        # Prefix ekle (e5 dual-encoder zorunluluğu)
        raw_q    = "query: "   + query
        zem_q    = "query: "   + lemma_query
        pas_text = "passage: " + passage

        # Toplu encode (3 vektör)
        vectors = embed_model.encode(
            [raw_q, zem_q, pas_text],
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 norm → cosine = dot product
            show_progress_bar=False,
        )
        emb_raw, emb_zem, emb_pas = vectors[0], vectors[1], vectors[2]

        sim_baseline = float(np.dot(emb_raw, emb_pas))
        sim_zemberek = float(np.dot(emb_zem, emb_pas))
        delta        = sim_zemberek - sim_baseline

        # OOV analizi
        oov_base = tok_baseline.count_oov(query)
        oov_zem  = tok_zemberek.count_oov(lemma_query)

        per_sample.append({
            "query":        query,
            "lemma_query":  lemma_query,
            "passage":      passage[:100] + ("..." if len(passage) > 100 else ""),
            "label":        label,
            "sim_baseline": round(sim_baseline, 4),
            "sim_zemberek": round(sim_zemberek, 4),
            "delta":        round(delta, 4),
            "improved":     delta > 0.0,
            "oov_rate_raw": round(oov_base["oov_rate"], 4),
            "oov_rate_zem": round(oov_zem["oov_rate"], 4),
            "oov_delta":    round(oov_zem["oov_rate"] - oov_base["oov_rate"], 4),
        })
        logger.debug(
            f"[{i:2d}/{len(qa_pairs)}] Δ={delta:+.4f}  "
            f"base={sim_baseline:.4f}  zem={sim_zemberek:.4f}  "
            f"OOV: {oov_base['oov_rate']:.3f}→{oov_zem['oov_rate']:.3f}"
        )

    # Toplam metrikler
    n = len(per_sample)
    mean_sim_base = float(np.mean([r["sim_baseline"] for r in per_sample]))
    mean_sim_zem  = float(np.mean([r["sim_zemberek"] for r in per_sample]))
    mean_delta    = float(np.mean([r["delta"]        for r in per_sample]))
    mean_oov_raw  = float(np.mean([r["oov_rate_raw"] for r in per_sample]))
    mean_oov_zem  = float(np.mean([r["oov_rate_zem"] for r in per_sample]))
    n_improved    = sum(1 for r in per_sample if r["improved"])

    # Sınıf bazlı analiz
    by_label: dict[int, list] = {0: [], 1: [], 2: []}
    for r in per_sample:
        by_label[r["label"]].append(r["delta"])
    per_label = {
        ["chitchat", "simple", "complex"][lab]: {
            "n": len(deltas),
            "mean_delta": round(float(np.mean(deltas)), 4) if deltas else None,
        }
        for lab, deltas in by_label.items() if deltas
    }

    return {
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
        "n_samples":        n,
        "embedding_model":  DEFAULT_MODEL,
        "mean_sim_baseline": round(mean_sim_base, 4),
        "mean_sim_zemberek": round(mean_sim_zem, 4),
        "mean_delta":        round(mean_delta, 4),
        "mean_oov_rate_raw": round(mean_oov_raw, 4),
        "mean_oov_rate_zem": round(mean_oov_zem, 4),
        "oov_improvement":   round(mean_oov_raw - mean_oov_zem, 4),  # pozitif = Zemberek daha az OOV
        "n_improved":        n_improved,
        "pct_improved":      round(n_improved / n * 100, 1),
        "per_label":         per_label,
        "per_sample":        per_sample,
    }


# ─── Ablation table güncelleme ─────────────────────────────────────────────────

def _update_ablation_table(results: dict):
    table_path = OUTPUT_DIR / "ablation_table.md"
    d = results["mean_delta"]
    verdict = "Zemberek retrieval kalitesini **artırıyor** [OK]" if d > 0 else "Zemberek net katkı gösteremiyor [WARN]"

    h4_section = (
        f"\n\n---\n\n"
        f"## H4 — Zemberek Tokenizasyon Ablasyonu\n\n"
        f"Tarih: {results['timestamp'][:10]}  |  "
        f"Örnekler: {results['n_samples']}  |  "
        f"Embedding: `{results['embedding_model']}`\n\n"
        f"| Konfigürasyon | Mean Cosine Sim | OOV Oranı | İyileştirilen (%) |\n"
        f"|---|---|---|---|\n"
        f"| Baseline (BERT-only, `use_lemma=False`) "
        f"| {results['mean_sim_baseline']:.4f} "
        f"| {results['mean_oov_rate_raw']:.4f} "
        f"| — |\n"
        f"| Zemberek+BERT (`use_lemma=True`) "
        f"| {results['mean_sim_zemberek']:.4f} "
        f"| {results['mean_oov_rate_zem']:.4f} "
        f"| {results['pct_improved']:.1f}% |\n\n"
        f"**Δ Mean Cosine Similarity = {results['mean_delta']:+.4f}**  \n"
        f"**OOV İyileşmesi = {results['oov_improvement']:+.4f}** "
        f"(pozitif = Zemberek daha az UNK üretiyor)  \n\n"
        f"**Yorum:** {verdict}  \n\n"
        f"Sınıf bazlı Δ:\n"
    )
    for cls, info in results["per_label"].items():
        h4_section += f"- {cls}: n={info['n']}, mean_Δ={info['mean_delta']:+.4f}\n"

    h4_section += f"\n*Oluşturulma: {results['timestamp']}*\n"

    with open(table_path, "a", encoding="utf-8") as f:
        f.write(h4_section)
    logger.info(f"Ablation tablosu güncellendi: {table_path}")


# ─── Ana fonksiyon ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="H4 Zemberek Offline Ablasyon")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Embedding modeli (varsayılan: {DEFAULT_MODEL})")
    parser.add_argument("--data",  type=str, default=None,
                        help="QA çifti JSONL dosyası (query, passage alanları gerekir)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("H4 Zemberek Offline Ablasyon Başlıyor")
    logger.info("=" * 60)

    # 1. Tokenizer'ları yükle (Zemberek yoksa erken çıkar)
    tok_baseline, tok_zemberek = _load_tokenizers()

    # 2. Embedding modelini yükle
    try:
        embed_model = _load_embed_model(args.model)
    except Exception as e:
        logger.error(f"Embedding modeli yüklenemedi: {e}")
        raise SystemExit(1)

    # 3. Test verisini yükle
    qa_pairs = _load_qa_pairs(args.data)
    logger.info(f"Test seti: {len(qa_pairs)} çift")

    # 4. Ablasyon çalıştır
    logger.info("\nAblasyon çalışıyor...")
    t0 = time.perf_counter()
    results = run_ablation(qa_pairs, embed_model, tok_baseline, tok_zemberek)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    results["elapsed_ms"] = elapsed_ms

    # 5. JSON kaydet
    json_path = OUTPUT_DIR / "zemberek_ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Sonuçlar kaydedildi: {json_path}")

    # 6. Ablation table güncelle
    _update_ablation_table(results)

    # 7. Konsol özeti
    logger.info("\n" + "=" * 55)
    logger.info("H4 ZEMBEREK ABLASYON ÖZETİ")
    logger.info("=" * 55)
    logger.info(f"  Örnekler          : {results['n_samples']}")
    logger.info(f"  Mean Sim Baseline : {results['mean_sim_baseline']:.4f}")
    logger.info(f"  Mean Sim Zemberek : {results['mean_sim_zemberek']:.4f}")
    logger.info(f"  Δ Ortalama        : {results['mean_delta']:+.4f}")
    logger.info(f"  OOV Oran (Ham)    : {results['mean_oov_rate_raw']:.4f}")
    logger.info(f"  OOV Oran (Zem)    : {results['mean_oov_rate_zem']:.4f}")
    logger.info(f"  OOV İyileşme      : {results['oov_improvement']:+.4f}")
    logger.info(f"  İyileştirilen     : {results['n_improved']}/{results['n_samples']} "
                f"(%{results['pct_improved']})")
    logger.info(f"  Süre              : {elapsed_ms:.0f} ms")
    logger.info("=" * 55)

    if results["mean_delta"] > 0:
        logger.info("SONUÇ: Zemberek lemmatizasyonu retrieval kalitesini ARTIRIYOR → H4 DOĞRULANDI [OK]")
    else:
        logger.info("SONUÇ: Zemberek bu test setinde net fayda gösteremiyor → H4 KISMI [WARN]")
        logger.info("  Not: Daha büyük ve çeşitli QA seti ile yeniden test önerilebilir.")


if __name__ == "__main__":
    main()
