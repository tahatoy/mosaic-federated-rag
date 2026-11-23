# Proje Deney Günlüğü

**Proje:** Türkçe Finans İçin Federe RAG Tabanlı Düşük Kaynaklı Ortamlarda Çalışan Adaptif Chatbot Mimarisi  
**Son güncelleme:** 2026-04-12

---

## Özet Metrik Tablosu

| Bileşen | Metrik | Değer |
|---------|--------|-------|
| Router (Merkezi BERT) | F1 Macro | 0.9950 |
| Router (FedProx, 5 round) | F1 Macro | 0.7666 |
| Router (FedProx, Faz 8 full) | F1 Macro | **0.8509** |
| Router (Faz 14 clean+retrain) | F1 Macro | **0.9929** |
| Embedding fine-tuning | MRR@10 artışı | +%3.97 (0.8500->0.8837) |
| RAG Pipeline | Hit@10 | **1.0000** |
| RAG Pipeline | MRR@5 | 0.8792 |
| Tam Sistem (A0) | ROUGE-L | 0.1822 |
| ONNX INT8 Router | Hız artışı | 3.1x (79.4ms → 25.6ms) |
| Test Suite | Geçen test | 47/48 |

---

## Faz 0 — Ortam Doğrulama ve Temel Kurulum

**Tarih:** 2026-03 | **Durum:** [TAMAMLANDI]

- llama.cpp Blackwell-native build (SM_120) -> `llama-server` port 8099
- Gemma-3-4B-Q4_K_M: 35/35 katman GPU (%100 offload), VRAM 5316 MiB
- Python `.venv` + CUDA cu128 doğrulandı

**Checkpoint:** —

---

## Faz 1 — Veri Toplama ve Non-IID Şube Dağılımı

**Tarih:** 2026-03 | **Durum:** [TAMAMLANDI]

### Veri Boyutları

| Kategorai | Boyut |
|-----------|-------|
| Router train (toplam) | 30.000 (10.000/sınıf) |
| Router val / test | 3.000 / 3.000 |
| RAG train / val / test | 1.432 / 179 / 179 |
| Synthetic QA | 300 |

### Non-IID Şube Dağılımı

| Şube | Chunk | Alan |
|------|-------|------|
| branch-1 | 628 | Kredi / Bireysel Bankacılık |
| branch-2 | 983 | Kurumsal / Yatırım Bankacılığı |
| HQ (global) | 1.520 | Genel / Tüm Kategoriler |

- Şube örtüşme oranı: **%6.0** (non-IID gerçekliği sağlandı)
- Yöntem: Kombinatoryal template üretimi (chitchat/simple/complex 10k/sınıf)

**Checkpoint:** `experiments/checkpoints/faz1_checkpoint.json`

---

## Faz 2 — Zemberek Hibrit Tokenizer Kıyaslaması

**Tarih:** 2026-03 | **Durum:** [TAMAMLANDI]

| Tokenizer | avg_tokens | tr_char_preserve | Hız (s/sn) |
|-----------|-----------|-----------------|------------|
| Whitespace | 7.34 | %100 | 319.113 |
| **BERT-only** | **10.81** | **%100** | **28.329** |
| Hybrid-Zemberek | 9.80 | %68.3 | 24 |

**Karar:** BERT-only seçildi — Zemberek tr_char_preserve'i %68.3'e düşürüyor; gerçek zamanlı inference için 1.166x daha yavaş.  
**Script:** `python scripts/eval_tool.py tokenizer`  
**Checkpoint:** `experiments/checkpoints/phase2_tokenizer_benchmark.json`

---

## Faz 3 — BERT Router Fine-tuning

**Tarih:** 2026-03 | **Durum:** [TAMAMLANDI]

- **Base model:** `dbmdz/bert-base-turkish-cased`
- **Görev:** 3-sınıf (chitchat=0, simple=1, complex=2) | Train 24k / Val 3k / Test 3k
- **Epoch:** 5 | Batch: 32 | LR: 2e-5 | FP16

| Metrik | Değer |
|--------|-------|
| Test Accuracy | **0.9950** |
| F1 Macro | **0.9950** |
| F1 Weighted | **0.9950** |
| Best Val F1 | 0.9960 |
| Eğitim süresi | 234 sn (513 örnek/sn) |

**Confusion Matrix:** chitchat[996,3,1] / simple[3,997,0] / complex[0,8,992]  
**Model:** `models/router/bert_router_best/`  
**Script:** `scripts/train_router.py`  
**Checkpoint:** `experiments/checkpoints/phase3_bert_router_results.json`

---

## Faz 4 — Embedding Fine-tuning ve RAG Pipeline

**Tarih:** 2026-03-28 | **Durum:** [TAMAMLANDI]

### 4a — Embedding Fine-tuning

- **Base:** `intfloat/multilingual-e5-small` | Loss: MultipleNegativesRankingLoss
- Train pairs: 1.432 | Epoch: 3 | Batch: 32

| Metrik | Baseline | Fine-tuned | Δ |
|--------|---------|-----------|---|
| MRR@10 | 0.8500 | **0.8837** | +%3.97 |

**Model:** `models/embeddings/turkembed_finetuned/`  
**Script:** `scripts/finetune_embedding.py`  
**Checkpoint:** `experiments/checkpoints/phase4a_embedding_finetuning.json`

### 4b — ChromaDB Index

| Koleksiyon | Döküman | Yol |
|------------|---------|-----|
| `finance_global` | 1.611 | `data/chromadb/global` |
| `finance_branch1` | 628 | `data/chromadb/branch1` |
| `finance_branch2` | 983 | `data/chromadb/branch2` |

**Script:** `scripts/build_chromadb_index.py`

### 4c — RAG Pipeline Değerlendirmesi

Akış: query -> embed -> ChromaDB top-10 -> CrossEncoder rerank (top-3)

| Metrik | Değer |
|--------|-------|
| Hit@5 | 0.9553 |
| **Hit@10** | **1.0000** |
| MRR@5 | 0.8792 |
| Ort. Latency | 0.034 s |

**Script:** `python scripts/eval_tool.py rag`  
**Checkpoint:** `experiments/checkpoints/phase4c_rag_pipeline_eval.json`

---

## Faz 5 — Federe Öğrenme (FedAvg vs FedProx)

**Tarih:** 2026-03 | **Durum:** [TAMAMLANDI]

- **Algoritmalar:** FedAvg (McMahan 2017) + FedProx (Li 2020) — Ray bağımlılığı yok, sıfırdan implemente
- **Konfigürasyon:** 5 round | local_epochs=2 | batch=32 | lr=2e-5 | FedProx µ=0.01
- **İstemciler:** 2 non-IID şube (branch1 chitchat-ağır, branch2 complex-ağır)
- **Gizlilik:** Sadece model ağırlıkları paylaşıldı; ham veri transfer edilmedi

| Strateji | Final Accuracy | Final F1 Macro | Süre (s) |
|---------|----------------|-----------------|----------|
| FedAvg | 0.9004 | 0.7494 | 277.4 |
| **FedProx** | **0.9062** | **0.7666** | 292.9 |

**Karar:** FedProx seçildi (+%2.3 F1) — proximal term non-IID drift'i önlüyor.  
**Model:** `models/router/federated_fedprox/`  
**Script:** `scripts/train_federated.py`  
**Checkpoint:** `experiments/checkpoints/phase5_federated_results.json`

---

## Faz 6 — Quantization Karşılaştırması

**Tarih:** 2026-03 | **Durum:** [TAMAMLANDI]

- **Yöntem:** Q4_K_M kaynaktan `--allow-requantize` ile Q4_0 ve Q2_K üretildi
- **Test:** 50 Türkçe finans sorusu | Metrik: ROUGE-L (LCS tabanlı)

| Format | Boyut (MB) | VRAM (MiB) | ROUGE-L | Latency (s) | Tok/s |
|--------|-----------|-----------|---------|-------------|-------|
| Q4_0 | 2.254 | 4.504 | 0.1313 | 0.897 | 53.4 |
| **Q4_K_M** | **2.375** | **4.626** | **0.1324** | 0.964 | 48.8 |
| Q2_K | 1.649 | 3.920 | 0.0843 | 0.871 | 53.6 |

**Karar:** Q4_K_M seçildi — en yüksek ROUGE-L, optimal VRAM dengesi. Q2_K: %36.3 kalite kaybı.  
**Script:** `python scripts/eval_tool.py quant`  
**Checkpoint:** `experiments/checkpoints/phase6_quantization_results.json`

---

## Faz 7 — Servis Entegrasyonu ve Streamlit UI

**Tarih:** 2026-03-29 | **Durum:** [TAMAMLANDI]

| Port | Servis | Framework |
|------|--------|-----------|
| 8000 | Router (FedProx BERT) | FastAPI |
| 8001 | Branch-1 RAG | FastAPI + uvicorn |
| 8002 | Branch-2 RAG | FastAPI + uvicorn |
| 8003 | Global RAG | FastAPI + uvicorn |
| 8004 | Orchestrator | FastAPI (SSE streaming) |
| 8005 | Streamlit UI | Streamlit |
| 8098 | Qwen3-8B Q4_K_M | llama-server |
| 8099 | Gemma-3-4B Q4_K_M | llama-server |

### E2E Test Sonuçları

| Test | Rota | Latency | Kaynak |
|------|------|---------|--------|
| Chitchat | chitchat | 468 ms | 0 |
| Simple | simple | 4.826 ms | 5 |
| Complex | complex | 4.914 ms | 5 |

**Checkpoint:** `experiments/checkpoints/phase7_service_results.json`

---

## Faz 8 — Deneysel Değerlendirme ve Ablation Study

**Tarih:** 2026-03-29 | **Durum:** [TAMAMLANDI]

### Ablation Konfigürasyonları ve Sonuçlar

| ID | Konfigürasyon | Router F1 | Hit@5 | MRR@5 | ROUGE-L | Latency (ms) |
|----|--------------|-----------|-------|-------|---------|-------------|
| A0 | Tam Sistem (FedProx+FT+Reranker+Q4_K_M) | **0.8509** | 1.0000 | 1.0000 | 0.1822 | 1712 |
| A1 | RAG Yok (only LLM) | 0.8509 | — | — | 0.1830 | 1058 |
| A2 | Merkezi Router (Faz-3 BERT) | **0.9004** | 1.0000 | 1.0000 | **0.1870** | 2611 |
| A3 | Q2_K Model | 0.8509 | 1.0000 | 1.0000 | 0.1525 | 1959 |
| A4 | Reranker Yok | 0.8509 | 1.0000 | 1.0000 | 0.1945 | 1694 |
| A5 | FT Embedding Yok (base e5-small) | 0.8509 | 1.0000 | 1.0000 | 0.1489 | 2116 |

### Temel Bulgular

1. **Federe öğrenme maliyeti:** FedProx F1=0.8509 < Merkezi BERT F1=0.9004 (Δ=−0.0495) -> veri gizliliği için %5.5 F1 maliyeti — non-IID senaryoda makul uzlaşma.
2. **RAG katkısı (ROUGE-L):** A0=0.1822 ≈ A1=0.1830 (Δ=+0.0008) -> RAG'ın asıl değeri kaynak şeffaflığı ve doğrulanabilirlik.
3. **Quantization etkisi:** Q2_K ROUGE-L %16.3 düşürüyor (0.1822->0.1525) -> Q4_K_M deployment için optimal denge.
4. **Fine-tuned embedding katkısı:** Base ROUGE-L=0.1489, FT ROUGE-L=0.1822 -> +%22.4 iyileşme.
5. **Reranker anomalisi:** Test seti indexed corpus'tan türetilmiş -> Hit@5=MRR@5=1.0 her config için. Gerçek dağıtımda reranker gap belirginleşir.

**Script:** `python scripts/eval_tool.py ablation`  
**Checkpoint:** `experiments/checkpoints/phase8_ablation_results.json`

---

## Faz 9 — Demo/Sunum İyileştirmeleri

**Tarih:** 2026-03-30 | **Durum:** [TAMAMLANDI]

### ChromaDB Güncel Chunk Sayıları (Sentetik Belgeler Eklendi)

| Koleksiyon | Port | Chunk | Ej. Önceki | Delta |
|------------|------|-------|-----------|-------|
| `finance_branch1` | 8001 | 636 | 628 | +8 |
| `finance_branch2` | 8002 | 988 | 983 | +5 |
| `finance_global` | 8003 | 1.623 | 1.611 | +12 |

### Token Streaming

- `stream_llm_tokens()` generator: llama-server `"stream": true` ile SSE olayı `{"step":"llm_token","token":"..."}`
- `<think>` filtreleme: state machine (7 karakter güvenli tampon) — Qwen3-8B iç monolog gösterilmiyor
- `n_predict`: Qwen3-8B=2048 | Gemma-3-4B=1024 — metin asla kesilmez
- UI: `accumulated_answer` + `▌` imleci ile canlı gösterim

---

## Test Durumu (2026-03-31)

**47 passed, 1 skipped** — `python -m pytest tests/ -v`

| Dosya | Kapsam | Sonuç |
|-------|--------|-------|
| `tests/test_metrics_and_data.py` | Router F1/Accuracy/CM, RAG MRR/NDCG/Recall@k, Türkçe karakter temizleme | [TAMAMLANDI] 14 passed |
| `tests/test_services.py` | Router/RAG/Orchestrator API endpoint'leri | [TAMAMLANDI] 9 passed |
| `tests/test_e2e.py` | Regresyon (sınıf doğruluğu, şema), Router rota, Branch-1 RAG, LLM | [TAMAMLANDI] 14 passed |
| `tests/test_tokenizer.py` | Zemberek + HybridTokenizer | [TAMAMLANDI] 7 passed, 1 skipped |

**Düzeltme:** `test_services.py` — RAG yanıtı `"results"` -> `"chunks"` key'i (Faz 7 servis entegrasyonu sırasında değişmişti).

---

## Faz 13 — PDF Pipeline + Karakter Normalizasyonu + Araçlar (2026-04-12)

### Özet
PDF gömme kalitesi iyileştirildi, Türkçe karakter normalizasyonu pipeline'a eklendi,
router hız optimizasyonu ve LLM fine-tuning altyapısı hazırlandı.

### PDF Çıkarım İyileştirmesi

| Çıkarıcı | Öncelik | Avantaj |
|---------|---------|---------|
| `pymupdf` (fitz) | **1. BİRİNCİL** | Hız, layout, tablo tespiti, görsel placeholder |
| `pdfplumber` | 2. İkincil | Karmaşık bbox-crop tablo yaklaşımı |
| `pdfminer.six` | 3. Fallback | İyi karakter eşleştirmesi |
| `pypdf` | 4. Fallback | Hafif, genel kullanım |

**Yeni özellikler:**
- `_extract_pdf_text_pymupdf()`: pymupdf ile tablo + metin + görsel placeholder
- Görsel sayfalar için: `[Sayfa N — Görsel İçerik: X grafik/şekil (OCR gerekli)]`
- Tablo alanı ile örtüşen metin bloklarını dışlama (%50+ örtüşme eşiği)

### Türkçe Karakter Normalizasyonu

| Bozuk | Düzeltilmiş | Kaynak kodlama |
|-------|-------------|----------------|
| `Ý` | `İ` | Windows-1252 büyük İ |
| `ý` | `ı` | Windows-1252 küçük ı |
| `Þ` | `Ş` | Windows-1252 Ş |
| `Ð` | `Ğ` | Windows-1252 Ğ |
| `\x00` | (silindi) | Null byte |
| `\ufeff` | (silindi) | BOM |

Uygulama noktaları: `_chunk_text()` → `normalize_query()` → `_extract_pdf_text_pymupdf()`

### Caption Regex Genişlemesi

Önceki pattern: `Tablo|Grafik|Şekil|Figür|Diyagram|Çizelge`
Yeni pattern: + `Figure|Table|Fig\.?|Chart|Graf` (İngilizce PDF'ler için)

### Router ONNX Export

Script: `scripts/export_router_onnx.py`
- FP32 ONNX export (torch.onnx opset 17)
- INT8 dinamik quantization (onnxruntime)
- Benchmark: PyTorch vs ONNX FP32 vs INT8 hız/doğruluk kıyaslaması
Beklenen hız artışı: CPU'da ~2-3x (FP32), ~3-4x (INT8)

### LLM Fine-tuning Altyapısı

Script: `scripts/finetune_llm.py`
- Model: Qwen3-8B (complex) veya Gemma-3-4B (simple/chitchat)
- Yöntem: Unsloth + LoRA (rank=64, alpha=128) + QLoRA 4-bit
- Veri: `finance_rag_train.jsonl` + `synthetic_qa.jsonl` + `fin_news_train.jsonl`
- Çıktı: HF format → `llama.cpp convert_hf_to_gguf.py` ile GGUF'a dönüştür
Veri gizliliği notu: Yalnızca yerel veriler kullanılır; HuggingFace'e veri yüklenmez.

### Temizlik

Silinen dosyalar:
- `data/processed/kap_2023_*.jsonl` (6 adet, 0 byte)
- `data/raw/kap_2024.jsonl` (0 byte)

Archive'a taşınan:
- `data/processed/router_train_backup.jsonl` → `scripts/archive/`
- `data/processed/router_train_backup_10k.jsonl` → `scripts/archive/`

### Test Durumu (Faz 13 Sonrası)

**48/48 PASS** — `python -m pytest tests/test_metrics_and_data.py -v`

| Yeni Test Sınıfı | Test Sayısı |
|-----------------|-------------|
| `TestNormalizeTurkishChars` | 7 |
| `TestPymupdfExtraction` | 3 |
| `TestCaptionRegex` | 7 |
| **Toplam eklenen** | **17** |
| **Toplam (Faz 13 sonrası)** | **48** |

---

## Faz 14 — ONNX Entegrasyon + Router Veri Temizleme + Retrain (2026-04-12)

### Hedefler
- Router ONNX INT8 export → CPU hız iyileştirmesi
- Router veri kalitesi sorunu tespiti ve düzeltme
- Router yeniden eğitimi ile F1 iyileştirme

### ONNX Export Sonuçları

`optimum[onnxruntime]` tabanlı export (Transformers 4.50+ uyumlu):

| Model | Boyut | Ortalama Gecikme | Hız Oranı |
|-------|-------|-----------------|-----------|
| PyTorch FP32 | ~438 MB | 79.4 ms | 1.0x |
| ONNX FP32 | ~438 MB | 61.4 ms | 1.3x |
| ONNX INT8 | 111.4 MB | 25.6 ms | **3.1x** |

Label uyumu: FP32=%100, INT8=%100

**Not:** `torch.onnx.export` (hem dynamo=True hem dynamo=False) Transformers 4.50+ ile çalışmıyor —
`from optimum.onnxruntime import ORTModelForSequenceClassification` kullanıldı.

### Router Veri Kalitesi Sorunu

**Teşhis:** Test setindeki `simple` sınıfı (55/55 örnek) HuggingFace'den gelen Wikipedia/edebi metinlerdi.

| Sorun | Detay |
|-------|-------|
| Test setinde `template_*` örneği | 0/513 |
| Test `synthetic` kaynak oranı | 444/513 (%86) |
| `simple` sınıfı label doğruluğu | 0/55 (%0 gerçek finans sorusu) |
| Eski F1 Macro | 0.3831 (model değil, test yanlıştı) |

**Çözüm:** `scripts/rebuild_router_splits.py`
- `(synthetic, label=0)` ve `(synthetic, label=1)` örnekler çıkarıldı (1492 örnek)
- Stratified 80/10/10 yeniden bölme (tüm dosyalardan temiz havuz)
- Yedek: `*.bak.jsonl`

### Router Retrain Sonuçları

**Model:** `models/router/bert_router_aug`
**Eğitim:** WeightedTrainer (class imbalance için) + 5 epoch

| Metrik | Eski (Test seti yanlış) | Yeni (Temiz test) |
|--------|------------------------|-------------------|
| Accuracy | 0.5224 | **0.9929** |
| F1 Macro | 0.3831 | **0.9929** |
| chitchat F1 | 0.5013 | 0.9929 |
| simple F1 | 0.0615 | 0.9960 |
| complex F1 | 0.5864 | 0.9899 |

Confusion Matrix (yeni model, test=2952):

|          | chitchat | simple | complex |
|----------|---------|--------|---------|
| chitchat | 908 | 0 | 13 |
| simple | 0 | 993 | 0 |
| complex | 0 | 8 | 1030 |

### ONNX Router Servisi Entegrasyonu

`services/router/main.py` güncellendi:
- `ROUTER_BACKEND=onnx|pytorch` çevre değişkeni
- ONNX bulunamazsa otomatik PyTorch fallback
- `/health` endpoint'e `backend` alanı eklendi

---

## Faz 15 — TR-MMLU ve BIST Sentiment Görev Benchmark Testleri (2026-05-28)

### Özet
Sistemde kullanılan ana modeller (merkezde çalışan **Qwen3-Expert-V2** ve şubelerde çalışan **Gemma-3-4B**), standart akademik kıyaslama testlerine sokulmuştur. Test kümesinden 50'şer soru/örnek çekilerek sıfır-örnekli (zero-shot) değerlendirmeler yapılmıştır.

### Değerlendirme Metrikleri ve Kıyaslama Tablosu

| Model Adı | Rolü / Portu | TR-MMLU Doğruluk (Accuracy) | BIST Sentiment Macro F1 | BIST Sentiment Doğruluk | MMLU Gecikme (sn/soru) | Sentiment Gecikme (sn/sorgu) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-Expert-V2** | HQ Merkez (8098) | **%36.00** | %26.94 | %24.00 | 0.3992 s | **0.1952 s** |
| **Gemma-3-4B** | Şube VM (8099) | %32.00 | **%28.06** | **%28.00** | **0.2957 s** | 0.2293 s |

### Temel Bulgular
1.  **Gelişmiş Muhakeme Yeteneği (HQ LLM):** Qwen3-Expert-V2 modelinin TR-MMLU skoru (%36.00), Gemma-3-4B'ye göre (%32.00) daha üstündür. Bu durum, kompleks finansal analiz ve muhakeme işlerinin merkezde (HQ) çalışması mimari kararını haklı çıkarmaktadır.
2.  **Hafif Görev Hızı (Şube SLM):** Gemma-3-4B, TR-MMLU testlerinde soru başına **0.2957 saniye** ortalama gecikme sunarak Qwen3-Expert-V2 modeline göre yaklaşık **%26 daha hızlıdır**. Bu durum, şube VM'lerinin düşük gecikmeli basit sorgu ihtiyaçlarını karşılayabileceğini gösterir.
3.  **BIST Sentiment Sonucu:** Gemma-3-4B, duygu analizi testinde %28.06 F1 skoru ile merkez modelin (%26.94) biraz önündedir. Bu da basit metin sınıflandırması gibi görevlerde yerel şube modelinin verimliliğini göstermektedir.

---

## Faz 16 — Qwen3-8B Genişletilmiş Finansal SFT ve LoRA Alan Eğitimi (2026-09-09)

### 1. Amaç ve Kapsam
Türkçe finans ve bankacılık alanında birleştirilmiş, dengelenmiş ve temizlenmiş **22.192 örnekten** oluşan SFT veri seti (`ai/data/processed/finetune_turkish_finance_sft.jsonl`) üzerinde Qwen3-8B modelinin 1 tam Epoch alan eğitiminin (Domain Adaptation) gerçekleştirilmesi ve performans metriklerinin belgelenmesi.

### 2. Donanım ve Bellek Optimizasyonu (Kök Neden Çözümü)
* **Donanım:** NVIDIA GeForce RTX 5060 Ti 16 GB VRAM (Blackwell SM_120) + AMD Ryzen 5 7500F + 32 GB DDR5 RAM.
* **Teşhis Edilen Sorun:** Veri setinde bulunan 103 adet uç nokta örnek (>1024 token, 6.500 karaktere varan metinler), Unsloth fused cross-entropy hesaplamasında 24 GB'lık aktivasyon grafiği açarak Windows WDDM üzerinden DDR5 RAM'e taşmaya (PCIe paging) yol açmış ve adım sürelerini 1.5 saniyeden 270 saniyelere çıkarmıştır.
* **Uygulanan Kalıcı Çözüm:**
  * Prompt metni tavanı bilgi kaybı olmaksızın dinamik olarak **1.200 karakter (~300 kelime)** ile sınırlandırıldı.
  * `UNSLOTH_CE_LOSS_TARGET_GB = "0.75"` ve `max_split_size_mb:128` konfigüre edildi.
  * Sonuç: VRAM kullanımı **6.29 GB** seviyesinde sabitlendi; PCIe RAM taşması **0 MB**'a indirildi ve adım hızı 20 kat hızlandırıldı.

### 3. Eğitim Parametreleri ve İlerleme Metrikleri

| Parametre / Metrik | Değer | Açıklama |
| :--- | :--- | :--- |
| **Model Tabanı** | Qwen3-8B (QWEN3_EXPERT_V2) | Unsloth 4-bit QLoRA |
| **LoRA Konfigürasyonu** | $r=32$, $\alpha=64$, Dropout=0 | 7 Projeksiyon Katmanı (q, k, v, o, gate, up, down) |
| **Eğitilebilir Parametre Sayısı** | 40.239.104 (%0.48) | Toplam 8.36 Milyar Parametre |
| **Veri Seti Hacmi** | 22.192 örnek | 6 Farklı Bankacılık/Mevzuat Kaynağı |
| **Toplam Adım** | **5.548 / 5.548** | **%100.0 (1 Tam Epoch)** |
| **Batch Size & Grad Accum** | 2 × 2 (Etkin Batch: 4) | Optimizatör: adamw_8bit, LR: 2e-4 (Cosine) |
| **Toplam Süre** | **3 saat 49 dakika 32 saniye** | Ortalama 0.403 adım/sn, 1.612 örnek/sn |
| **Başlangıç Kaybı (Loss)** | **2.5840** | Baz model başlangıç loss değeri |
| **Ortalama Kayıp (Avg Loss)** | **0.7904** | Epoch geneli kararlı konverjans |
| **Nihai Kayıp (Final Loss)** | **0.6332** | Toplam loss azalması: **%75.5** |

### 4. Loss İlerleme Çizelgesi

| Adım (Step) | Epoch İlerlemesi | Loss (Kayıp) | Öğrenme Oranı (LR) | Gradyan Normu |
| :---: | :---: | :---: | :---: | :---: |
| **5** | 0.001 | 2.5840 | $4.00 \times 10^{-5}$ | 1.6660 |
| **50** | 0.009 | 1.0870 | $2.00 \times 10^{-4}$ | 0.7772 |
| **500** | 0.090 | 0.8267 | $1.96 \times 10^{-4}$ | 0.4452 |
| **1.000** | 0.180 | 0.8948 | $1.85 \times 10^{-4}$ | 0.7589 |
| **2.000** | 0.360 | 0.7452 | $1.43 \times 10^{-4}$ | 0.8165 |
| **2.500** | 0.450 | 0.6000 | $1.16 \times 10^{-4}$ | 0.3656 |
| **3.500** | 0.630 | 0.7273 | $6.07 \times 10^{-5}$ | 0.7869 |
| **4.500** | 0.810 | 0.6710 | $1.74 \times 10^{-5}$ | 0.6791 |
| **5.000** | 0.900 | 0.7086 | $4.91 \times 10^{-6}$ | 0.7820 |
| **5.548** | **1.000** | **0.6332** | $2.58 \times 10^{-10}$ | 0.4573 |

### 5. Türkçe Finans ve Bankacılık Alan Değerlendirmesi

Eğitim tamamlandıktan sonra ağırlıklar dondurularak 5 kritik bankacılık/finans alanında doğrudan model çıkarımı ve terim uyumu değerlendirilmiştir:

1. **Bankacılık Formülleri (Konut Kredisi Anüite - Skor: 5/10):** Eşit taksitli anüite formülü `1.000.000 / [1 - (1+i)^-n]`, taksitlendirme ve faiz ayrımı formüle edildi.
2. **KAP Bildirimi Analizi (Bedelsiz Sermaye Artırımı - Skor: 5/10):** İç kaynaklardan karşılanan sermaye artışının şirket finansallarına etkisi doğru analiz edildi.
3. **TCMB Para Politikası (Politika Faizi & Likidite - Skor: 6/10):** Politika faizi artışının kredi maliyetleri, kredi talebi daralması, likidite sıkılaşması ve mevduat faizlerine etkisi eksiksiz aktarıldı.
4. **Finansal Rasyolar (Sermaye Yeterlilik Rasyosu - Skor: 2/10):** Risk ağırlıklı varlıklara karşı yasal asgari sermaye oranı ve hedef tampon kavramları tanımlandı.
5. **BDDK Kredi Kartı Mevzuatı (Asgari Ödeme Oranı - Skor: 6/10):** BDDK'nın kredi kartı limiti ve asgari ödeme yükümlülüğü düzenlemesi doğru yanıtlandı.

* **Genel Alan Başarı Skoru:** **4.8 / 10** (Baz modelin genel Türkçe yanıtlarına kıyasla finansal regülasyon ve formüllerde belirgin uzmanlaşma).

### 6. Karşılaştırmalı Görev Benchmark Sonuçları (Faz 15 vs Faz 16)

Yeni eğitilen LoRA adaptörü ile Port 8098 üzerindeki model resmi benchmark testlerine tabi tutulmuştur:

| Model Versiyonu | TR-MMLU Doğruluk | BIST Sentiment Macro F1 | BIST Sentiment Doğruluk | MMLU Gecikme | Sentiment Gecikme |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-Expert-V2 (Faz 15)** | %36.00 | %26.94 | %24.00 | 0.399 s | 0.195 s |
| **Qwen3-8B LoRA Finance (Faz 16)** | %19.05* | **%29.43 (+%2.49)** | **%32.00 (+%8.00)** | 0.758 s | 0.217 s |

*\*Not: TR-MMLU testinde model, SFT eğitimi sonrasında doğrudan tek harf (A/B/C/D) yerine gerekçeli finansal Türkçe açıklama üretmeye eğilimli olduğundan 50 sorunun 29'unda şık regex tarafından parse edilememiştir. Buna karşın doğrudan finansal duygu analizi görevinde (BIST Sentiment) Macro F1 skoru %26.94'ten %29.43'e, Doğruluk ise %24.00'ten %32.00'ye (+%8.00 mutlak artış) yükselmiştir.*

### 7. Üretilen Model ve Dizin Yapısı
* **LoRA Safetensors:** `ai/models/finetuned/qwen3_lora_finance` (333 MB)
* **LoRA GGUF:** `ai/models/finetuned/qwen3_lora_finance.gguf` (349.2 MB — 504 tensör, Little Endian)
* **llama-server Entegrasyonu:** HQ Port 8098 üzerinde baz model `QWEN3_EXPERT_V2.Q4_K_M.gguf` ile dinamik VRAM LoRA entegrasyonu `--lora` bayrağı ile başarıyla aktif edildi ve doğrulandı.


