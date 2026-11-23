---
license: apache-2.0
language:
- tr
tags:
- finance
- banking
- turkish
- gemma3
- gguf
- lora
- quantized
base_model: google/gemma-3-4b-it
pipeline_tag: text-generation
---

# GEMMA3-FINANCE-V1 (Gemma-3-4B Turkish Finance SFT)

**GEMMA3-FINANCE-V1**, **MOSAIC-v2** Yüksek Lisans Tez Projesi (*"Türkçe Finans İçin Federe RAG Tabanlı Düşük Kaynaklı Ortamlarda Çalışan Adaptif Chatbot Mimarisi ve Uygulaması"*) kapsamında, kısıtlı kaynaklı uç noktalarda (Edge / Şube ortamları) Türkçe finans, bankacılık ve mevzuat sorgularını verimli şekilde yanıtlamak üzere özel olarak eğitilmiş ve **Q4_K_M GGUF** formatına kuantize edilmiş bir SLM (Küçük Dil Modeli) modelidir.

## Model ve Eğitim Özellikleri

- **Temel Model:** `unsloth/gemma-3-4b-it` (Google Gemma 3 4B Instruct)
- **Eğitim Verisi:** 22.735 örnek (~4.87M Token) Türkçe finansal SFT veri seti (`finetune_turkish_finance_sft.jsonl`)
  - Piyasa ve Şirket Analizleri (14.743 örnek)
  - KAP Şirket Bildirimleri ve Finansal Duygu (3.478 örnek)
  - Türkçe Finansal Muhakeme ve Hesaplama (1.791 örnek)
  - BDDK & TCMB Resmi Mevzuat ve Tebliğler (1.423 örnek)
  - Bankacılık Ürünleri ve Kredi Kılavuzları (1.345 örnek)
- **Eğitim Parametreleri:** LoRA ($r=32, \alpha=64$), `max_seq_len=512`, `lr=2e-4`, AdamW 8-bit, 1.000 adım.
- **Eğitim Kaybı (Loss):** 3.58  **0.8489** (Yaklaşık %76 kayıp düşüşü).
- **Kuantizasyon:** GGUF Q4_K_M (2.32 GB - Düşük kaynaklı bankacılık uç cihazları için optimize edilmiştir).

---

## Değerlendirme ve Benchmark Sonuçları

Model, Türkçe finans terminolojisi ve BIST duygu analizi test seti üzerinde değerlendirilmiştir:

| Metrik / Test | GEMMA3-FINANCE-V1 | Tez Başarı Kriteri | Durum |
| :--- | :---: | :---: | :---: |
| **BIST Finansal Duygu Analizi (Macro F1)** | **0.6284** | `> 0.6000` |  **Başarılı** |
| **BIST Duygu Analizi Doğruluğu (Accuracy)** | **%92.50** | `> %70.0` |  **Başarılı** |
| **Model Boyutu (VRAM / Disk)** | **2.32 GB** | `< 4 GB VRAM` |  **Başarılı** |

---

## Kullanım Talimatları

### 1. llama.cpp / llama-server ile Çalıştırma

Model bağımsız (standalone) bir GGUF olarak derlenmiştir. Doğrudan `llama-server` ile başlatabilirsiniz:

```bash
# Bağımsız Q4_K_M modeli ile sunucu başlatma (Port 8099)
llama-server \
  -m GEMMA3_FINANCE_V1.Q4_K_M.gguf \
  --port 8099 \
  --ctx-size 4096 \
  -ngl 99
```

### 2. LoRA Adaptörü Olarak Kullanım

Temel Gemma-3-4B modeli üzerine LoRA adaptörü olarak yüklemek için:

```bash
llama-server \
  -m gemma-3-4b-Q4_K_M.gguf \
  --lora gemma-3-4b-finance-lora.gguf \
  --port 8099 \
  --ctx-size 4096 \
  -ngl 99
```

---

## Mimarideki Yeri (MOSAIC-v2 Dağıtık Topoloji)

Bu model, MOSAIC-v2 sisteminde **Şube / Edge düğümlerinde** (VM-1 Bireysel Bankacılık ve VM-2 Kurumsal Bankacılık) ve HQ'da **hızlı SSS / chitchat ve temel finans sorgularını** düşük kaynak (2 vCPU, <4 GB RAM/VRAM) tüketerek anında yanıtlamak üzere konumlandırılmıştır.

