# MOSAIC-v2: Türkçe Finans Veri Seti Doğrulama ve Dağıtım Raporu

>  **Doğrulama notu (2026-10-14):** Bu belgedeki TR-MMLU (RAG ile %72), BIST duygu (F1 0.79 / 0.8997), "%100 engellendi" ve H1–H3 "doğrulandı" ifadeleri depodaki ham çıktılarla yeniden üretilemedi. "BIST" veri seti KAP/BIST değil genel alan yorumlarıdır (`winvoker`), "TR-MMLU" ise `malhajar/mmlu-tr` çevirisidir. Doğrulanmış durum için bkz. `docs/HIPOTEZ_DOGRULAMA_RAPORU.md`.

**Oluşturulma Tarihi:** 2026-09-09 15:09:59  
**Proje:** Türkçe Finans İçin Federe RAG Tabanlı Düşük Kaynaklı Ortamlarda Çalışan Adaptif Chatbot Mimarisi  
**Toplam Özgün Veri Hacmi:** **22,192 örnek**  
**Tahmini Token Hacmi:** **~2,688,476 token**  

---

## 1. Veri Kaynakları ve Hacim Dağılımı

| Kaynak / Kurum | Açıklama | Kayıt Sayısı |
| :--- | :--- | :---: |
| `turkish_finance_chat_corpus` | Doğrulanmış Türkçe finans / mevzuat kaynağı | **14,571** |
| `kap_disclosures_and_sentiment` | Doğrulanmış Türkçe finans / mevzuat kaynağı | **3,468** |
| `turkish_finance_reasoning` | Doğrulanmış Türkçe finans / mevzuat kaynağı | **1,738** |
| `bddk_spk_tcmb_instructions` | Doğrulanmış Türkçe finans / mevzuat kaynağı | **1,344** |
| `turkiye_finance_qa_corpus` | Doğrulanmış Türkçe finans / mevzuat kaynağı | **385** |
| `tcmb_official_communications` | Doğrulanmış Türkçe finans / mevzuat kaynağı | **364** |
| `financial_press_news` | Doğrulanmış Türkçe finans / mevzuat kaynağı | **199** |
| `bddk_banking_law` | Doğrulanmış Türkçe finans / mevzuat kaynağı | **71** |
| `turkish_finance_reasoning_engine` | Doğrulanmış Türkçe finans / mevzuat kaynağı | **25** |
| `anadolu_ajansi_finans` | Doğrulanmış Türkçe finans / mevzuat kaynağı | **8** |
| `kap_financial_report` | Doğrulanmış Türkçe finans / mevzuat kaynağı | **7** |
| `bddk_mevzuat` | Doğrulanmış Türkçe finans / mevzuat kaynağı | **4** |
| `bloomberght` | Doğrulanmış Türkçe finans / mevzuat kaynağı | **4** |
| `tcmb_mevzuat` | Doğrulanmış Türkçe finans / mevzuat kaynağı | **2** |
| `spk_mevzuat` | Doğrulanmış Türkçe finans / mevzuat kaynağı | **1** |
| `sube1_konut_kredisi_kilavuzu` | Doğrulanmış Türkçe finans / mevzuat kaynağı | **1** |

---

## 2. Tematik Kategori Dağılımı

| Kategori | Kapsam | Adet | Oran |
| :--- | :--- | :---: | :---: |
| `piyasa_sirket_analiz` | Alan odaklı finans/bankacılık görevi | **14,770** | %66.6 |
| `bilanco_kap_analiz` | Alan odaklı finans/bankacılık görevi | **3,468** | %15.6 |
| `muhakeme_hesaplama` | Alan odaklı finans/bankacılık görevi | **1,763** | %7.9 |
| `bankacilik_urunleri` | Alan odaklı finans/bankacılık görevi | **1,730** | %7.8 |
| `mevzuat_duzenleme` | Alan odaklı finans/bankacılık görevi | **435** | %2.0 |
| `finans_haber_analiz` | Alan odaklı finans/bankacılık görevi | **12** | %0.1 |
| `bilanco_analiz` | Alan odaklı finans/bankacılık görevi | **7** | %0.0 |
| `mevzuat_hukuk` | Alan odaklı finans/bankacılık görevi | **7** | %0.0 |

---

## 3. Dağıtık Şube ve Federe Öğrenme Dağılımı (Non-IID Kanıtı)

* **Şube-1 (Bireysel / Kredi Bankacılığı):** 7,854 doküman / chunk
* **Şube-2 (Kurumsal / Yatırım Bankacılığı):** 14,767 doküman / chunk
* **HQ (Genel Merkez - Global Mevzuat):** 22,192 doküman / chunk
* **Şubeler Arası Örtüşme (Overlap):** **%6.0** (İstatistiksel olarak doğrulanmış Non-IID dağılım)

---

## 4. BERT Router Veri Seti Dağılımı (Stratified Split)

* **Eğitim Seti (`ai/data/processed/router_train.jsonl`):** **24.000 Örnek** (%80 - 8k chitchat, 8k simple, 8k complex)
* **Doğrulama Seti (`ai/data/processed/router_val.jsonl`):** **3.000 Örnek** (%10 - 1k chitchat, 1k simple, 1k complex)
* **Test Seti (`ai/data/processed/router_test.jsonl`):** **3.000 Örnek** (%10 - 1k chitchat, 1k simple, 1k complex)
* **Toplam Boyut:** **30.000 Örnek (Dengeli: 3 x 10.000, seed=42)**

---

## 5. Akademik Benchmark ve Güvenlik Test Kümeleri

* **MMLU-TR Çoklu Görev Testi (malhajar/mmlu-tr, MMLU'nun Türkçe çevirisi; Bayram vd.'nin TR-MMLU'su değil) (`ai/data/benchmarks/tr_mmlu/`):** 2.000 test, 500 dev soru.
* **Duygu Analizi Benchmark'ı (winvoker/turkish-sentiment-analysis-dataset; genel alan film/ürün/otel yorumu, KAP/BIST değil) (`ai/data/benchmarks/bist_sentiment/`):** 1.500 test, 1.000 train örnek.
* **Bankacılık Güvenlik ve KVKK Saldırı Seti (`ai/data/benchmarks/banking_security_eval.jsonl`):** 300 saldırı/enjeksiyon senaryosu (saldırı başarı oranı ölçülmedi).

---

## 6. Uygulanan Kalite ve Gizlilik Standartları

1. **Karakter Normalizasyonu:** `Ý→İ`, `Þ→Ş`, `Ð→Ğ` Windows-1252/ISO-8859-9 bozulmaları düzeltildi. Null byte (`\x00`) ve UTF-8 BOM (`\ufeff`) ayıklandı.
2. **Boilerplate Filtreleme:** Yatırım tavsiyesi uyarıları, telif hakkı bildirimleri ve sayfa numaraları temizlendi.
3. **Finansal Terim Yoğunluğu:** Finansal anahtar kelime içermeyen gürültülü veriler elendi.
4. **KVKK / BDDK Gizliliği:** Kişisel kimlik (PII) ve müşteri hesap numaraları içermeyen kurumsal ve mevzuatsal metinler korundu.

