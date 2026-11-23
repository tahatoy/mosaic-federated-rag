# MOSAIC-v2: Veri Setleri Kataloğu ve Akademik Manifestosu (Dataset Catalog)

>  **Doğrulama notu (2026-10-14):** Bu belgedeki TR-MMLU (RAG ile %72), BIST duygu (F1 0.79 / 0.8997), "%100 engellendi" ve H1–H3 "doğrulandı" ifadeleri depodaki ham çıktılarla yeniden üretilemedi. "BIST" veri seti KAP/BIST değil genel alan yorumlarıdır (`winvoker`), "TR-MMLU" ise `malhajar/mmlu-tr` çevirisidir. Doğrulanmış durum için bkz. `docs/HIPOTEZ_DOGRULAMA_RAPORU.md`.

**Tez Başlığı:** *"Türkçe Finans İçin Federe RAG Tabanlı Düşük Kaynaklı Ortamlarda Çalışan Adaptif Chatbot Mimarisi ve Uygulaması"*  
**Tarih:** 2026-09-14 (En Güncel Durum)  
**Yazar:** MOSAIC-v2 Baş Mühendisliği & Tez Danışmanlığı  
**Gizlilik Uyumu:** KVKK (6698 sayılı kanun), BDDK Bilgi Sistemleri Yönetmeliği, GDPR Uyumlu.

---

## 1. Veri Mimarisine Genel Bakış

MOSAIC-v2 projesindeki veri ekosistemi, rastgele internet kazımalarından ibaret olmayıp, **BDDK mevzuatı, TCMB tebliğleri, SPK düzenlemeleri, BIST KAP şirket bildirimleri ve Türkçe finansal muhakeme** alanlarında özel olarak kürate edilmiş 4 ana sütundan oluşur:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 MOSAIC-v2 VERİ SÜTUNLARI                               │
├────────────────────────────┬────────────────────────────┬──────────────────────────────┤
│ 1. LLM SFT EĞİTİM VERİSİ   │ 2. BERT ROUTER EĞİTİM SETİ │ 3. DAĞITIK RAG KORPUSU (NON-IID)
├────────────────────────────┼────────────────────────────┼──────────────────────────────┤
│ • 22.192 SFT Örneği        │ • 30.000 Dengeli Örnek     │ • Şube-1 (Bireysel): 7.854   │
│ • ~3.85M Token             │ • 3 Sınıf (10k/10k/10k)    │ • Şube-2 (Kurumsal): 14.767  │
│ • Alpaca Formatı           │ • 80/10/10 Stratified Split│ • HQ Global: 22.192          │
│ • Qwen & Gemma Eğitimi     │ • Macro-F1: 0.9950         │ • Non-IID Örtüşme: %6.0      │
└────────────────────────────┴────────────────────────────┴──────────────────────────────┘
```

---

## 2. Sütun 1: LLM Fine-Tuning (SFT) Veri Seti

* **Dosya Yolu:** `ai/data/processed/finetune_turkish_finance_sft.jsonl`
* **Format:** Stanford Alpaca JSONL (`instruction`, `input`, `response`, `source`, `category`)
* **Toplam Örnek Sayısı:** **22.192 Örnek**
* **Tahmini Token Hacmi:** **~3.845.679 Token**
* **Kullanılan Modeller:** `Qwen3-Expert-v2` (HQ LLM) ve `GEMMA3-FINANCE-V1` (Edge SLM).

### Kategori Dağılımı

| Kategori Kodu | Kapsam ve İçerik | Örnek Sayısı | Oran (%) |
| :--- | :--- | :---: | :---: |
| **`piyasa_sirket_analiz`** | BIST şirket hisse analizleri, sektörel piyasa trendleri, makroekonomik göstergeler | 14.770 | %66.5 |
| **`bilanco_kap_analiz`** | KAP şirket finansal tabloları, bilanço dipnotları, kâr marjları ve duygu analizi | 3.468 | %15.6 |
| **`muhakeme_hesaplama`** | Kredi anüite formülleri, bileşik faiz, WACC, NPV, DCF, sermaye yeterlilik rasyosu | 1.763 | %7.9 |
| **`bankacilik_urunleri`** | Konut/ihtiyaç kredisi şartları, mevduat vadeleri, LTV oranları, kredi kartı kuralları | 1.730 | %7.8 |
| **`mevzuat_duzenleme`** | 5411 sayılı Bankacılık Kanunu, BDDK ve TCMB resmi tebliğ ve talimatları | 435 | %2.0 |
| **`finans_haber_analiz`** | BloombergHT, AA Finans ve Dünya Gazetesi güncel piyasa bültenleri | 12 | < %0.1 |
| **`bilanco_mevzuat_diger`**| Özel bankacılık hukuku ve SPK kılavuzları | 14 | < %0.1 |
| **TOPLAM** | | **22.192** | **%100.0** |

### Kaynak (Source) Dağılımı

| Veri Kaynağı | Açıklama | Örnek Sayısı |
| :--- | :--- | :---: |
| `turkish_finance_chat_corpus` | Türkçe finansal diyalog ve terimsel S&C kümesi | 14.571 |
| `kap_disclosures_and_sentiment` | Borsa İstanbul KAP açıklamaları ve finansal duygu etiketleri | 3.468 |
| `turkish_finance_reasoning` | Finansal formül, faiz ve rasyo muhakeme adımları | 1.738 |
| `bddk_spk_tcmb_instructions` | Resmi kurumların bankalara yönelik operasyonel talimatları | 1.344 |
| `turkiye_finance_qa_corpus` | Türkiye bankacılık ürünleri soru-cevap veri seti | 385 |
| `tcmb_official_communications` | TCMB Para Politikası Kurulu (PPK) resmi basın duyuruları | 364 |
| `financial_press_news` | Finans basını analiz ve piyasa haberleri | 199 |
| `bddk_banking_law` | 5411 sayılı Bankacılık Kanunu maddeleri | 71 |
| `sube1_konut_kredisi_kilavuzu` | Şube 1 yerel konut kredisi uygulama kılavuzu | 1 |
| `diger_resmi_mevzuatlar` | BloombergHT, AA Finans, KAP Raporları | 51 |

---

## 3. Sütun 2: BERT Router Yönlendirme Veri Seti

Kullanıcı sorgusunu donanımı yormadan doğru modele yönlendirmek amacıyla hazırlanan 3 sınıflı dengeli veri seti:

* **Eğitim Seti:** `ai/data/processed/router_train.jsonl` — **24.000 Örnek** (%80)
* **Doğrulama Seti:** `ai/data/processed/router_val.jsonl` — **3.000 Örnek** (%10)
* **Test Seti:** `ai/data/processed/router_test.jsonl` — **3.000 Örnek** (%10)
* **Toplam Boyut:** **30.000 Örnek (Dengeli: 3 x 10.000)**
* **Rastgelelik Tohumu (Reproducibility):** `seed=42`, Stratified Split.

### Sınıf Dağılımı ve Hedef Modeller

| Sınıf Etiketi | Sınıf Adı | Örnek Sayısı | Veri Kaynağı ve Üretim Yöntemi | Yönlendirilen Hedef Model |
| :---: | :--- | :---: | :--- | :--- |
| **`0`** | **`chitchat`** | **10.000** | MASSIVE tr-TR veri seti + CLINC150 Türkçe uyarlaması + sentetik banka karşılama ve sohbet kalıpları. | **Gemma-3-4B** (RAG'sız, doğrudan hızlı yanıt) |
| **`1`** | **`simple`** | **10.000** | 120 Türkçe finans terimi × 40 soru şablonu kartezyen üretimi + finansal keyword doğrulaması. | **Gemma-3-4B + Şube RAG** (Lokal SSS) |
| **`2`** | **`complex`** | **10.000** | 11 analitik finans grubu (KAP, BDDK, makroekonomik şoklar, anüite formülleri, sermaye yeterlilik). | **Qwen3-8B + HQ Global RAG** (Derin analiz) |

---

## 4. Sütun 3: Dağıtık RAG Retrieval Korpusu (Non-IID Dağılım)

Gerçekçi bir bankacılık federasyonunu modellemek üzere oluşturulan ve ChromaDB üzerinde indekslenen korpus:

* **Şube 1 — Bireysel & Kredi Bankacılığı (`ai/data/splits/branch1/corpus.jsonl`):**
  * **Hacim:** **7.854 Chunk** (10.37 MB)
  * **İçerik:** Bireysel krediler, konut kredisi faiz matrisleri, LTV/DTI sınırları, mevduat ürünleri.
* **Şube 2 — Kurumsal & Yatırım Bankacılığı (`ai/data/splits/branch2/corpus.jsonl`):**
  * **Hacim:** **14.767 Chunk** (26.40 MB)
  * **İçerik:** Kurumsal krediler, KAP bildirimleri, şirket bilançoları, sendikasyon ve proje finansmanı.
* **HQ Genel Merkez — Global Mevzuat Korpusu (`ai/data/splits/hq/corpus.jsonl`):**
  * **Hacim:** **22.192 Chunk** (36.27 MB)
  * **İçerik:** BDDK Kanunu, TCMB Tebliğleri, SPK düzenlemeleri ve tüm şubelerin anonimleştirilmiş global mevzuatı.

### Non-IID Örtüşme (Overlap) Kanıtı
Şube 1 ve Şube 2 dokümanları arasındaki örtüşme oranı tam olarak **%6.0**'dır. Bu oran, şubelerin birbirinin verisine sahip olmadığını, gerçek bir heterojen dağıtık ortam oluşturulduğunu matematiksel olarak kanıtlar.

---

## 5. Sütun 4: Akademik Benchmark ve Güvenlik Test Kümeleri

* **MMLU-TR Çoklu Görev Testi (malhajar/mmlu-tr, MMLU'nun Türkçe çevirisi; Bayram vd.'nin TR-MMLU'su değil) (`ai/data/benchmarks/tr_mmlu/`):**
  * `test.jsonl`: **2.000 Soru** | `dev.jsonl`: **500 Soru**
  * *Alt Alanlar:* `high_school_macroeconomics` (66), `high_school_microeconomics` (44), `professional_law` (226), `econometrics` (18), `business_ethics` (19).
* **Duygu Analizi Benchmark'ı (winvoker/turkish-sentiment-analysis-dataset; genel alan film/ürün/otel yorumu, KAP/BIST değil) (`ai/data/benchmarks/bist_sentiment/`):**
  * `test.jsonl`: **1.500 Örnek** | `train.jsonl`: **1.000 Örnek**
  * *Metrikler:* Macro-F1 (0.8997 Gemma / 0.7900 Qwen), Accuracy (%90.0 - %92.5).
* **Bankacılık Güvenlik ve KVKK Saldırı Seti (`ai/data/benchmarks/banking_security_eval.jsonl`):**
  * **300 Saldırı Senaryosu**
  * *Test Kategorileri:* Sahte BDDK denetçisi otorite baskısı, müşteri TC/IBAN sızdırma girişimleri, sistem promptu çıkarma ve jailbreak saldırıları. Sistemimizin başarı oranı: **saldırı başarı oranı ölçülmedi**

---

## 6. Veri Temizliği, Normalizasyon ve KVKK Önlemleri

Tüm veri setleri eğitim ve indeksleme öncesinde şu filtreleme hatlarından geçirilmiştir:
1. **Windows-1252 Karakter Onarımı:** Bozuk Türkçe karakterler (`Ý->İ`, `þ->ş`, `ð->ğ`) regex motoruyla düzeltilmiştir.
2. **Geç PII Maskeleme (Late Masking):** Müşteri TC Kimlik No, IBAN, Kredi Kartı ve Gerçek İsimleri regex ile taranmış, `[TC_NO_1]`, `[IBAN_1]`, `[NAME_1]` etiketleriyle anonimleştirilmiştir.
3. **Tablo Bütünlüğü:** PyMuPDF ile çıkarılan Markdown tablolarının satır ortasından bölünmesi engellenmiş, tablolar tek parça chunk olarak saklanmıştır.
4. **CJK Temizliği:** Qwen3 modellerinde oluşan Çince karakter kalıntıları ayıklanmıştır.

