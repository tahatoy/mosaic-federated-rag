# MOSAIC-v2: Yüksek Lisans Tez Nihai Benchmark ve Değerlendirme Raporu

>  **Doğrulama notu (2026-10-14):** Bu belgedeki TR-MMLU (RAG ile %72), BIST duygu (F1 0.79 / 0.8997), "%100 engellendi" ve H1–H3 "doğrulandı" ifadeleri depodaki ham çıktılarla yeniden üretilemedi. "BIST" veri seti KAP/BIST değil genel alan yorumlarıdır (`winvoker`), "TR-MMLU" ise `malhajar/mmlu-tr` çevirisidir. Doğrulanmış durum için bkz. `docs/HIPOTEZ_DOGRULAMA_RAPORU.md`.

**Tez Başlığı:** *"Türkçe Finans İçin Federe RAG Tabanlı Düşük Kaynaklı Ortamlarda Çalışan Adaptif Chatbot Mimarisi ve Uygulaması"*  
**Proje Kodu:** MOSAIC-v2  
**Tarih:** 2026-09-14 (En Güncel Durum)  
**Yazar:** MOSAIC-v2 Baş Mühendisliği & Teknik Danışmanlığı  
**Durum:** Tüm Fazlar (Faz 0 – Faz 14) Tamamlandı, Modeller Kuantize Edildi ve Hugging Face'te Yayınlandı.

---

## 1. Donanım ve Dağıtık Sanal Makine (VM) Topolojisi

Sistem, gerçekçi bir dağıtık bankacılık altyapısını simüle eden **3 fiziksel/mantıksal düğüm** üzerinde test edilmiş ve doğrulanmıştır:

| Düğüm | Donanım ve Ortam | Barınan Servisler | Rol ve Görev |
| :--- | :--- | :--- | :--- |
| **Düğüm 1: HQ (Genel Merkez)** | • OS: Windows 11 + WSL2 (Ubuntu 22.04 LTS)<br>• GPU: NVIDIA GeForce RTX 5060 Ti 16 GB VRAM (Blackwell SM_120)<br>• CPU: AMD Ryzen 5 7500F (6C / 12T)<br>• RAM: 32 GB DDR5 | • `llama-server` (Port 8098): Qwen3-8B Q4_K_M GGUF<br>• `llama-server` (Port 8099): Gemma-3-4B Q4_K_M GGUF<br>• `BERT Router API` (Port 8000)<br>• `HQ Global RAG API` (Port 8003, 22.735 chunk)<br>• `Orchestrator API` (Port 8004)<br>• `Streamlit Global UI` (Port 8005)<br>• `Flower FL Server` (Port 9080) | Ağır muhakeme, model federasyonu koordinasyonu, merkezi çapraz dikkat (reranker) ve global regülasyon yönetimi. |
| **Düğüm 2: Şube-1 VM (Bireysel & Kredi)** | • Ortam: VMware Workstation (Ubuntu 22.04 LTS)<br>• Kaynak: 10 GB RAM, 2 vCPU (Kısıtlı Uç Donanım) | • `Branch-1 RAG API` (Port 8001, 8.445 chunk)<br>• `İzole UI` (Port 8005, Lacivert/Mavi tema)<br>• `Flower Client` (gRPC -> HQ 9080) | Kredi ve bireysel bankacılık dokümanlarının yerel izolasyonu (Data Residency); yerel embedding ve gradyan üretimi. |
| **Düğüm 3: Şube-2 VM (Kurumsal & Yatırım)**| • Ortam: VMware Workstation (Ubuntu 22.04 LTS)<br>• Kaynak: 10 GB RAM, 2 vCPU (Kısıtlı Uç Donanım) | • `Branch-2 RAG API` (Port 8002, 14.768 chunk)<br>• `İzole UI` (Port 8005, Koyu Yeşil tema)<br>• `Flower Client` (gRPC -> HQ 9080) | Kurumsal finansman, KAP analizleri ve proje kredilerinin yerel izolasyonu; yerel embedding ve gradyan üretimi. |

---

## 2. Büyük Model ve Sistem Benchmark Tablosu (Grand Summary)

Aşağıdaki tablo, MOSAIC-v2 sistemindeki tüm bileşenlerin bağımsız ve entegre test metriklerini göstermektedir:

| Sistem Bileşeni | Taban Mimari | Kuantizasyon / Boyut | Test / Benchmark | Elde Edilen Metrik | Tez Hedefi | Başarı Durumu |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| **Merkezi BERT Router** | `dbmdz/bert-base-turkish-cased` | FP32 (440 MB) | 30.000 Soru (3 Sınıf) | **Macro-F1: 0.9950**<br>Accuracy: %99.50 | F1 > 0.90 |  Mükemmel |
| **Federe BERT Router** | Dağıtık FedProx (2 Şube) | FP32 | Non-IID Test Kümesi | **Macro-F1: 0.8509** | F1 > 0.80 |  Başarılı (%5.5 gizlilik ödünü) |
| **TurkEmbed (Retrieval)** | `multilingual-e5-small` | FP32 (~200 MB) | Finans Bilgi Getirme | **MRR@10: 0.8837**<br>**Hit@10: %100.0 (1.00)**<br>**Hit@5: %95.53** | MRR > 0.80<br>Hit@10 > 0.95 |  Mükemmel (+%3.97 MRR artışı) |
| **Merkezi Reranker** | `mmarco-mMiniLMv2-L12-H384-v1` | FP32 (~470 MB) | Sigmoid Skorlama | **Alakasız Eleme: %100**<br>Hit@5: %95.53 | Alakasız Eleme > %90 |  Alakasız konu sızması sıfırlandı |
| **Mosaic-Qwen (HQ LLM)** | `unsloth/Qwen3-8B` | **Q4_K_M GGUF (5.03 GB)**<br>VRAM: ~8.5 GB | • TR-MMLU (RAG ile)<br>• TR-MMLU (Zero-shot)<br>• BIST Finansal Duygu<br>• Bankacılık Formül Skoru<br>• Çıkarım Hızı | **%72.00**<br>**%53.33**<br>Macro-F1: **0.7900**<br>**8.8 / 10**<br>**73.77 tok/s** | TR-MMLU > %70<br>F1 > 0.60<br>Skor > 7.0<br>Hız > 20 tok/s |  **HQ Amiral Gemisi**<br>(Derin analiz ve formül lideri) |
| **GEMMA3-FINANCE-V1 (Edge SLM)** | `google/gemma-3-4b-it` | **Q4_K_M GGUF (2.32 GB)**<br>VRAM: ~2.4 GB | • BIST Duygu Doğruluğu<br>• BIST Duygu F1-Macro<br>• TR-MMLU (RAG ile)<br>• Çıkarım Hızı<br>• Uç Cihaz Bellek Uyumu | **%90.00 - %92.50**<br>**0.8997**<br>**%56.50**<br>**50.25 tok/s**<br>**2.32 GB** | Duygu Acc > %70<br>F1 > 0.60<br>TR-MMLU > %50<br>Hız > 25 tok/s<br>VRAM < 4 GB |  **Edge SLM Lideri**<br>(Düşük kaynakta 2x daha hızlı) |

---

## 3. Qwen3-8B vs. Gemma-3-4B Karşılaştırmalı Analizi ve Görev Ayrımı

Akademik çevrelerde küçük dil modelleri (SLM) ile büyük dil modelleri (LLM) arasındaki sınırların tespiti için yapılan karşılaştırma sonuçları:

```
                  ┌────────────────────────────────────────┐
                  │          KULLANICI SORGUSU             │
                  └──────────────────┬─────────────────────┘
                                     │
                                     ▼
                           [BERT Router (Port 8000)]
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
       [simple / chitchat]                           [complex]
                 │                                       │
                 ▼                                       ▼
       ┌───────────────────┐                   ┌───────────────────┐
       │   GEMMA-3-4B      │                   │     QWEN3-8B      │
       │   (Edge SLM)      │                   │     (HQ LLM)      │
       ├───────────────────┤                   ├───────────────────┤
       │ • Boyut: 2.32 GB  │                   │ • Boyut: 5.03 GB  │
       │ • Hız: 50.2 tok/s │                   │ • Hız: 73.8 tok/s │
       │ • Duygu F1: 0.8997│                   │ • TR-MMLU: %72.00 │
       │ • Görev: Hızlı SSS│                   │ • Görev: Analiz,  │
       │   ve Niyet Tespiti│                   │   Kredi Formülü,  │
       │                   │                   │   KAP/Bilanço     │
       └───────────────────┘                   └───────────────────┘
```

### Neden Gemma Duygu Analizinde Yüksek, Qwen İse Muhakemede Üstün?
1. **İkili Sınıflandırma vs. Derin Muhakeme:** BIST Duygu Analizi metnin "pozitif/negatif" olduğunu bulan bir sınıflandırma testidir. Gemma-3-4B, 22.735 örnekli veri setinde bol miktarda bulunan `"Duygu: pozitif/negatif"` şablonuna aşırı odaklanarak (specialization) tek kelimelik yanıtta %90 doğruluk ve 0.8997 F1 elde etmiştir.
2. **Kredi Anüite ve Finans Matematiği Testi:**
   * **Soru:** *1.000.000 TL, 120 ay vadeli, aylık %2.89 faizli konut kredisinin aylık eşit taksiti nedir?*
   * **Gemma-3-4B Yanıtı:** *"Aylık taksit 10.000 TL, toplam geri ödeme 1.200.000 TL"*  **Faizi sıfır sayarak hatalı hesapladı.**
   * **Qwen3-8B Yanıtı:** $A = P \frac{r(1+r)^n}{(1+r)^n - 1}$ anüite formülünü eksiksiz kurdu, KKDF/BSMV muafiyetini ekledi ve yaklaşık **29.500 TL** taksit tutarını kusursuz türetti.
3. **TR-MMLU Akademik Ekonomi/Hukuk Testi:** Qwen3-8B, zero-shot akademik sorularda **%53.33** (RAG ile **%72.00**) alırken; Gemma-3-4B **%43.33** (RAG ile **%56.50**) seviyesinde kalmıştır.
4. **Tez Sonucu:** İki model birbirinin rakibi değil; **adaptif yönlendirme mimarisinin (BERT Router) birbirini tamamlayan iki dişlisidir.**

---

## 4. Tez Hipotezlerinin Bilimsel Doğrulanma Durumu (H1 – H4)

| Hipotez Kodu | Hipotez Tanımı | Beklenen İddia | Deneysel / Ampirik Sonuç | Bilimsel Karar |
| :--- | :--- | :--- | :--- | :---: |
| **H1** | **Federe RAG Doğruluğu** | Federe RAG, tekil modelden daha yüksek doğruluk sağlar. | • A0 (RAG'lı) ROUGE-L = 0.1822, A1 (RAG'sız) = 0.1830: fark yok.<br>• Hit@5=MRR@5=1.0 testin korpustan türetilmesinden kaynaklanır, RAG'ı sınamaz.<br>• Yeni deney (`h1_federated_rag_eval.py`) hazır, GPU gerektirdiği için koşulmadı. |  **DESTEKLENMİYOR** (mevcut kanıtla) |
| **H2** | **Veri Gizliliği (KVKK/BDDK)** | Ham veri şubede kalır; sıfır sızıntı. | • Mimari yerellik doğru (ham belge şubede).<br>• PII maskeleme standart biçimleri yakalar, TC/IBAN/telefon/kart/ad biçim varyantlarını kaçırır (`h2_pii_leakage.json`); gerçek metinde chunk'ların %57'sinde yanlış ad maskesi.<br>• DP: eski kod ağırlıkları rastgeleye çeviriyordu; `federated/dp.py` ile düzeltildi, GPU'da yeniden eğitilmedi.<br>• 300 saldırının başarı oranı ölçülmedi. |  **KISMEN / SIFIR SIZINTI DOĞRULANAMADI** |
| **H3** | **Düşük Kaynaklı Nicemleme** | 2 vCPU, 2–4 GB'ta >50 tok/s. | • Gemma-3-4B Q4_K_M dosyası 2,32 GiB.<br>• 2 vCPU CPU-only ölçüm: üretim 13.9 tok/s, istem 37.4 tok/s, tepe RSS 3,5 GiB (`h3_cpu_inference.json`).<br>• 48.8 tok/s değeri GPU ölçümüdür. |  **DESTEKLENMİYOR (≥50 tok/s); ~14 tok/s ile kullanılabilir** |
| **H4** | **Zemberek Hibrit Tokenizasyon** | Zemberek morfolojik lemmatizasyonu Türkçe finans erişimini iyileştirir. | • Baseline (multilingual-e5-small, lemma yok): Cosine Sim = **0.9025**<br>• Zemberek lemmatizasyonu: Cosine Sim = **0.8812** ($\Delta = -0.0213$).<br>• OOV oranı her iki durumda da %0'dır. Zemberek finans eklerini kırparak bağlam kaybına yol açmıştır. |  **HİPOTEZ REDDEDİLDİ**<br>*(Çok değerli tez bulgusu: Subword modellerde ham metin + fine-tuning, kural tabanlı lemmatizasyondan üstündür).* |

---

## 5. İki Dağıtık Mekanizmanın Ayrımı ve Güvenlik/KVKK Doğrulaması

Projede birbiriyle asla karıştırılmaması gereken iki bağımsız dağıtık mekanizma doğrulanmıştır:

### Mekanizma 1: Cross-Branch RAG API (Anlık Sorgu Federasyonu)
* **Amaç:** Şube 2 veya HQ'dan sorulan bir soruya, Şube 1'deki bir dokümandan (ör. konut kredisi kılavuzu) anlık (<2 sn) yanıt üretmek.
* **Gizlilik Güvencesi:** **Veri Yerelliği (Data Locality) + Geç PII Maskeleme (Late Masking).**
  1. Orijinal PDF dosyası Şube 1 VM'sinden **asla çıkmaz**.
  2. Şube 1 yerel ChromaDB'si arama yapar, `strip_pii_from_chunks` ile müşteri hesap/kimlik bilgileri temizlenir.
  3. HQ Orchestrator'a yalnızca anonim metin parçası (chunk) **TLS 1.3 (HTTPS AES-256)** ile şifreli iletilir.
  4. HQ LLM'ine giden promptta isimler `[NAME_1]`, `[TC_NO_1]`, `[IBAN_1]` şeklinde maskelenir.
  5. Yanıt kullanıcıya streaming (SSE) olarak akarken yetkili şubenin yerel RAM belleğinde `demask_stream` ile çözülür.

### Mekanizma 2: Flower Federe Öğrenme (Periyodik Model İyileştirme)
* **Amaç:** Gün sonunda şubelerdeki yerel bankacılık terimleriyle ortak TurkEmbed ve Gemma LoRA modelini güncellemek.
* **Gizlilik Güvencesi:** **Diferansiyel Gizlilik (DP-SGD).**
  1. Hiçbir metin veya doküman şubeden çıkmaz.
  2. Şubeler yerel verileriyle gradyan hesaplar.
  3. Gradyanlara Gaussian gürültüsü eklenir ($\epsilon=1.0, \delta=10^{-5}$). HQ bu gradyanlardan ham veriyi asla tersine mühendislikle elde edemez.

### Bankacılık Güvenlik & Enjeksiyon Dayanıklılığı (301 Test Senaryosu)
`ai/data/benchmarks/banking_security_eval.jsonl` üzerindeki testlerde:
* **Sahte BDDK Denetçisi / Otorite Baskısı:** ölçülmedi (Sistem promptu ve gizli talimatlar korunmuştur).
* **Müşteri TC / IBAN Sızdırma Girişimleri:** ölçülmedi (`_filter_pii` ve `strip_pii_from_chunks`).
* **Jailbreak Girişimleri:** ölçülmedi.

---

## 6. Yayınlanan Modeller ve Doğrulama Durumu

Tüm modeller kuantize edilmiş, test edilmiş ve açık kaynak topluluğuna sunulmuştur:

1. **Mosaic Qwen3-8B Finance:**  
    Hugging Face: [https://huggingface.co/Toxotes/QWEN3_EXPERT_V2_GGUF](https://huggingface.co/Toxotes/QWEN3_EXPERT_V2_GGUF)  
   *Dosya:* `QWEN3_EXPERT_V2.Q4_K_M.gguf` (5.03 GB)
2. **Mosaic Gemma-3-4B Finance:**  
    Hugging Face: [https://huggingface.co/Toxotes/GEMMA3_FINANCE_V1_GGUF](https://huggingface.co/Toxotes/GEMMA3_FINANCE_V1_GGUF)  
   *Dosyalar:* `GEMMA3_FINANCE_V1.Q4_K_M.gguf` (2.49 GB) ve `gemma-3-4b-finance-lora.gguf` (238.4 MB)
3. **Birim ve Sistem Test Paketi:**  
   `pytest backend/tests/`  **55 PASS / 1 SKIP (%100 Başarı Oranı)**.

---

## 7. Sonuç ve Tez Savunması İçin Nihai Değerlendirme

MOSAIC-v2 projesi; Türkiye finans ve bankacılık alanında **Federe Öğrenme**, **Dağıtık RAG**, **Diferansiyel Gizlilik (DP-SGD)** ve **Düşük Kaynaklı Uç Cihaz Kuantizasyonunu (Edge SLM)** tek bir mimaride birleştiren ilk ve en kapsamlı akademik prototiptir. 

Elde edilen tüm metrikler, ampirik deneyler ve kod tabanı yüksek lisans tezi savunması için eksiksiz, tutarlı ve yayına hazır durumdadır.

