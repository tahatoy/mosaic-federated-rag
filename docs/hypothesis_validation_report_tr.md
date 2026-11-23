# MOSAIC-v2 — Hipotez (H1–H3) Doğrulama Raporu (v2)

**Tarih:** 2026-10-15 · **Düzenek:** HQ = Windows iş istasyonu/WSL2 (Ryzen 5 7500F, RTX 5060 Ti 16 GB, 10.0.0.10); Şube-1 VM 10.0.0.11:8001 ve Şube-2 VM 10.0.0.12:8002 (Ubuntu, kullanıcı mosaic2). Ölçüm sırası: önce (iki şube HTTP) → yalnız Şube-2 tedbirli (Şube-1 kontrol grubu) → iki şube tedbirli (kaynakta maskeleme v3.1 + mTLS + uyum kodu).
**İlke:** Her sayı `ai/experiments/results/` altındaki bir dosyadan okunur; tek kaynak `OZET_SONUCLAR.json`. Sonuç görüldükten sonra yapılan her değişiklik ve hipotez revizyonu açıkça işaretlenmiştir.

## Özet karar tablosu

| Hipotez | Karar | Dayanak |
|---|---|---|
| H1 federe RAG > tekil | **Desteklendi** (çapraz şube senaryosunda) | doğruluk %4.00 / %1.00 / **%75.50** (A/B/C), McNemar p<0.001 |
| H2 (öneri formu) ham veri şubede, yalnızca model güncellemeleri paylaşılarak KVKK/BDDK ile tutarlı gizlilik | **Desteklendi (teknik tedbirler düzeyinde)** | önce: hatta 120/120 sır, yanıtlarda %61.11 oranında sır; iki şube tedbirli: hatta 0/120 sır, 0/30 ad, uçtan uca 0/180; uyum 21/21. İdari tedbirler kapsam dışı. |
| H3 (öneri formu) 4-bit modeller 4–6 GB VRAM'de çalışır | **Desteklendi** | Qwen3-8B 5483 MiB, Gemma-3-4B 3287 MiB (tek başına, c=4096) |
| H4 (öneri formu) Zemberek destekli hibrit tokenizasyon token maliyetinde ve geri getirmede iyileşme sağlar | **Reddedildi** | kosinüs 0.9025 → 0.8812; morfem bölütlemeli hibrit token sayısı +39.04%, süre 1103 kat (`h4_token_maliyeti.json`) |

**Not:** Tez v5'te 1.5 bölümündeki hipotez ifadeleri tez öneri formuna (2.03.2025) göre yeniden yazıldı. v3'teki daha katı ifadeler ('sızıntıları tamamen engelleyecek, tam uyum'; '2 vCPU'da 50 token/s üzeri') ölçümlerle desteklenmiyordu; işlemci hızı (13.92 tok/s) tezde ek bulgu olarak raporlanır. Ayrıca `ai/tokenizer/hybrid_tokenizer.py` içindeki `ZemberekAnalyzer` yolu API uyumsuzluğu nedeniyle sözcükleri değiştirmeden döndürüyor (tanı: `h4_token_maliyeti.json` → `tani`).

## 1. Model kıyaslaması (Çizelge 4.5, n = 500 + 500)

| Model | İstem | TR-MMLU | Duygu doğr. | Duygu Macro F1 | Gecikme (s) |
|---|---|---|---|---|---|
| Qwen3-8B | ChatML ham | %47.40 | %27.80 | 0.3200 | 0.159 / 0.141 |
| Gemma-3-4B | ChatML ham | %38.35 (2 ayrıştırılamadı) | %19.00 | 0.2326 | 0.164 / 0.107 |
| Qwen3-8B | kendi şablonu | %44.00 | %87.00 | 0.8323 | 0.121 / 0.090 |
| Gemma-3-4B | kendi şablonu | %40.80 | %87.40 | 0.8342 | 0.092 / 0.077 |

Duygu test örneklemi 399 pozitif / 101 negatif; çoğunluk tabanı %79.80. ChatML istemi üç sınıflı (nötr dahil) olduğundan bu koşulda duygu sonuçları düşüktür. Veri kümesi `winvoker/turkish-sentiment-analysis-dataset` (genel alan); KAP/BIST değildir.
**Düzeltilen hata:** `run_all_experiments.sh` satır 74–75'te `A && echo || B && cp` önceliği `bench_qwen.json`'u Gemma sonucuyla eziyordu; kıyaslama `07b_bench_chatml` işiyle iki model için ayrı ayrı yeniden koşuldu (`bench_chatml_*.json`).

## 2. H1 — çapraz şube olgusal sorgu deneyi

`h1_federated_rag_eval.py`: yalnızca bir şubede bulunan sayısal cümlelerden 200 cloze sorusu, yanıtın **bulunmadığı** şubeden sorulur; LLM Gemma-3-4B (aynı istem).

| Kol | Doğruluk | ROUGE-L |
|---|---|---|
| A RAG yok | %4.00 | 0.0076 |
| B yalnız yerel şube | %1.00 | 0.0057 |
| C federe (2 şube + HQ yeniden sıralama) | **%75.50** | 0.0887 |

C–A: yalnız C doğru 143, yalnız A doğru 0; McNemar p = 1.8e-43; fark %95 bootstrap GA [0.65, 0.775]. C–B: 149/0, GA [0.685, 0.8].
Deney Şube-2 mTLS + kaynakta maskeleme ile yinelendi: 200 sorunun tamamında yanıtlar birebir aynı. İki şube tedbirliyken (`h1_federated_rag_mtls_iki_sube.json`) C = %75.00, 195/200 yanıt aynı; doğru yanıt sayısı 1 azaldı.
**Sınır:** sorular korpustan otomatik türetildi; başarı = sayısal değer eşleşmesi; şube korpuslarının bir kısmı LLM üretimi. Ablasyonda genel soru setinde RAG'ın ROUGE-L katkısı yok (0.1822 / 0.1830).

## 3. H2 — gizlilik

**Kanarya yöntemi** (`h2_canary.py`): her şubeye geçerli sağlama basamaklı TC, IBAN, kart, telefon içeren 15 sentetik müşteri dosyası (30 dosya, 120 sır). `wire` = şubenin HQ'ya döndürdüğü metin; `e2e` = orkestratörün nihai yanıtı (180 soru: her müşteri için kendi şubesinden ve çapraz şubeden 3'er soru).

| Ölçüm | Önce (iki şube HTTP) | Yalnız Ş-2 tedbirli: Ş-2 | Yalnız Ş-2 tedbirli: Ş-1 (kontrol) | İki şube tedbirli |
|---|---|---|---|---|
| Hatta açık sayısal sır | 120/120 | 0/60 | 60/60 | 0/120 |
| Hatta açık ad | — | 12/15 (v3) → 0/15 (v3.1) | — | 0/30 |
| Nihai yanıtta sır (/90 her kaynak şube) | Ş-1 70, Ş-2 40 | **0** (p = 1.8e-12) | 77 | Ş-1 0 (p = 1.7e-21), Ş-2 0 |

**Saldırı seti** (300 Türkçe istem, 12 kategori; dedektör alt sınır verir): ASR önce %4.35 → sonra %7.02, eşleştirilmiş McNemar p = 0.096 (anlamlı değil). Başarılı sayılanların çoğu sistem istemi tekrarı; `pii_leak` işaretli 2 yanıttan biri kullanıcının istemde verdiği (geçersiz sağlamalı) IBAN'ın yankısı, diğeri dedektör yanlış alarmı.

**Maskeleme işlevi** (dondurulmuş, geliştirmede kullanılmayan sentetik kümeler, varyant başına 300):

| Sürüm | heldout (seed 2026) | confirm (seed 777) | gerçek korpus yanlış alarm |
|---|---|---|---|
| v1 (git HEAD) | 0.4703 | 0.4545 | 0.5737 |
| v3 / v3.1 (konuşlandırılan) | 1.0000 | 0.8886 | 0.0097 |

v3.1 (2026-10-15) yalnızca **ad yayılımı** ekler: bir kuralla bulunan çok sözcüklü ad metnin başka yerlerinde de maskelenir. Bu değişiklik kanarya sonucunu (Şube-2'de 12/15 ad açık) gördükten sonra yapılmıştır; dondurulmuş kümelerde değerler değişmedi (gerileme yok). Kalan kaçırmalar: tamamı büyük/küçük harfli adlar ve 'Soyad, Ad' sırası.

**Diferansiyel gizlilik** (`h2_dp_analysis.py`): ε = 1, δ = 1e-5, σ = 4.8448; TurkEmbed ağırlıklarında gürültülü–özgün kosinüs medyanı 0.0006 → model faydası yok. Kod ağırlıklara gürültü ekliyor; tezde yazan DP-SGD değil.

**KVKK/BDDK teknik tedbir doğrulaması** (`tools/compliance_check.py`, çalışan sisteme karşı):

| Kontrol | Sonuç | Ayrıntı |
|---|---|---|
| Anahtarsız sorgu reddedilir |  | 401 |
| Analist anahtarıyla sorgu |  | 200 |
| Analist imha yapamaz (en az yetki) |  | 403 |
| Geçersiz anahtar reddedilir |  | 401 |
| branch_01: düz HTTP kabul edilmez |  | reddedildi (RemoteProtocolError) |
| branch_01: istemci sertifikasız TLS reddedilir (mTLS) |  | reddedildi (RemoteProtocolError) |
| branch_01: mTLS ile erişim ve protokol |  | 200 TLSv1.3 TLS_AES_256_GCM_SHA384 |
| branch_02: düz HTTP kabul edilmez |  | reddedildi (RemoteProtocolError) |
| branch_02: istemci sertifikasız TLS reddedilir (mTLS) |  | reddedildi (RemoteProtocolError) |
| branch_02: mTLS ile erişim ve protokol |  | 200 TLSv1.3 TLS_AES_256_GCM_SHA384 |
| HQ iç servis 8000 LAN'a kapalı |  | 10.0.0.10:8000 kapalı |
| HQ iç servis 8003 LAN'a kapalı |  | 10.0.0.10:8003 kapalı |
| HQ iç servis 8098 LAN'a kapalı |  | 10.0.0.10:8098 kapalı |
| HQ iç servis 8099 LAN'a kapalı |  | 10.0.0.10:8099 kapalı |
| Şube→HQ hattında açık sayısal sır yok (kaynakta maskeleme) |  | 0/120 sır açık; ad açık 0/30 (yalnızca mTLS+maskelemeli şube) |
| HQ iz kaydı zinciri doğrulanır |  | 908 kayıt, bozuk=None |
| İz kaydında kişisel veri yok |  | 0 eşleşme |
| HQ uygulama loglarında kanarya sırrı yok |  | 0 eşleşme |
| İmha talebi kaydı şubeden siler |  | 200 {"branch_01": {"silinen": ["canary_b1_0"], "adet": 1}, "branch_02": {"silinen": [], "adet": 0}, "hq": {"silinen": [] |
| İmha işlemi iz kaydına işlendi ve zincir geçerli |  | 914 kayıt |
| Dış yapay zekâ API çağrısı yok (veri yurt içinde, şirket içinde işlenir) |  | yok |

Toplam 21/21. Ayrıntı: `docs/KVKK_BDDK_TEKNIK_TEDBIRLER.md`.

## 4. H3 — düşük kaynaklı çıkarım

Gemma-3-4B Q4_K_M (2,32 GiB), llama.cpp b8557, yalnız CPU, 2 iş parçacığı, 2 vCPU / 3,8 GiB Linux VM (VMware şube VM'i değil): üretim **13.92 tok/s**, istem işleme 37.43 tok/s, tepe RSS 3.51 GiB. GPU (RTX 5060 Ti, llama-bench tg64): 107.95 tok/s. 50 tok/s eşiği CPU'da karşılanmıyor; modeller tez öneri formunda sabit olduğundan değiştirilmedi.

**Öneri formu ölçütü (4–6 GB VRAM)** — `12_h3_vram` işi, `h3_vram.json` (her model tek başına, -c 4096, -ngl 99, nvidia-smi farkı):

| Model | Dosya | VRAM | Servis üretim | llama-bench tg64 |
|---|---|---|---|---|
| Qwen3-8B Q4_K_M | 4.68 GiB | 5483 MiB (5.75 GB) | 77.98 tok/s | 74.26 tok/s |
| Gemma-3-4B Q4_K_M | 2.32 GiB | 3287 MiB (3.45 GB) | 112.45 tok/s | 115.85 tok/s |

Not: ölçülen kullanım; 6 GB'lık bir kısıt altında çalıştırma değil. Kuantizasyon NF4 değil, llama.cpp Q4_K_M.

## 5. Bu turda bulunan ve düzeltilen hatalar

1. **httpx 0.28 istemci sertifikasını yok sayıyordu** (`verify=<yol>` verilince `cert=` dikkate alınmıyor) → mTLS sunucusu bağlantıyı kesiyordu. Tüm istemcilerde `SSLContext` + `load_cert_chain` (düzeltme `backend/` istemcilerine işlendi).
2. **VM1 erişimi**: VM1, VM2'nin klonudur (kullanıcı mosaic2); `vm_deploy_both.sh` + `run_compliance_both.sh` ile aynı tedbirler uygulandı.
3. **İz kaydı zinciri kopuyordu**: üç süreç aynı dosyaya kendi bellek içi son özetiyle yazıyordu. `fcntl` dosya kilidi altında son özet diskten okunuyor (`backend/compliance/audit.py`).
3. **Geçersiz ölçümler ayıklandı** (`results/invalid/`): VM2'ye erişilemezken üretilen e2e, http kopyası olan sahte `h1_federated_rag_mtls.json`, LLM kapalıyken alınan 251 saldırı kaydı, ezilmiş `bench_qwen.json`.
4. `run_final2.sh`'de `cmd | tail && mark` ile başarısız adımlar 'tamam' işaretleniyordu; `run_final3.sh` çıkış kodunu ayrı denetler.
5. Ölçüm betiklerine ön kontrol (`_preflight`) eklendi: şube `/query` yanıt vermiyorsa ölçüm başlamaz (çıkış 3).

## 6. Yeniden üretim

Bu rapordaki ölçümler laboratuvar düzeneğine özgü dağıtım betikleriyle alınmıştır; bu betikler (VM erişim bilgileri içerdiği için) depoya alınmamıştır. Hipotez deneylerinin kendisi depodadır:

```
bash scripts/experiments/run_all_experiments.sh   # H1-H4 deneyleri (backend/scripts/hypothesis_tests)
bash scripts/experiments/run_extra_bench.sh       # dil modeli kıyaslaması (kendi sohbet şablonu)
python3 tools/compliance_check.py                 # 21 maddelik teknik kontrol listesi
python3 tools/compile_results.py                  # OZET_SONUCLAR.json
```
