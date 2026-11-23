# MOSAIC-v2 — KVKK ve BDDK Teknik Tedbir Eşlemesi

> **Kapsam ve sınır.** Bu belge, 6698 sayılı KVKK md. 12 (veri güvenliği), Kişisel Veri Güvenliği Rehberi
> (teknik ve idari tedbirler) ve BDDK *Bankaların Bilgi Sistemleri ve Elektronik Bankacılık Hizmetleri Hakkında
> Yönetmelik* (RG 15.03.2020, 31069) kapsamındaki **teknik** tedbirlerin MOSAIC-v2'de nasıl karşılandığını ve
> nasıl doğrulandığını gösterir. **Hukuki uyum** (aydınlatma, açık rıza/hukuki sebep, VERBİS kaydı, kişisel veri
> envanteri, saklama-imha politikası, veri işleyen sözleşmeleri, BS yönetişimi ve BDDK bağımsız denetimi) kurumun
> idari süreçleriyle sağlanır; bu yazılım tek başına "KVKK/BDDK uyumlu" sayılamaz. Doğru ifade: *"KVKK ve BDDK'nın
> öngördüğü teknik tedbirleri uygulayan ve bunları deneysel olarak doğrulayan bir mimari."*

## 1. Teknik tedbir eşleme tablosu

| # | Gereksinim (dayanak) | MOSAIC-v2'deki karşılığı | Doğrulama (`tools/compliance_check.py`) |
|---|---|---|---|
| 1 | Aktarımda şifreleme; düğümler arası karşılıklı kimlik doğrulama (KVKK md.12; BDDK BS Yön. – kimlik doğrulama, bilgi güvenliği) | Şube RAG servisleri TLS 1.2+ ve **zorunlu istemci sertifikası (mTLS)** ile çalışır (`backend/scripts/tls_serve.py`); kurum içi CA yalnızca yetkili düğümlere sertifika verir; HQ istemcisi `RAG_CLIENT_CERT/KEY` | Düz HTTP reddi, sertifikasız TLS reddi, mTLS ile 200 + protokol sürümü |
| 2 | Veri minimizasyonu, maskeleme (KVKK md.4 ölçülülük; Rehber: veri maskeleme) | **Kaynakta maskeleme**: şube, parçaları göndermeden önce maskeler (`BRANCH_PII_MASKING`), yer tutucular geri açılamaz (`[GİZLİ_TÜR]`); maskeleyici v3.1: ayraç toleranslı desen + TC/IBAN(mod-97)/Luhn sağlaması + ad sözlüğü + ad yayılımı | Kanarya sırlarının şube→HQ hattında açık görünmemesi |
| 3 | Kimlik doğrulama ve yetkilendirme, en az yetki, yetki matrisi (Rehber; BDDK BS Yön.) | Orkestratörde API anahtarı (yalnız SHA-256 özeti saklanır) + rol (analist / denetçi / yönetici) yetki matrisi (`backend/compliance/access.py`, `AUTH_REQUIRED=true`) | Anahtarsız 401, geçersiz anahtar 401, analistin imha denemesi 403 |
| 4 | İz kayıtları ve bütünlüğü (BDDK BS Yön. – iz kayıt mekanizması; Rehber: işlem kayıtları) | Her istek için HMAC-SHA256 imzalı, **hash zincirli** iz kaydı (çok süreçli yazımda `fcntl` kilidiyle tek doğrusal zincir): kim, ne zaman, hangi uç nokta, hangi belge kimlikleri, maskeli sorgunun özeti (`backend/compliance/audit.py`); ham sorgu/metin yazılmaz | Zincir doğrulaması, kurcalama tespiti, iz kaydında kişisel veri olmaması |
| 5 | Loglarda kişisel veri bulunmaması (Rehber) | Ham sorgu loglanmaz (maskeli sorgu loglanır), `ENABLE_PII_LOGGING=false` | Uygulama loglarında kanarya sırrı aranır |
| 6 | Ağ güvenliği / segmentasyon (BDDK BS Yön.) | HQ iç servisleri (yönlendirici, HQ RAG, LLM sunucuları) yalnızca `127.0.0.1`'de dinler (`BIND_HOST`); dışa açık uçlar: kimlik doğrulamalı orkestratör ve mTLS'li şubeler | LAN IP'sinden iç portların kapalı olması |
| 7 | Silme, yok etme, anonim hale getirme (KVKK md.7; Rehber) | `/erase`: ilgili kişi tanımlayıcısıyla (ör. TC) şubelerde kaynak kayıtlar ve HQ'daki damıtılmış kopyalar (`distill_*`) zincirleme silinir; işlem iz kaydına yazılır | Kanarya müşterisinin silinip sorgularda dönmemesi + iz kaydı |
| 8 | Saklama süresi ve imha (Rehber: saklama-imha politikası) | İz kayıtları `AUDIT_RETENTION_DAYS` (varsayılan 730 gün) sonunda imha edilir; imha işlemi de iz kaydına yazılır | Birim testi |
| 9 | Yurt içi birincil sistem, yurt dışına aktarım (BDDK BS Yön.; Bankacılık K. md.73; KVKK md.9) | Tüm modeller (Qwen3-8B, Gemma-3-4B, TurkEmbed, BERT) şirket içi sunucularda çalışır; harici yapay zekâ API'si yoktur | Kaynak kodda harici yapay zekâ API çağrısı taranır |
| 10 | Federe öğrenmede gizlilik (KVKK md.12) | Ham belge değil, yalnızca model güncellemeleri paylaşılır; L2 kırpma + Gauss mekanizması (`backend/federated/dp.py`) | `h2_dp_analysis.json` (gizlilik-fayda ödünleşimi) |

## 2. Bilinen sınırlılıklar (dürüst kayıt)

- **Ad-soyad tespiti** sözlük, etiket ve unvan ipuçlarına dayanır; bir kuralla bulunan ad metnin geri kalanında da maskelenir (v3.1). Tamamı büyük/küçük harfli adlar ve 'Soyad, Ad' sırası hâlâ kaçabilir (confirm kümesinde ad duyarlılığı 0.3875). Üretimde Türkçe NER modeliyle desteklenmelidir.
- **Bilgi damıtma** (`distill_knowledge`) şube parçalarını (artık maskelenmiş hâliyle) HQ'da kalıcı saklar; bu bir aktarım/saklama faaliyetidir ve saklama-imha politikasında tanımlanmalıdır.
- **Durağan veri şifrelemesi** (ChromaDB dizinleri) uygulama katmanında yapılmamıştır; VM disklerinde LUKS/BitLocker gibi tam disk şifrelemesi ile sağlanmalıdır.
- **İki bileşenli kimlik doğrulama** (BDDK, elektronik bankacılık müşterileri için) bu iç asistanın kapsamı dışındadır; kurum kimlik sağlayıcısı (SSO/MFA) ile entegrasyon gerekir.
- Diferansiyel gizlilikte ε=1 düzeyi model faydasını ciddi biçimde düşürmektedir; parametre seçimi kurumun risk iştahına göre belgelenmelidir.

## 3. Kurumun yapması gereken idari tedbirler (yazılım dışı)
Aydınlatma metni ve hukuki sebep analizi; VERBİS kaydı; kişisel veri envanteri ve bilgi varlık envanteri; saklama-imha
politikası; erişim yetki matrisinin onaylanması ve periyodik gözden geçirilmesi; anahtar/sertifika yönetimi prosedürü;
log izleme ve olay müdahale (72 saat ihlal bildirimi, Kurul kararı 2019/10); sızma testi ve BDDK bağımsız BS denetimi;
personel eğitimi ve gizlilik taahhütleri.


## 4. Doğrulama sonuçları (2026-10-15, iki şube tedbirli)

`tools/compliance_check.py`, `run_compliance_both.sh` içinde çalışan sisteme karşı (HQ + VM1 10.0.0.11 + VM2 10.0.0.12, ikisi de mTLS ve kaynakta maskeleme ile). Sonuç: **21/21 kontrol geçti**. Ham çıktı: `ai/experiments/results/kvkk_bddk_dogrulama_iki_sube.json`.

| Kontrol | Dayanak | Sonuç | Ayrıntı |
|---|---|---|---|
| Anahtarsız sorgu reddedilir | BDDK BS Yön. kimlik doğrulama; KVKK Rehberi erişim |  | 401 |
| Analist anahtarıyla sorgu | yetki matrisi |  | 200 |
| Analist imha yapamaz (en az yetki) | KVKK Rehberi yetki matrisi |  | 403 |
| Geçersiz anahtar reddedilir | kimlik doğrulama |  | 401 |
| branch_01: düz HTTP kabul edilmez | aktarımda şifreleme |  | reddedildi (RemoteProtocolError) |
| branch_01: istemci sertifikasız TLS reddedilir (mTLS) | BDDK karşılıklı kimlik doğrulama |  | reddedildi (RemoteProtocolError) |
| branch_01: mTLS ile erişim ve protokol | TLS 1.2+ |  | 200 TLSv1.3 TLS_AES_256_GCM_SHA384 |
| branch_02: düz HTTP kabul edilmez | aktarımda şifreleme |  | reddedildi (RemoteProtocolError) |
| branch_02: istemci sertifikasız TLS reddedilir (mTLS) | BDDK karşılıklı kimlik doğrulama |  | reddedildi (RemoteProtocolError) |
| branch_02: mTLS ile erişim ve protokol | TLS 1.2+ |  | 200 TLSv1.3 TLS_AES_256_GCM_SHA384 |
| HQ iç servis 8000 LAN'a kapalı | BDDK ağ güvenliği/segmentasyon |  | 10.0.0.10:8000 kapalı |
| HQ iç servis 8003 LAN'a kapalı | BDDK ağ güvenliği/segmentasyon |  | 10.0.0.10:8003 kapalı |
| HQ iç servis 8098 LAN'a kapalı | BDDK ağ güvenliği/segmentasyon |  | 10.0.0.10:8098 kapalı |
| HQ iç servis 8099 LAN'a kapalı | BDDK ağ güvenliği/segmentasyon |  | 10.0.0.10:8099 kapalı |
| Şube→HQ hattında açık sayısal sır yok (kaynakta maskeleme) | KVKK veri minimizasyonu / maskeleme |  | 0/120 sır açık; ad açık 0/30 (yalnızca mTLS+maskelemeli şube) |
| HQ iz kaydı zinciri doğrulanır | BDDK iz kayıt mekanizması (bütünlük) |  | 908 kayıt, bozuk=None |
| İz kaydında kişisel veri yok | KVKK veri minimizasyonu |  | 0 eşleşme |
| HQ uygulama loglarında kanarya sırrı yok | KVKK Rehberi: loglarda kişisel veri |  | 0 eşleşme |
| İmha talebi kaydı şubeden siler | KVKK md.7 silme/yok etme |  | 200 {"branch_01": {"silinen": ["canary_b1_0"], "adet": 1}, "branch_02": {"silinen": [], "adet": 0}, "hq": {"si |
| İmha işlemi iz kaydına işlendi ve zincir geçerli | hesap verebilirlik |  | 914 kayıt |
| Dış yapay zekâ API çağrısı yok (veri yurt içinde, şirket içinde işlenir) | BDDK yurt içi birincil sistem / KVKK md.9 |  | yok |

Ek ölçümler: kanarya testinde iki şube tedbirliyken şube→HQ hattında 0/120 sayısal sır ve 0/30 ad açık; orkestratör yanıtlarında 0/180 sızıntı (önceki yapılandırmada 120/120 ve 110/180). Ayrıntı: `docs/HIPOTEZ_DOGRULAMA_RAPORU.md` §3.

**Bu turda düzeltilenler:** (1) httpx 0.28'de `verify=<yol>` ile `cert=` yok sayılıyordu, mTLS istemcileri `SSLContext` ile yeniden yazıldı; (2) iz kaydı zinciri üç süreç aynı dosyaya yazdığında kopuyordu, `fcntl` kilidi eklendi; (3) ad yayılımı eklendi; (4) VM1'e de aynı tedbirler uygulandı.

Hukuki not: Bu tablo teknik tedbirlerin uygulandığını gösterir; "KVKK/BDDK uyumlu" hükmü kurumsal/idari tedbirler ve hukuki değerlendirme gerektirir (§3).
