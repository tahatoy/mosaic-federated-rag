# VMware Ubuntu İstemci Kurulum Rehberi
## Türkçe Adaptif Federe Chatbot — Şube Kurulumu

### Gereksinimler
- VMware Workstation / Player
- Ubuntu 24.04 LTS ISO (Python 3.12 desteği için)
- 10 GB RAM, 2 CPU, 40 GB disk
- Ana makine ile aynı ağ segmenti
- (Önerilen) AMD EPYC veya Ryzen Pro işlemci → AMD SME donanım desteği

---

### Adım 1: VMware VM Oluşturma
1. VMware → "Create a New Virtual Machine"
2. Typical kurulum, Ubuntu 22.04 ISO seç
3. RAM: 10240 MB, CPU: 2, Disk: 40 GB
4. Network: Bridged (Ana makine ile iletişim için)
5. Kurulumu tamamla, snapshot al: "Temiz Kurulum"

---

### Adım 2: Ubuntu Temel Ayarları
```bash
# Sistem güncelle
sudo apt-get update && sudo apt-get upgrade -y

# SSH etkinleştir (VS Code Remote için)
sudo systemctl enable ssh && sudo systemctl start ssh

# IP'yi öğren
ip addr show | grep "inet " | grep -v 127.0.0.1

# AMD SME durumunu kontrol et (AMD işlemcilerde)
dmesg | grep -i "memory encryption" || echo "AMD SME bilgisi bulunamadı (Intel/diğer CPU)"
```

---

### Adım 3: GitHub'dan Projeyi Çek
```bash
cd ~
git clone https://github.com/<KULLANICI>/<REPO>.git chatbot
cd chatbot
```

---

### Adım 4: Kurulum Scriptini Çalıştır
```bash
bash client_setup/setup_client.sh
```

---

### Adım 5: Sertifikaları Kopyala (Ana Makineden)
```bash
# Ana makinede (WSL terminal):
scp certs/client.crt certs/client.key certs/ca.crt \
    <VM_KULLANICI>@<VM_IP>:~/chatbot/certs/
```

---

### Adım 6: .env Dosyasını Düzenle
```bash
nano .env
# SERVER_HOST= kısmına ana makinenin IP'sini gir
# BRANCH_ID= kısmına bu şubeye özgü ID gir (örn. branch_01)
# DP_EPSILON= Differential Privacy gizlilik bütçesi (varsayılan: 1.0)
# DP_DELTA=   DP toleransı (varsayılan: 1e-5)
# DP_SENSITIVITY= Gradyan kırpma eşiği (varsayılan: 1.0)
```

---

### Adım 7: Şube Servislerini Başlat (Tek Komut)
Sanal makinede hem Flower istemcisini hem de Chatbot arayüzünü tek komutla başlatabilirsiniz:
```bash
source .venv/bin/activate
python client_setup/start_client.py
```
- **Arayüz Adresi:** `http://<VM_IP>:8501`
- **Flower Adresi:** Arka planda `flower-server:9080` adresine bağlanır.

---

### Adım 8: Sunum Sırasında Veritabanını Sıfırlama
Tez sunumu sırasında testleri tekrarlamak için veritabanlarını şu şekilde temizleyebilirsiniz:

**Şube (Client) Tarafında:**
- UI arayüzünde sol menüdeki **" Veritabanını Sıfırla"** butonuna basın. Bu işlem o şubenin yerel ChromaDB koleksiyonunu tamamen temizler.

**Ana Makine (HQ) Tarafında:**
HQ veritabanını sıfırlamak için WSL terminalinde şu komutu kullanın:
```bash
# Servisleri durdurduktan sonra
rm -rf data/chromadb/global/*
```

---

### Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| Türkçe Karakter | UI içinde özel CSS (Inter font) yüklüdür, karakterler otomatik düzelir |
| Port 8501 Hatası | `killall -9 python` komutuyla eski süreçleri temizleyin |
| 404 Hatası | `.env` dosyasında `SERVER_LLM_PORT=8004` (Orchestrator) olduğundan emin olun |

---

### Gizlilik Mimarisi Notu

Bu proje **Differential Privacy (Gaussian Mekanizması)** kullanır:

- `fl.common.SecureAggregation` (homomorfik şifreleme tabanlı) tercih edilmez.
- `diffprivlib.mechanisms.Gaussian` ile her federe güncellemeye matematiksel gürültü eklenir.
- **ε (epsilon)**: Gizlilik bütçesi. Küçük değer → güçlü koruma, düşük model doğruluğu.
- **δ (delta)**: İhlal olasılığı toleransı. Veri kümesi büyüklüğünün tersinden küçük olmalı.
- **AMD SME**: AMD EPYC / Ryzen Pro işlemcilerde bellek içeriği donanım düzeyinde şifrelenir. Ek yazılım gerekmez; kernel modülü ile otomatik etkinleşir.

---

### KVKK/GDPR Uyumluluk Notları
- Şube verisi `data/` klasöründe tutulur, GitHub'a push edilmez (.gitignore)
- Flower iletişimi TLS ile şifrelenir, ham veri sunucuya gönderilmez
- Differential Privacy ile sunucuya gönderilen gradyanlar matematiksel olarak koruma altındadır
- `ENABLE_PII_LOGGING=false` varsayılan olarak kapalıdır
- ChromaDB dosyaları VM içinde kalır, yedekleme şifreli yapılır
