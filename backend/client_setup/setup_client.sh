#!/bin/bash
# ============================================================
# setup_client.sh — VMware Ubuntu İstemci Kurulum Scripti
# Türkçe Adaptif Chatbot — Client Node
# Kullanım: bash setup_client.sh
# ============================================================

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log_info()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error()   { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

echo "=================================================="
echo "  Türkçe Adaptif Chatbot — Client Kurulum Scripti"
echo "=================================================="

# ---------- BÖLÜM 1: Sistem Bağımlılıkları ----------
log_info "Sistem paketleri güncelleniyor..."
sudo apt-get update -qq
# Python 3.12 ve gerekli araçlar kuruluyor
sudo apt-get install -y \
    python3.12 python3.12-venv python3.12-dev \
    python3-pip \
    build-essential \
    git curl wget \
    openssl \
    libssl-dev \
    poppler-utils \
    tesseract-ocr tesseract-ocr-tur \
    libmagic-dev

# ---------- BÖLÜM 2: Python Ortamı ----------
log_info "Python sanal ortamı oluşturuluyor (Python 3.12)..."
python3.12 -m venv .venv
source .venv/bin/activate

log_info "pip güncelleniyor..."
pip install --upgrade pip wheel setuptools -q

# ---------- BÖLÜM 3: Python Bağımlılıkları ----------
log_info "Python bağımlılıkları kuruluyor (requirements_client.txt)..."
REQ_FILE="requirements_client.txt"
if [ ! -f "$REQ_FILE" ] && [ -f "backend/client_setup/requirements_client.txt" ]; then
    REQ_FILE="backend/client_setup/requirements_client.txt"
fi
if [ ! -f "$REQ_FILE" ]; then
    log_error "requirements_client.txt bulunamadı! Lütfen repo kök dizininde olduğunuzdan emin olun."
fi
pip install -r "$REQ_FILE" -q
log_info "Bağımlılıklar kuruldu."

# ---------- BÖLÜM 4: Klasör Yapısı ----------
log_info "Klasör yapısı oluşturuluyor..."
mkdir -p certs backend/certs data/raw data/processed chroma_db logs config

# ---------- BÖLÜM 5: Sertifika Kontrolü ----------
log_info "Sertifika kontrolü yapılıyor..."
# backend/certs varsa kök certs'e otomatik eşle
if [ -f "backend/certs/ca.crt" ] && [ ! -f "certs/ca.crt" ]; then
    cp -r backend/certs/* certs/ 2>/dev/null || true
    log_info "Sertifikalar backend/certs klasöründen certs/ dizinine otomatik kopyalandı."
fi

if [ -f "certs/ca.crt" ] && [ -f "certs/client.crt" ] && [ -f "certs/client.key" ]; then
    chmod 600 certs/client.key 2>/dev/null || true
    chmod 644 certs/client.crt certs/ca.crt 2>/dev/null || true
    log_info "Sertifikalar doğrulandı ve izinler ayarlandı."
else
    log_warning "Sertifikalar eksik — test modunda verify=False ile devam edilebilir."
fi

# ---------- BÖLÜM 6: .env Dosyası ----------
log_info ".env dosyası kontrol ediliyor..."
if [ ! -f ".env" ]; then
    if [ ! -f ".env.client.example" ]; then
        log_error ".env.client.example bulunamadı!"
    fi
    cp .env.client.example .env
    log_warning ".env dosyası .env.client.example'dan oluşturuldu."
    log_warning "Lütfen .env dosyasını düzenle: nano .env"
    log_warning "SERVER_HOST ve BRANCH_ID alanlarını doldur!"
else
    log_info ".env dosyası zaten mevcut."
fi

# ---------- BÖLÜM 7: ChromaDB Test ----------
log_info "ChromaDB bağlantısı test ediliyor..."
python3 -c "
import chromadb
client = chromadb.PersistentClient(path='./chroma_db')
col = client.get_or_create_collection('test')
col.add(documents=['test'], ids=['test_1'])
client.delete_collection('test')
print('ChromaDB OK')
" && log_info "ChromaDB çalışıyor." || log_warning "ChromaDB test başarısız — kurulum devam ediyor."

# ---------- BÖLÜM 8: Embedding Modeli İndir ----------
log_info "Türkçe embedding modeli indiriliyor (paraphrase-multilingual-MiniLM-L12-v2)..."
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model.save('./models/embedding')
print('Embedding modeli kaydedildi: ./models/embedding')
"
log_info "Embedding modeli hazır."

# ---------- BÖLÜM 9: Systemd Servisi (Opsiyonel) ----------
log_info "Systemd servisi oluşturuluyor (opsiyonel)..."
VENV_PATH="$(pwd)/.venv"
PROJECT_PATH="$(pwd)"
SERVICE_FILE="/etc/systemd/system/chatbot-client.service"

sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Adaptif Chatbot Flower Client
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PROJECT_PATH
Environment=PATH=$VENV_PATH/bin
ExecStart=$VENV_PATH/bin/python client_setup/flower_rag_client.py
Restart=on-failure
RestartSec=10
StandardOutput=append:$PROJECT_PATH/logs/client.log
StandardError=append:$PROJECT_PATH/logs/client_error.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
log_info "Servis oluşturuldu: chatbot-client.service"
log_info "Başlatmak için: sudo systemctl start chatbot-client"
log_info "Otomatik başlat: sudo systemctl enable chatbot-client"

# ---------- ÖZET ----------
echo ""
echo "=================================================="
echo -e "${GREEN}  KURULUM TAMAMLANDI${NC}"
echo "=================================================="
echo "Sonraki adımlar:"
echo "  1. .env dosyasını düzenle: nano .env"
echo "  2. Sertifikaları kopyala (eksikse): scp ..."
echo "  3. Şube verisini data/raw/ klasörüne ekle"
echo "  4. Flower client'ı başlat:"
echo "     source .venv/bin/activate && python client_setup/flower_rag_client.py"
echo "     VEYA: sudo systemctl start chatbot-client"
echo "=================================================="
