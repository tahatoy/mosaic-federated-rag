#!/bin/bash
# generate_certs.sh — Flower TLS Sertifikası Üretimi
# Sadece ana makinede çalıştır: bash scripts/generate_certs.sh

set -e

CERT_DIR="./certs"
mkdir -p "$CERT_DIR"

echo "[1/4] CA (Certificate Authority) oluşturuluyor..."
openssl genrsa -out "$CERT_DIR/ca.key" 4096
openssl req -new -x509 -days 3650 -key "$CERT_DIR/ca.key" \
  -out "$CERT_DIR/ca.crt" \
  -subj "/C=TR/ST=Istanbul/O=AdaptifChatbot/CN=FlowerCA"

echo "[2/4] Sunucu sertifikası oluşturuluyor..."
openssl genrsa -out "$CERT_DIR/server.key" 2048
openssl req -new -key "$CERT_DIR/server.key" \
  -out "$CERT_DIR/server.csr" \
  -subj "/C=TR/ST=Istanbul/O=AdaptifChatbot/CN=flower-server"

# SAN (Subject Alternative Name) — IP ve hostname için
cat > "$CERT_DIR/server_ext.cnf" << EOF
[v3_req]
subjectAltName = @alt_names
[alt_names]
IP.1 = 127.0.0.1
IP.2 = ${HQ_LAN_IP:-127.0.0.1}
DNS.1 = localhost
DNS.2 = flower-server
EOF

openssl x509 -req -days 3650 \
  -in "$CERT_DIR/server.csr" \
  -CA "$CERT_DIR/ca.crt" \
  -CAkey "$CERT_DIR/ca.key" \
  -CAcreateserial \
  -out "$CERT_DIR/server.crt" \
  -extensions v3_req \
  -extfile "$CERT_DIR/server_ext.cnf"

echo "[3/4] İstemci sertifikası oluşturuluyor..."
openssl genrsa -out "$CERT_DIR/client.key" 2048
openssl req -new -key "$CERT_DIR/client.key" \
  -out "$CERT_DIR/client.csr" \
  -subj "/C=TR/ST=Istanbul/O=AdaptifChatbot/CN=flower-client"

openssl x509 -req -days 3650 \
  -in "$CERT_DIR/client.csr" \
  -CA "$CERT_DIR/ca.crt" \
  -CAkey "$CERT_DIR/ca.key" \
  -CAcreateserial \
  -out "$CERT_DIR/client.crt"

echo "[4/4] İzinler ayarlanıyor..."
chmod 600 "$CERT_DIR"/*.key
chmod 644 "$CERT_DIR"/*.crt

echo ""
echo "[OK] Sertifikalar oluşturuldu: $CERT_DIR/"
echo "   ca.crt      → Tüm makinelere kopyalanacak (güven zinciri)"
echo "   server.crt/key → Sadece ana makinede kalır"
echo "   client.crt/key → GitHub repo üzerinden DEĞİL, güvenli kopyalama ile istemciye"
echo ""
echo "[UYARI] client.crt ve client.key dosyalarını SCP ile kopyala:"
echo "   scp certs/client.crt certs/client.key certs/ca.crt user@<VM_IP>:~/chatbot/certs/"
