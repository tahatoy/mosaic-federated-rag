# Certificates

Private keys and certificates are **not** distributed with this repository.
Generate a fresh CA and the server/client certificates for your own deployment:

```bash
HQ_LAN_IP=<hq-ip> bash backend/scripts/generate_certs.sh   # CA, Flower server certificate
```

Expected files (all git-ignored):

| File | Used by |
|------|---------|
| `ca.crt` / `ca.key` | institutional CA (keep `ca.key` offline) |
| `branch1.crt/.key`, `branch2.crt/.key` | branch RAG servers (mTLS, TLS 1.3) |
| `hq_client.crt/.key` | HQ orchestrator → branch RAG |
| `branch1_client.crt/.key`, `branch2_client.crt/.key` | branch UI → local branch RAG |
| `server.crt/.key` | Flower server |

`server_ext.cnf` is a template for the server certificate's subjectAltName.
