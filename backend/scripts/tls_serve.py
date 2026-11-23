#!/usr/bin/env python3
"""MOSAIC servislerini TLS / mTLS ile başlatan sarmalayıcı (kodu değiştirmeden).
Kullanım: RAG_TLS_CERT=... RAG_TLS_KEY=... [RAG_TLS_CLIENT_CA=ca.crt] python backend/scripts/tls_serve.py backend/backend_api.py --service rag
RAG_TLS_CLIENT_CA verilirse istemci sertifikası ZORUNLUdur (mTLS, ssl.CERT_REQUIRED)."""
import os, runpy, ssl, sys
import uvicorn

_orig = uvicorn.run

def _run(app, **kw):
    cert, key, ca = os.getenv("RAG_TLS_CERT"), os.getenv("RAG_TLS_KEY"), os.getenv("RAG_TLS_CLIENT_CA")
    if cert and key:
        kw.update(ssl_certfile=cert, ssl_keyfile=key)
        if ca:
            kw.update(ssl_ca_certs=ca, ssl_cert_reqs=ssl.CERT_REQUIRED)
        print(f"[tls_serve] TLS etkin (mTLS={'evet' if ca else 'hayır'}) cert={cert}", flush=True)
    return _orig(app, **kw)

uvicorn.run = _run
target = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
runpy.run_path(target, run_name="__main__")
