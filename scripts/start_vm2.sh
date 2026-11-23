#!/usr/bin/env bash
# Start branch node 2 (branch RAG with mTLS + source-side PII masking, branch UI on :8005).
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/branch_start.sh" 2
