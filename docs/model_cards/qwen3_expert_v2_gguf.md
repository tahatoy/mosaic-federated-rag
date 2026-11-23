---
license: apache-2.0
language:
- tr
tags:
- finance
- turkish
- qwen3
- gguf
- quantized
---

# QWEN3_EXPERT_V2 GGUF

This is a GGUF quantized version of the **QWEN3_EXPERT_V2** model, specialized for the Turkish finance domain.

## Model Details
- **Architecture:** Qwen3 (8B equivalent)
- **Quantization:** Q4_K_M (Recommended for a good balance between speed and quality)
- **Domain:** Turkish Finance
- **Base Model:** Qwen3-8B (Finetuned with expert finance data)

## Usage with llama.cpp

```bash
./llama-cli -m QWEN3_EXPERT_V2.Q4_K_M.gguf -p "Faiz oranları hakkında bilgi verir misin?" -n 512
```

## About MOSAIC-v2
This model is part of the **MOSAIC-v2** project, an adaptive, federated RAG chatbot architecture specialized for the Turkish finance domain.
