---
language:
- tr
license: apache-2.0
tags:
- text-classification
- bert
- turkish
- federated-learning
- finance
- router
base_model: dbmdz/bert-base-turkish-cased
---

# MOSAIC — BERT Query Router (FedProx, Türkçe Finans)

Bu model, **MOSAIC** tez projesinin bir parçası olarak eğitilmiştir.  
Türkçe finans sorularını üç sınıfa yönlendiren bir **query router** modelidir.

## Model Detayları

| Özellik | Değer |
|---------|-------|
| Baz model | `dbmdz/bert-base-turkish-cased` |
| Görev | 3-sınıf metin sınıflandırma |
| Eğitim yöntemi | Federe öğrenme — FedProx (μ=0.01, 5 round) |
| F1 skoru | **0.9950** (macro, test seti) |
| Sınıflar | `chitchat (0)` · `simple (1)` · `complex (2)` |
| Dil | Türkçe |
| Eğitim verisi | 30.000 örnek (10.000/sınıf) |

## Sınıf Tanımları

- **chitchat (0):** Genel konuşma, selamlama, hava durumu — RAG gerektirmez
- **simple (1):** Şube düzeyinde basit finans soruları — Şube RAG + Gemma-3-4B
- **complex (2):** Karmaşık analiz, karşılaştırma — Global RAG + Qwen3-8B

## Kullanım

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="tahatoy/mosaic-bert-router",
    tokenizer="tahatoy/mosaic-bert-router",
)

result = classifier("Konut kredisi için gelir belgesi gerekli mi?")
# [{'label': 'LABEL_1', 'score': 0.997}]  -> simple
```

## Etiket Haritası

```python
id2label = {0: "chitchat", 1: "simple", 2: "complex"}
```

## Proje Bağlamı

Bu model, **"Türkçe Finans İçin Federe RAG Tabanlı Düşük Kaynaklı Ortamlarda Çalışan Adaptif Chatbot Mimarisi"** tez çalışmasının Faz 3 ve Faz 5 aşamalarında üretilmiştir.

- **Faz 3:** BERT central fine-tuning (F1=0.9950)
- **Faz 5:** FedAvg vs FedProx karşılaştırması -> FedProx seçildi (F1=0.8509 federe, 2 şube + 1 HQ)

Kaynak kod: [github.com/tahatoy/Mosaic](https://github.com/tahatoy/Mosaic)
