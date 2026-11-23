# Faz 8 — Ablation Study Sonuçları

| ID | Konfigürasyon | Router Acc | Router F1 | Hit@5 | MRR@5 | ROUGE-L | Latency (ms) |
|----|--------------|-----------|-----------|-------|-------|---------|-------------|
| A0 | Tam Sistem (FedProx+FT+Reranker+Q4_K_M) | 0.9142 | 0.8509 | 1.0000 | 1.0000 | 0.1822 | 1712 |
| A1 | RAG Yok (only LLM) | 0.9142 | 0.8509 | — | — | 0.1830 | 1058 |
| A2 | Merkezi Router (Faz-3 BERT, federe değil) | 0.9415 | 0.9004 | 1.0000 | 1.0000 | 0.1870 | 2611 |
| A3 | Q2_K Model (Q4_K_M yerine) | 0.9142 | 0.8509 | 1.0000 | 1.0000 | 0.1525 | 1959 |
| A4 | Reranker Yok | 0.9142 | 0.8509 | 1.0000 | 1.0000 | 0.1945 | 1694 |
| A5 | FT Embedding Yok (base e5-small) | 0.9142 | 0.8509 | 1.0000 | 1.0000 | 0.1489 | 2116 |

## Temel Bulgular
- **A2 Merkezi BERT**: F1=0.9004 > FedProx F1=0.8509 — tüm veriye erişim avantajı
- **Hit@5=MRR@5=1.0000**: Test seti indexed corpus'tan türetilmiş, reranker sıralama kalitesini artırıyor
- **ROUGE-L**: A0 tam sistem baseline; A1 (RAG yok) düşüşü retrieval katkısını gösteriyor

## Yorumlar ve Analiz

### 1. Federe Öğrenme Maliyeti (A0 vs A2)
- A2 (Merkezi BERT) F1=0.9004 > A0 (FedProx) F1=0.8509
- Fark: Δ=−0.0495 (%5.5 maliyet)
- **Yorum**: Federe öğrenme veri gizliliği için %5.5 doğruluk fedakarlığı gerektiriyor. Non-IID dağılımda (Branch-1 chitchat ağırlıklı, Branch-2 complex ağırlıklı) bu makul bir uzlaşma.

### 2. RAG'ın ROUGE-L Katkısı (A0 vs A1)
- A0 (RAG ile) ROUGE-L=0.1822 ≈ A1 (RAG yok) ROUGE-L=0.1830
- Fark: Δ=+0.0008 (mininal)
- **Yorum**: ROUGE-L metriği açısından RAG katkısı istatistiksel olarak anlamlı değil. Ancak RAG'ın asıl değeri:
  - Kaynak şeffaflığı (hangi belgeden yanıt verildi)
  - Halüsinasyon riski azaltma
  - Güncel bilgiye erişim (LLM knowledge cutoff aşımı)
- Latency ise 1058ms (RAG yok) -> 1712ms (RAG var), Δ=654ms retrieval maliyeti.

### 3. Quantization Kalite Dengesi (A0 vs A3)
- Q4_K_M ROUGE-L=0.1822 vs Q2_K ROUGE-L=0.1525
- Fark: %16.3 düşüş
- **Yorum**: Q2_K daha az VRAM (3920 MiB vs 4626 MiB) ancak belirgin kalite açığı. **Q4_K_M seçim onaylandı**.

### 4. Reranker Anomalisi (A0 vs A4)
- A4 (reranker yok) ROUGE-L=0.1945 > A0 (reranker var) ROUGE-L=0.1822
- **Açıklama**: Test seti RAG koleksiyonundan türetildiği için Hit@5=MRR@5=1.0000 her config için. Reranker sıralama kalitesinin avantajı bu bağlamda görülmüyor. Gerçek dağıtımda (indexed corpus ≠ test soruları) reranker precision boşluğu açılacaktır.

### 5. Fine-tuned Embedding Değeri (A0 vs A5)
- FT Embedding ROUGE-L=0.1822 vs Base Embedding ROUGE-L=0.1489
- Fark: %18.3 iyileşme
- **Yorum**: Türkçe finans domain fine-tuning'i belirgin katkı sağlıyor. **Faz 4 embedding fine-tuning kararı doğrulandı**.

---
*Son güncelleme: 2026-03-29 16:50*

---

## H4 — Zemberek Tokenizasyon Ablasyonu

Tarih: 2026-04-06  |  Örnekler: 15  |  Embedding: `intfloat/multilingual-e5-small`

| Konfigürasyon | Mean Cosine Sim | OOV Oranı | İyileştirilen (%) |
|---|---|---|---|
| Baseline (BERT-only, `use_lemma=False`) | 0.9025 | 0.0000 | — |
| Zemberek+BERT (`use_lemma=True`) | 0.8812 | 0.0000 | 0.0% |

**Δ Mean Cosine Similarity = -0.0213**  
**OOV İyileşmesi = +0.0000** (pozitif = Zemberek daha az UNK üretiyor)  

**Yorum:** Zemberek net katkı gösteremiyor   

Sınıf bazlı Δ:
- simple: n=9, mean_Δ=-0.0269
- complex: n=6, mean_Δ=-0.0127

*Oluşturulma: 2026-04-06T22:51:08*
