---
language:
- tr
license: apache-2.0
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- turkish
- finance
- rag
- federated-learning
- generated_from_trainer
- dataset_size:1432
- loss:MultipleNegativesRankingLoss
base_model: intfloat/multilingual-e5-small
widget:
- source_sentence: Altın yatırımı mantıklı mı?
  sentences:
  - 'Temel Finans Okuryazarlığı: Fiyat Keşfi, Likidite Sağlama, Şeffaflık, Risk Diversifikasyonu,
    Ekonomik Büyümeye Katkıda Bulunma, Şirket Performansının İzlenmesi gibi temel
    finans kavramlarını anlamak gerekir.'
  - Pozitif yaklaşımın temel özellikleri; dışsal gözlem yerine işletme içi gözlemlere
    dayalı olması, analitik yöntemlerin kullanılması, başarı ve riskin birlikte ele
    alınması şeklinde özetlenebilir.
  - Altın, enflasyona karşı değer koruma aracı olarak görülür ancak faiz getirmez.
    Portföyünüzün %5-10'u kadar altın bulundurmak çeşitlendirme açısından faydalı
    olabilir. Tam altın, altın hesabı veya altın fonları gibi seçenekler mevcuttur.
- source_sentence: Bu terimi açıkla.
  sentences:
  - TİM İhracat Endeksi'ne dahil edilen firmaların sayısı 55'dir ve bu rakam yeni
    halka arzlarla birlikte daha da artması beklenmektedir.
  - Dıyıral f sans texrole er. trans sekteri,n.. don.şture” ve firârsa şlenie daha
    ozlı, esışılekilir ve verini bae getiren kır diz Tekneloj yı çer « İşte Baz oneTrlıc
    ta fnans tekrolo, e
  - Bir ülkenin belirli bir dönemde dış dünya ile yaptığı tüm ekonomik işlemlerin
    kaydıdır. Ülkeye giren toplam para ile ülkeden çıkan toplam para arasındaki dengeyi
    gösterir.
- source_sentence: Para piyasasının örgütlenmiş temel kurumu nedir?
  sentences:
  - Para piyasasının örgütlenmiş temel kurumu bankalardır. Bankalar, işletmelerin
    nakit ihtiyacını karşılamada en çok kullanılan finansman yöntemidir.
  - Faktoring, gerçekleştirilen hizmetlerin niteliğine ve üstlenilen risk şekline
    göre değişik türlerde sunulur. Faktoring türleri arasında yurtiçi (domestic) ve
    uluslararası (international) faktoring, fatura (invoice) faktoringi, açık (undisclosed)
    faktoring, sezonluk faktoring, toptan (bulk) faktoring, vade (maturity) faktoring,
    acente faktoring, full servis rücu edilemez faktoring ve açıklanmamış faktoring
    bulunur. Her faktoring türü, farklı hizmet ve risk üstlenme şekillerine sahiptir.
    Örneğin, fatura faktoringinde satıcı, faktoring kuruluşu nezdinde alacağını peşin
    tahsil etme olanağını elde eder, ancak faturanın ödenmemesi durumunda faktoring
    kuruluşu satıcıya yönelir. Açık faktoringde ise müşteriler faktoring işlemi ve
    faktor kuruluşundan habersizdirler. Uluslararası faktoring, yurt dışı işlemlerle
    ilgilenen faktoring kuruluşları tarafından gerçekleştirilir ve satıcılar, yabancı
    ithalatçının kredibilitesinin belirlenmesi konusundaki bilgi yetersizliği nedeniyle
    doğrudan tehdit edilirler.
  - Döviz kurlarındaki beklenmedik değişiklikler nedeniyle zarar etme ihtimalidir.
    Özellikle ithalat-ihracat yapan veya dövizle borçlanan şirketler için çok önemlidir.
- source_sentence: Kredi borcumu yapılandırdığımda kredi notum etkilenir mi?
  sentences:
  - Finansal kaldıraç etkisi, borçlanmanın öz sermaye kârlılığı üzerindeki etkisidir.
    Öz sermaye çarpanı ne kadar büyükse, öz sermaye kârlılığı da o kadar büyür. Bu,
    borçlanmanın bir işletmenin kârlılığını artırabileceğini gösterir.
  - Piyasa riski, piyasadaki dalgalanmalar ve değişikliklerin işletmenin finansal
    performansını olumsuz etkileme olasılığına karşılık gelen risktir.
  - Borcunuzu ödeyemez duruma gelip yasal takibe düşmeden yapılandırma yapmak, kredi
    notunuzu yasal takibe düşmek kadar kötü etkilemez. Aksine, borcunuza sahip çıktığınızı
    ve ödeme niyetinde olduğunuzu gösterir. Ancak yine de sicilinizde bir not olarak
    yer alabilir.
- source_sentence: 1970'li yıllarda finansal yönetimde hangi gelişmeler yaşanmıştır?
  sentences:
  - Başabaş noktası, işletmelerin üretim ve satış stratejilerini belirlemelerinde
    önemli bir role sahiptir. İşletmelerin, başabaş noktasını dikkate alarak üretim
    ve satış düzeylerini belirlemeleri, kârlılıklarını artırabilir ve zararlarını
    azaltabilir. Ayrıca, başabaş noktası, işletmelerin maliyet yapılarını ve fiyatlandırma
    stratejilerini değerlendirmelerine de yardımcı olur.
  - 1970'li yıllarda bu eğilim daha da artmıştır. Bu yılların diğer önemli bir özelliği
    de işletme değerini ve hissedarların - ortakların - servetini maksimum kılmada
    riskle düzeltilmiş modellerin geliştirilmiş olmasıdır.
  - 'Kredi Kartlarının Doğru Kullanılması ve Finansal Sorumluluk


    Kredi kartlarının doğru kullanımı ve finansal sorumlulukla ele alınması önemlidir.

    Kredi kartlarının yanlış kullanımı, finansal sorunlara yol açabilir.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer based on intfloat/multilingual-e5-small
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: finance val
      type: finance_val
    metrics:
    - type: cosine_accuracy@1
      value: 0.8547486033519553
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.8938547486033519
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.9329608938547486
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.9664804469273743
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.8547486033519553
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.29795158286778395
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.18659217877094966
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.09664804469273741
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.8547486033519553
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.8938547486033519
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.9329608938547486
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.9664804469273743
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.9030323757534492
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.883541722089208
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.8854218662631415
      name: Cosine Map@100
---

# MOSAIC — Türkçe Finans Embedding Modeli (multilingual-e5-small fine-tuned)

> **MOSAIC** tez projesinin Faz 4 çıktısı. `intfloat/multilingual-e5-small` baz modeli, 1.432 Türkçe finans QA çiftiyle fine-tune edilmiştir.
>
> Proje: *"Türkçe Finans İçin Federe RAG Tabanlı Düşük Kaynaklı Ortamlarda Çalışan Adaptif Chatbot Mimarisi"*

---

## Hızlı Başlangıç

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("tahatoy/mosaic-turkembed-finetuned")

query = "Konut kredisi faiz oranı nedir?"
passages = ["Bankamız konut kredilerinde...", "Diğer bilgi..."]
embeddings = model.encode([query] + passages)
```

---

## SentenceTransformer based on intfloat/multilingual-e5-small

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) <!-- at revision c007d7ef6fd86656326059b28395a7a03a7c5846 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the  Hub
model = SentenceTransformer("tahatoy/mosaic-turkembed-finetuned")
# Run inference
sentences = [
    "1970'li yıllarda finansal yönetimde hangi gelişmeler yaşanmıştır?",
    "1970'li yıllarda bu eğilim daha da artmıştır. Bu yılların diğer önemli bir özelliği de işletme değerini ve hissedarların - ortakların - servetini maksimum kılmada riskle düzeltilmiş modellerin geliştirilmiş olmasıdır.",
    'Kredi Kartlarının Doğru Kullanılması ve Finansal Sorumluluk\n\nKredi kartlarının doğru kullanımı ve finansal sorumlulukla ele alınması önemlidir.\nKredi kartlarının yanlış kullanımı, finansal sorunlara yol açabilir.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.4738,  0.0634],
# [ 0.4738,  1.0000, -0.0332],
# [ 0.0634, -0.0332,  1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Dataset: `finance_val`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value     |
|:--------------------|:----------|
| cosine_accuracy@1   | 0.8547    |
| cosine_accuracy@3   | 0.8939    |
| cosine_accuracy@5   | 0.933     |
| cosine_accuracy@10  | 0.9665    |
| cosine_precision@1  | 0.8547    |
| cosine_precision@3  | 0.298     |
| cosine_precision@5  | 0.1866    |
| cosine_precision@10 | 0.0966    |
| cosine_recall@1     | 0.8547    |
| cosine_recall@3     | 0.8939    |
| cosine_recall@5     | 0.933     |
| cosine_recall@10    | 0.9665    |
| **cosine_ndcg@10**  | **0.903** |
| cosine_mrr@10       | 0.8835    |
| cosine_map@100      | 0.8854    |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 1,432 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              |
  | details | <ul><li>min: 5 tokens</li><li>mean: 15.06 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 23 tokens</li><li>mean: 72.26 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                  | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Kullanıcının yatırım fonu hakkındaki sorusunu yanıtla.</code>                                                                                                         | <code>Yatırım fonları, içerdikleri varlıklara ve yatırım stratejilerine göre çeşitlenir. Başlıca türleri şunlardır:<br><br>- **Hisse Senedi Fonları:** Portföyünün en az %80'i hisse senetlerinden oluşur.<br>- **Borçlanma Araçları Fonları:** Genellikle düşük riskli ve sabit getirili menkul kıymetlere (tahvil, bono) yatırım yapar.<br>- **Karma Fonlar:** Hisse senedi, tahvil gibi farklı araçları bir arada bulundurur.<br>- **Para Piyasası Fonları:** Kısa vadeli ve likiditesi yüksek araçlardan oluşur.<br>- **Katılım Fonları:** Faizsiz finans ilkelerine uygun varlıklara (sukuk, katılım hisseleri) yatırım yapar.<br>- **Borsa Yatırım Fonları (ETF):** Belirli bir borsa endeksini takip eder ve hisse senedi gibi anlık alınıp satılabilir.<br>- **Serbest Fonlar:** Yüksek risk alabilen nitelikli yatırımcılara yönelik esnek fonlardır.</code>                                                                                                                                                                                    |
  | <code>92 gün vadeli, 500.000 nominal değerli bir hazine bonosu için yatırımcının beklediği getiri oranı 9,614 ise bu hazine bonosu hangi fiyattan satın alınmalıdır?</code> | <code>Hazine bonosunun üzerinde yazılı değer nominal değeri gösterir ve bu değer bononun vade tarihindeki (gelecekteki) değeridir. Dolayısıyla nominal değer bononun gelecek değeridir. Soruda gelecekteki değeri belli olan bu bononun satın alma günündeki, diğer bir ifade ile bugunku değeri sorulmaktadır. Hazine Bonosunun Nominal Değeri (F) - 500.000, Vade (n) - 92 gün - 92/365 yıl, İskonto Oranı (i) - 9,614. Satış Fiyatı (P) - ? 500.000 / (1 + 0,09614*(92/365)) = 482.958. Yatırımcı 92 gün vadeli, 500.000 nominal değerli olan bu hazine bonosunu satış tarihinde 482.958'den satın alacaktır. Vade dolduğunda ise hazine bonosu iade edilerek 500.000 (Anapara) geri alınacaktır. Alış fiyatı ile nominal değer arasındaki fark kadar faiz geliri elde edilmiş olacaktır. Aradaki farkın 92 günlük getiriyi gösterdiğini unutmamak gerekir. Elde edilen faiz - 500.000 - 482.958 - 17.042. Faiz Oranı - 17.042 / 482.958 = 0,035 yada 3,53. 92 günlük faizin yıllık basit faizle karşılığı: 0,035 * (360/92) = 0,138 veya 1...</code> |
  | <code>Bütçe nasıl oluşturulur?</code>                                                                                                                                       | <code>Bütçe oluşturmak için ilk adım hedef belirlemektir. Kısa, orta ve uzun vadeli finansal hedeflerinizi belirleyin. Gelirlerinizi ve giderlerinizi listeleyin. Giderlerinizi zorunlu ve isteğe bağlı olarak ayırın. Gelirinizden zorunlu giderleri çıkarın, kalan kısmı harcama ve birikim için paylaştırın. 50/30/20 ya da 70/20/10 gibi oranlarla planlama yapabilirsiniz. Her ay giderlerinizi gözden geçirerek bütçenizi güncelleyin.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `fp16`: True
- `eval_strategy`: steps
- `per_device_eval_batch_size`: 32
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: True
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `eval_strategy`: steps
- `per_device_eval_batch_size`: 32
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | finance_val_cosine_ndcg@10 |
|:-----:|:----:|:--------------------------:|
| -1    | -1   | 0.8615                     |
| 1.0   | 45   | 0.8820                     |
| 2.0   | 90   | 0.9030                     |
| 3.0   | 135  | 0.9030                     |


### Framework Versions
- Python: 3.12.3
- Sentence Transformers: 5.3.0
- Transformers: 5.4.0
- PyTorch: 2.11.0+cu130
- Accelerate: 1.13.0
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->

---

## Proje Bağlamı

Bu model MOSAIC federe öğrenme mimarisinin üç katmanlı RAG'ında kullanılmaktadır:

| ChromaDB Koleksiyonu | Belge Sayısı | Port |
|----------------------|-------------|------|
| `finance_branch1` | 636 chunk | 8001 |
| `finance_branch2` | 988 chunk | 8002 |
| `finance_global` | 1.623 chunk | 8003 |

**Mimari:** `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker + hibrit arama (dense + BM25)

Kaynak kod: [github.com/tahatoy/Mosaic](https://github.com/tahatoy/Mosaic)