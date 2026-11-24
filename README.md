# MOSAIC: Privacy-Preserving Federated RAG for Turkish Banking

MOSAIC is a two-layer distributed retrieval-augmented generation (RAG) architecture for Turkish banking. It gives
staff organization-wide access to information without moving branch documents to headquarters, and it runs on
mid-range hardware with 4-bit quantized language models.

This repository contains the system code, the experiment drivers and the result files behind the article
*MOSAIC: A Privacy-Preserving Federated Retrieval-Augmented Generation Architecture for Turkish Banking Branches on
Edge Hardware* (M. T. Toy and A. A. Süzen, Isparta University of Applied Sciences). The article is derived from the
first author's master's thesis.

## Contents

- [Architecture](#architecture)
- [Main results](#main-results)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Models](#models)
- [Configuration](#configuration)
- [Running the system](#running-the-system)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Tests](#tests)
- [Data availability](#data-availability)
- [Citation](#citation)
- [License](#license)

## Architecture

MOSAIC separates two distributed mechanisms that are often combined in federated RAG proposals:

| Layer | When it runs | What crosses node boundaries |
|-------|--------------|------------------------------|
| Real-time cross-branch retrieval | on every query | masked top-k text chunks over mutual TLS |
| Scheduled federated learning (Flower) | on a schedule, for example nightly | LoRA adapter or model updates (FedAvg, FedProx, optional differential privacy) |

```
                  +----------------------------- Headquarters (HQ) -----------------------------+
 user --> UI ---> | BERT router :8000 --> orchestrator :8004 --> cross-encoder re-ranker         |
                  |       |                      |                                              |
                  |       |               HQ RAG :8003 (shared chunks only)                     |
                  |       v                                                                     |
                  | Gemma-3-4B :8099 / Qwen3-8B :8098 (llama.cpp, Q4_K_M)                       |
                  | Flower server :9080 (scheduled FedAvg / FedProx)                            |
                  +----------------+-----------------------------------------+------------------+
                                   | mTLS (TLS 1.3)                          | mTLS (TLS 1.3)
                  +----------------v-----------------+       +---------------v------------------+
                  | Branch-1 RAG :8001               |       | Branch-2 RAG :8002               |
                  | local ChromaDB, source-side PII  |       | local ChromaDB, source-side PII  |
                  | masking, audit log, Flower client|       | masking, audit log, Flower client|
                  +----------------------------------+       +----------------------------------+
```

Main components:

- **Query router.** A BERTurk classifier assigns each query to one of three classes (everyday, simple, complex). It
  is exported to ONNX INT8 (25.6 ms per query on CPU) and combined with keyword and confidence rules that escalate
  uncertain queries to the larger model.
- **Retrieval.** Each branch embeds its documents with MOSAIC-Embed, a `intfloat/multilingual-e5-small` model
  fine-tuned on query-to-passage pairs, and searches them with HNSW (ef = 200). Headquarters merges the candidates from
  all branches and re-ranks them with a cross-encoder.
  Headquarters stores only the chunks that all branches share.
- **Privacy safeguards.** Personal data are masked inside the branch service before any text leaves the branch
  (Turkish ID, IBAN, card and phone numbers with checksum validation, and person names). Services communicate over
  mutual TLS. Access is controlled with role-based API keys stored as SHA-256 hashes, events are written to an
  HMAC-SHA256 audit chain, and data-subject erasure is supported.
- **Federated learning.** Flower with FedAvg and FedProx trains LoRA adapters. An optional Gaussian mechanism adds
  differential privacy, with Rényi-DP accounting.

## Main results

Results of the final configuration reported in the article:

| Measurement | Result |
|-------------|--------|
| Exact match on 500 cross-branch questions, no retrieval vs. federated retrieval | 3.0% vs. 89.2% (centralized index: 89.6%, p = 0.69) |
| Same questions, MOSAIC-Embed vs. the base e5-small model in the live federated system | 89.2% vs. 86.2% (p = 0.0015) |
| Live system, 200 paraphrased questions: exact match / judged correct / judged faithful | 84.5% / 81.0% / 79.0% |
| Human evaluation of 100 answers (intermediate base-e5 run), two raters: inter-rater agreement on correctness | Cohen's kappa = 0.85 |
| Agreement between the LLM judge and the human raters on correctness | Cohen's kappa = 0.83 and 0.82 |
| Answers leaking synthetic customer secrets (canary test, 180 requests) | 110/180 before, 0/180 with source-side masking and mTLS |
| Regulation-derived technical controls (KVKK, BDDK) | 21 of 21 passed |
| TR-MMLU, 6,200 questions, ChatML prompt: Qwen3-8B vs. Gemma-3-4B | 53.82% vs. 37.79% |
| GPU memory of the Q4_K_M models | 5.75 GB (Qwen3-8B), 3.45 GB (Gemma-3-4B) |

Hypothesis outcomes: H1 (federated retrieval improves cross-branch accuracy) supported; H2 (privacy safeguards)
supported at the level of technical safeguards; H3 (4-bit deployment on mid-range hardware) supported; H4
(Zemberek-based morphological processing improves retrieval) rejected.

## Repository layout

```
backend/
  ai_engine.py              router, branch and HQ RAG services, orchestrator, PII masking (FastAPI)
  backend_api.py            API layer used by the user interfaces
  federated/                Flower server and clients, differential privacy, proof engine
  compliance/               API-key access control and audit chain
  client_setup/             branch-side Flower client and setup script
  certs/                    certificate templates only (keys are never committed)
  scripts/                  data, training, evaluation and maintenance tools
    hypothesis_tests/       original H1-H4 experiments (canary leakage, PII masking, DP analysis, token cost)
    q1_revision/            extended evaluation used in the article (E0-E10)
  tests/                    pytest suite
ai/
  models/                   training code for the router and embeddings, quantization benchmark
  tokenizer/                Zemberek wrapper, hybrid tokenizer, tokenizer benchmarks
  evaluation/               metrics and ablation runners
  experiments/              experiment logs and result files
    results/                thesis-stage results (H1-H4)
    results/q1/             results reported in the article
frontend/                   React (Vite) interface, Streamlit interface and web server
scripts/                    node start-up scripts and figure scripts
  experiments/              experiment drivers (q1/ holds one driver per article experiment)
tools/                      compliance checker, result compiler, llama.cpp LoRA conversion patch
docs/                       system handbook, deployment and VM guides, safeguard report, model cards
config/                     example API-key file
```

Most inline comments, log messages and the reports under `docs/` whose names end in `_tr` are written in Turkish.

## Requirements

- Python 3.12 on Linux or WSL2. In the testbed, headquarters ran Windows 11 with WSL2 and each branch was an Ubuntu
  virtual machine with 2 vCPUs and 10 GB of RAM.
- An NVIDIA GPU with at least 8 GB of memory at headquarters (the experiments used an RTX 5060 Ti).
- [llama.cpp](https://github.com/ggml-org/llama.cpp) built with CUDA, with the server binary at
  `ai/llama.cpp/build/bin/llama-server`.
- OpenSSL for the certificates.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt               # headquarters
pip install -r requirements-branch.txt        # branch nodes
pip install -r requirements-experiments.txt   # evaluation scripts
```

## Models

Model weights are not stored in this repository. The published models are:

| Model | Location |
|-------|----------|
| Qwen3-8B financial expert, GGUF (Q4_K_M) | [Toxotes/QWEN3_EXPERT_V2_GGUF](https://huggingface.co/Toxotes/QWEN3_EXPERT_V2_GGUF) |
| Gemma-3-4B finance, GGUF (Q4_K_M) | [Toxotes/GEMMA3_FINANCE_V1_GGUF](https://huggingface.co/Toxotes/GEMMA3_FINANCE_V1_GGUF) |
| BERT query router | [Toxotes/fin-bert-tr-router](https://huggingface.co/Toxotes/fin-bert-tr-router) |
| MOSAIC-Embed, retrieval embedding model (final configuration) | [Toxotes/mosaic-embed](https://huggingface.co/Toxotes/mosaic-embed) |
| Initial fine-tuned embedding model (first configuration, superseded) | [Toxotes/mosaic-turkembed-finetuned](https://huggingface.co/Toxotes/mosaic-turkembed-finetuned) |

Expected local paths are `ai/models/finetuned/*.gguf`, `ai/models/router/bert_router_best` and
`ai/models/embeddings/turkembed_v3` for MOSAIC-Embed
(`huggingface-cli download Toxotes/mosaic-embed --local-dir ai/models/embeddings/turkembed_v3`). The services select
the embedding model with `EMBEDDING_MODEL_KEY` (`mosaic_embed` or `e5small_base`). The initial fine-tuned model was
trained on instruction-response pairs and generalized worse than its base model to cross-branch questions; it is kept
only to reproduce the first-configuration results. Model cards are in `docs/model_cards/`.

## Configuration

1. Copy the example environment files and fill in the addresses of your nodes:

   ```bash
   cp .env.hq.example .env.hq              # on the headquarters machine
   cp .env.branch1.example .env.branch1    # on branch 1
   cp .env.branch2.example .env.branch2    # on branch 2
   ```

2. Generate a certificate authority and the server and client certificates. The full list of expected files is in
   `backend/certs/README.md`.

   ```bash
   HQ_LAN_IP=<hq-address> bash backend/scripts/generate_certs.sh
   ```

3. Create API keys if authentication is enabled. Keys are stored only as SHA-256 hashes in `config/api_keys.json`
   (see `config/api_keys.example.json` and `backend/compliance/access.py`).

The addresses `10.0.0.10` (headquarters), `10.0.0.11` (branch 1) and `10.0.0.12` (branch 2) that appear in scripts,
documentation and result files are example values for the laboratory network.

## Running the system

```bash
bash scripts/start_hq.sh     # headquarters: language models, router, HQ RAG, orchestrator, UI on :8005, Flower server
bash scripts/start_vm1.sh    # branch 1: mTLS RAG service with source-side masking and the branch UI
bash scripts/start_vm2.sh    # branch 2
```

The same commands are available as `make hq`, `make branch1` and `make branch2`. Detailed set-up instructions are in
`docs/deployment_guide.md`, `docs/vm_setup_guide.md` and `docs/system_handbook.md`.

## Reproducing the experiments

Each driver in `scripts/experiments/q1/` starts the services it needs, runs one experiment and writes JSON results to
`ai/experiments/results/q1/`. The drivers expect the headquarters and branch services of the testbed to be reachable.

| Experiment | Driver | Script | Output |
|------------|--------|--------|--------|
| Paraphrased question sets | `q1_01c_paraphrase_v2.sh` | `e0_paraphrase_questions.py` | `e0_paraphrases.json` |
| Retrieval benchmark (BM25, e5, BGE-M3, TurkEmbed4Retrieval, initial fine-tuned model, RRF, re-ranking) | `q1_02_retrieval.sh` | `e1_retrieval_baselines.py` | `e1_retrieval.json`, `e1_retrieval_me.json` |
| Cross-branch generation (H1) and component ablation | `q1_07a_gen_main.sh`, `q1_07b_gen_quant.sh` | `e2_crossbranch_generation.py` | `e2_generation*.json` |
| Federated embedding training and differential privacy | `q1_05_fl_embedding.sh` | `e3_fl_embedding_dp.py` | `e3_fl_embedding_qp.json` (query–passage pairs, reported in the paper); `--data ir` writes the earlier `e3_fl_embedding.json` |
| Router baselines and repeated FedAvg / FedProx | `q1_06_router.sh` | `e4_router_baselines_fl.py` | `e4_router.json` |
| Live end-to-end test with an LLM judge | `q1_04_e2e_judge.sh` | `e5_e2e_judge.py` | `e5_e2e_judge*.json` |
| Zemberek root reduction in retrieval (H4) | `q1_08_h4_zemberek.sh` | `e6_h4_zemberek_retrieval.py` | `e6_h4_zemberek.json`, `e6_h4_zemberek_me.json` |
| Flower between the branch machines | `q1_10_flower.sh` | `e7_flower_server.py`, `e7_flower_client.py` | `e7_flower_*.json` (`e7_flower_qp_*.json`: query–passage pairs, reported in the paper) |
| Diagnosis of the federated-central gap | `q1_09_gapdiag.sh` | `e8_fed_gap_diag.py` | `e8_fed_gap*.json` |
| Router on out-of-template questions | `q1_11_router_ood.sh` | `e9_router_ood.py` | `e9_router_ood.json` |
| Intermediate configuration: re-indexing with the base e5-small model | `q1_12_reindex.sh` | `reindex_e5base.py` | none |
| Re-evaluation of the intermediate configuration (exact match, judge, canary) | `q1_13_eval_v2.sh`, `q1_14_eval_v2b.sh` | E2, E5, E8, `h2_canary.py` | `*_v2.json` |
| TR-MMLU, two models and two prompt conditions | `q1_15_tr_mmlu.sh` | `e10_tr_mmlu.py`, `e10_summary.py` | `e10_tr_mmlu_*.json` |
| MOSAIC-Embed: training queries, training, retrieval evaluation | `q1_16_mosaic_embed_train.sh` | `e12a_gen_train_queries.py`, `e12b_train_turkembed_v2.py`, `e12c_eval_turkembed_v2.py` | `e12*.json` |
| MOSAIC-Embed: offline generation check with simulated federated retrieval | `q1_17_mosaic_embed_e13.sh` | `e13_embed_generation.py` | `e13_embed_generation.json` |
| Final configuration: re-indexing with MOSAIC-Embed | `q1_18_reindex_mosaic_embed.sh` | `reindex_e5base.py --model` | none |
| Final configuration: exact match, live test with judge, canary, paired comparisons | `q1_19_eval_mosaic_embed.sh` | E2, E5, `h2_canary.py`, `e14_mosaic_embed_summary.py` | `*_me.json`, `*_v3.json`, `e14_*.json` |
| Final configuration: re-runs after withdrawing the first domain-specific model (full arms, quantized base model, masking off, prompt-attack rate) and retrieval latency | none | E2, `e15_me_rerun_summary.py`, `e16_retrieval_latency_me.py` | `e2_generation_me_*.json`, `e15_*.json`, `e16_retrieval_latency_me.json` |
| Human evaluation of 100 answers by two raters | none (labels collected with `human_eval/annotation_form.html`) | `e11_human_eval.py` | `human_eval/human_eval_summary.json` |

The original hypothesis experiments are driven by `scripts/experiments/run_all_experiments.sh` and write their
results to `ai/experiments/results/`. The scripts that drew the article figures are in `scripts/figures/`.

## Tests

```bash
python -m pytest backend/tests -v
```

Tests that need running services are skipped automatically when the services are not available. Some test modules
import the full model stack and therefore need the packages in `requirements.txt`.

## Data availability

The branch corpora, the fine-tuning corpus and the router dataset are not distributed with this repository
(`ai/data/` is excluded); `ai/data/DATASET_CATALOG.md` lists their sources. The customer records used in the privacy
experiments are synthetic canaries generated for testing. The public benchmarks used are mmlu-tr, TR-MMLU
(Bayram et al., 2025) and a general-domain Turkish sentiment dataset.

## Citation

If you use this code, please cite the article. Until its publication details are available, the repository can be
cited as:

```bibtex
@software{toy_mosaic_2026,
  author = {Toy, Mehmet Taha and Süzen, Ahmet Ali},
  title  = {{MOSAIC}: A Privacy-Preserving Federated Retrieval-Augmented Generation Architecture for Turkish Banking},
  year   = {2026},
  url    = {https://github.com/tahatoy/mosaic-federated-rag}
}
```

Citation metadata is also provided in `CITATION.cff`.

## License

This project is released under the [MIT License](LICENSE). Third-party models, datasets and libraries keep their own
licenses.

## Disclaimer

MOSAIC is a research prototype. It implements technical safeguards only and makes no claim of full compliance with
KVKK, BDDK or any other regulation.
