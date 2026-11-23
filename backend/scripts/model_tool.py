#!/usr/bin/env python3
"""
MOSAIC-v2 Model Tool
====================
Consolidated CLI tool for model training, quantization, merging, and data processing.

Subcommands:
  - train-router      : Fine-tune BERT router
  - train-federated   : Federated learning simulation (FedAvg, FedProx)
  - quantize          : Quantize HF models to GGUF
  - export-onnx       : Export router to ONNX (FP32 & INT8)
  - merge-lora        : Merge LoRA weights into base model
  - rebuild-splits    : Clean and rebuild router data splits
"""

import argparse
import os
import sys
import json
import random
import subprocess
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import numpy as np
import torch
from loguru import logger

# --- Global Configurations ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "ai" / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "ai" / "models" / "router"
CKPT_DIR = PROJECT_ROOT / "ai" / "experiments" / "checkpoints"
LLAMA_CPP = PROJECT_ROOT / "ai" / "llama.cpp"

warnings.filterwarnings("ignore")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==============================================================================
# SUBCOMMAND: train-router
# ==============================================================================

def train_router(args):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, f1_score, accuracy_score
    from torch.utils.data import Dataset
    from transformers import (
        BertForSequenceClassification,
        BertTokenizerFast,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
        DataCollatorWithPadding,
    )

    BERT_MODEL = "dbmdz/bert-base-turkish-cased"
    LABEL_NAMES = ["chitchat", "simple", "complex"]
    MAX_LEN = 128
    BATCH_SIZE = 32
    LR = 2e-5
    WARMUP_RATIO = 0.1

    data_file = Path(args.train) if args.train else DATA_DIR / "router_train.jsonl"
    output_dir = Path(args.output_dir) if args.output_dir else MODEL_DIR / "bert_router_best"
    n_epochs = args.epochs if args.epochs else 5

    logger.info("BERT Router Fine-tuning")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"Training data: {data_file}")
    logger.info(f"Output: {output_dir}")

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return

    rows = [json.loads(l) for l in data_file.open(encoding="utf-8")]
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]

    if args.val and args.test:
        val_rows = [json.loads(l) for l in Path(args.val).open(encoding="utf-8")]
        test_rows = [json.loads(l) for l in Path(args.test).open(encoding="utf-8")]
        t_texts, t_labels = texts, labels
        v_texts, v_labels = [r["text"] for r in val_rows], [r["label"] for r in val_rows]
        te_texts, te_labels = [r["text"] for r in test_rows], [r["label"] for r in test_rows]
    else:
        t_texts, tmp_texts, t_labels, tmp_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        v_texts, te_texts, v_labels, te_labels = train_test_split(
            tmp_texts, tmp_labels, test_size=0.5, random_state=42, stratify=tmp_labels
        )

    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)

    class RouterDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    def encode(texts_list):
        return tokenizer(texts_list, max_length=MAX_LEN, truncation=True, padding=True, return_tensors="pt")

    train_ds = RouterDataset(encode(t_texts), t_labels)
    val_ds = RouterDataset(encode(v_texts), v_labels)
    test_ds = RouterDataset(encode(te_texts), te_labels)

    label_counts = Counter(t_labels)
    total = sum(label_counts.values())
    class_weights = torch.tensor(
        [total / (len(label_counts) * label_counts[i]) for i in range(len(LABEL_NAMES))],
        dtype=torch.float,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro")),
        }

    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL, num_labels=3,
        id2label={i: n for i, n in enumerate(LABEL_NAMES)},
        label2id={n: i for i, n in enumerate(LABEL_NAMES)},
    )

    train_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=42,
    )

    trainer = WeightedTrainer(
        model=model, args=train_args, train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()
    test_result = trainer.predict(test_ds)
    test_preds = np.argmax(test_result.predictions, axis=-1)
    logger.info("\n" + classification_report(te_labels, test_preds, target_names=LABEL_NAMES, digits=4))

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Model saved to {output_dir}")

# ==============================================================================
# SUBCOMMAND: train-federated
# ==============================================================================

def train_federated(args):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    from torch.utils.data import DataLoader, Dataset
    from sklearn.metrics import accuracy_score, f1_score

    set_seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 3
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    LR = 2e-5
    FEDPROX_MU = args.mu
    LOCAL_EPOCHS = args.local_epochs
    NUM_ROUNDS = args.rounds
    BASE_MODEL_DIR = Path(args.model_dir) if args.model_dir else MODEL_DIR / "bert_router_best"

    class RouterDataset(Dataset):
        def __init__(self, texts: List[str], labels: List[int], tokenizer):
            enc = tokenizer(texts, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
            self.ids = enc["input_ids"]
            self.masks = enc["attention_mask"]
            self.lbls = torch.tensor(labels, dtype=torch.long)
        def __len__(self): return len(self.lbls)
        def __getitem__(self, i): return {"input_ids": self.ids[i], "attention_mask": self.masks[i], "labels": self.lbls[i]}

    def load_branch_data(branch_id: int):
        by_class = {0: [], 1: [], 2: []}
        for line in (DATA_DIR / "router_train.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            lbl = r.get("label")
            txt = r.get("text", "").strip()
            if txt and lbl in (0, 1, 2): by_class[lbl].append(txt)
        ratios = {1: {0: 0.70, 1: 0.50, 2: 0.30}, 2: {0: 0.30, 1: 0.50, 2: 0.70}}
        rnd = random.Random(42 + branch_id)
        texts, labels = [], []
        for cls, ratio in ratios[branch_id].items():
            pool = by_class[cls][:]
            rnd.shuffle(pool)
            n = max(300, min(600, int(len(pool) * ratio * 0.15)))
            texts += pool[:n]; labels += [cls] * n
        return texts, labels

    def get_weights(model): return [v.cpu().numpy().copy() for v in model.state_dict().values()]
    def set_weights(model, weights):
        sd = model.state_dict()
        for k, w in zip(sd.keys(), weights): sd[k] = torch.tensor(w)
        model.load_state_dict(sd, strict=True)

    def load_model(weights=None):
        m = AutoModelForSequenceClassification.from_pretrained(str(BASE_MODEL_DIR), num_labels=NUM_CLASSES, local_files_only=True).to(DEVICE)
        if weights is not None: set_weights(m, weights)
        return m

    def local_train(global_weights, dataset, use_fedprox=False):
        model = load_model(global_weights)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=max(1, int(len(loader) * LOCAL_EPOCHS * 0.1)), num_training_steps=len(loader) * LOCAL_EPOCHS)
        if use_fedprox: global_tensors = [torch.tensor(w).to(DEVICE) for w in global_weights]
        model.train()
        for _ in range(LOCAL_EPOCHS):
            for batch in loader:
                opt.zero_grad()
                out = model(input_ids=batch["input_ids"].to(DEVICE), attention_mask=batch["attention_mask"].to(DEVICE), labels=batch["labels"].to(DEVICE))
                loss = out.loss
                if use_fedprox:
                    prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_tensors) if p.shape == g.shape)
                    loss += (FEDPROX_MU / 2.0) * prox
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
        return get_weights(model), len(dataset)

    def aggregate(client_weights, client_sizes):
        total = sum(client_sizes)
        agg = [np.zeros_like(w) for w in client_weights[0]]
        for cw, n in zip(client_weights, client_sizes):
            for i, w in enumerate(cw): agg[i] += w * (n / total)
        return agg

    def evaluate(weights, tokenizer):
        val_path = DATA_DIR / "router_val.jsonl"
        texts, labels = [], []
        for line in val_path.open(encoding="utf-8"):
            r = json.loads(line)
            lbl = r.get("label")
            txt = r.get("text", "").strip()
            if txt and lbl in (0, 1, 2): texts.append(txt); labels.append(lbl)
        ds = RouterDataset(texts[:2000], labels[:2000], tokenizer)
        loader = DataLoader(ds, batch_size=64)
        model = load_model(weights)
        model.eval()
        preds = []
        with torch.no_grad():
            for b in loader:
                out = model(input_ids=b["input_ids"].to(DEVICE), attention_mask=b["attention_mask"].to(DEVICE))
                preds += out.logits.argmax(-1).cpu().tolist()
        return accuracy_score(labels[:2000], preds), f1_score(labels[:2000], preds, average="macro")

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR), local_files_only=True)
    branch_datasets = {bid: RouterDataset(*load_branch_data(bid), tokenizer) for bid in [1, 2]}

    results = {}
    for strategy in ["FedAvg", "FedProx"]:
        logger.info(f"Simulating {strategy}...")
        global_weights = get_weights(load_model())
        for rnd in range(1, NUM_ROUNDS + 1):
            cw_list, cs_list = [], []
            for ds in branch_datasets.values():
                w, n = local_train(global_weights, ds, use_fedprox=(strategy == "FedProx"))
                cw_list.append(w); cs_list.append(n)
            global_weights = aggregate(cw_list, cs_list)
            acc, f1 = evaluate(global_weights, tokenizer)
            logger.info(f"  Round {rnd}: acc={acc:.4f}, f1={f1:.4f}")
        results[strategy] = (acc, f1, global_weights)

    best_strategy = max(results, key=lambda x: results[x][1])
    logger.info(f"Best Strategy: {best_strategy}")
    save_dir = MODEL_DIR / f"federated_{best_strategy.lower()}"
    save_dir.mkdir(parents=True, exist_ok=True)
    load_model(results[best_strategy][2]).save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    logger.info(f"Federe model saved to {save_dir}")

# ==============================================================================
# SUBCOMMAND: quantize
# ==============================================================================

def quantize(args):
    input_model = Path(args.model_path) if args.model_path else PROJECT_ROOT / "models" / "finetuned" / "qwen3_merged_v2"
    if not input_model.exists():
        input_model = PROJECT_ROOT / "models" / "finetuned" / "qwen3_merged"

    output_gguf = Path(args.output) if args.output else Path(os.environ.get("GGUF_DIR", "ai/models/finetuned")) / f"{args.output_name}.gguf"
    output_gguf.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Quantizing {input_model} to {output_gguf}")
    convert_script = LLAMA_CPP / "convert_hf_to_gguf.py"

    cmd = ["python3", str(convert_script), str(input_model), "--outfile", str(output_gguf), "--outtype", args.type]
    logger.info(f"Running: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd)

    if result.returncode == 0:
        logger.info(f"Quantization successful in {time.time() - start:.1f}s")
        if not args.no_restart:
            restart_llama_server(str(output_gguf), args.port)
    else:
        logger.error("Quantization failed")

def restart_llama_server(model_path: str, port: int):
    logger.info("Restarting llama-server...")
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(2)
    cmd = [str(LLAMA_CPP / "build" / "bin" / "llama-server"), "-m", model_path, "--host", "127.0.0.1", "--port", str(port), "-ngl", "99", "-c", "2048", "--threads", "6"]

    env = os.environ.copy()
    if sys.platform != "win32":
        lib_dir = str(LLAMA_CPP / "build" / "bin")
        ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_path}" if ld_path else lib_dir

    with open("/tmp/llama_server.log", "w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    logger.info(f"llama-server started on port {port}")

# ==============================================================================
# SUBCOMMAND: export-onnx
# ==============================================================================

def export_onnx(args):
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model_dir = PROJECT_ROOT / args.model_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / "router_fp32.onnx"
    int8_path = output_dir / "router_int8.onnx"

    logger.info(f"Exporting {model_dir} to ONNX")
    ort_model = ORTModelForSequenceClassification.from_pretrained(str(model_dir), export=True)
    ort_model.save_pretrained(str(output_dir))

    if (output_dir / "model.onnx").exists():
        (output_dir / "model.onnx").rename(fp32_path)

    logger.info("Quantizing to INT8")
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    if args.test:
        logger.info("Running benchmarks...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()
        queries = ["Merhaba", "Konut kredisi faizleri", "DCF analizi nasıl yapılır?"]
        for q in queries:
            inputs = tokenizer(q, return_tensors="pt")
            t0 = time.perf_counter()
            with torch.no_grad(): model(**inputs)
            logger.info(f"PT: {q} -> {(time.perf_counter()-t0)*1000:.2f}ms")

            inputs_np = tokenizer(q, return_tensors="np")
            sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
            t0 = time.perf_counter()
            sess.run(None, {k: v for k, v in inputs_np.items() if k in [i.name for i in sess.get_inputs()]})
            logger.info(f"ONNX-INT8: {q} -> {(time.perf_counter()-t0)*1000:.2f}ms")

# ==============================================================================
# SUBCOMMAND: merge-lora
# ==============================================================================

def merge_lora(args):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    lora_path = Path(args.lora_path) if args.lora_path else PROJECT_ROOT / "models" / "finetuned" / "qwen3_lora"
    base_path = args.base_path
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "models" / "finetuned" / "qwen3_merged_v2"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Merging LoRA {lora_path} with base {base_path}")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(lora_path))
    model = model.merge_and_unload()
    model = model.to(torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))
    logger.info(f"Merged model saved to {output_path}")

# ==============================================================================
# SUBCOMMAND: rebuild-splits
# ==============================================================================

def rebuild_splits(args):
    logger.info("Rebuilding router splits...")
    NOISY = {("synthetic", 0), ("synthetic", 1)}
    LABEL_NAMES = {0: "chitchat", 1: "simple", 2: "complex"}
    all_rows = []
    for f in [DATA_DIR / "router_train.jsonl", DATA_DIR / "router_val.jsonl", DATA_DIR / "router_test.jsonl"]:
        if f.exists():
            all_rows.extend([json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()])

    clean = [r for r in all_rows if (r.get("source"), r.get("label")) not in NOISY]
    logger.info(f"Clean: {len(clean)} | Noisy: {len(all_rows) - len(clean)} | Total: {len(all_rows)}")

    by_label = defaultdict(list)
    for r in clean: by_label[r["label"]].append(r)

    train, val, test = [], [], []
    for lbl, items in sorted(by_label.items()):
        random.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * args.test_ratio))
        n_val  = max(1, int(n * args.val_ratio))
        test.extend(items[:n_test])
        val.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])
        logger.info(f"  {LABEL_NAMES.get(lbl, lbl)}: {n} -> train {n-n_test-n_val} / val {n_val} / test {n_test}")

    if args.dry_run:
        logger.info("[Dry-run] No files written.")
        return

    # Backup and write
    for rows, name in [(train, "router_train"), (val, "router_val"), (test, "router_test")]:
        path = DATA_DIR / f"{name}.jsonl"
        bak = DATA_DIR / f"{name}.bak.jsonl"
        if path.exists() and not bak.exists():
            path.rename(bak)
            logger.info(f"  Backed up {path.name} to {bak.name}")
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    logger.info("Splits written successfully")

# ==============================================================================
# MAIN CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="MOSAIC-v2 Model Tool")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # train-router
    p_train = subparsers.add_parser("train-router")
    p_train.add_argument("--train", type=str)
    p_train.add_argument("--val", type=str)
    p_train.add_argument("--test", type=str)
    p_train.add_argument("--output-dir", type=str)
    p_train.add_argument("--epochs", type=int)

    # train-federated
    p_fed = subparsers.add_parser("train-federated")
    p_fed.add_argument("--model-dir", type=str)
    p_fed.add_argument("--rounds", type=int, default=5)
    p_fed.add_argument("--local-epochs", type=int, default=2)
    p_fed.add_argument("--mu", type=float, default=0.01)

    # quantize
    p_quant = subparsers.add_parser("quantize")
    p_quant.add_argument("--model-path", type=str)
    p_quant.add_argument("--output-name", default="qwen3-8b-finance-Q4_K_M")
    p_quant.add_argument("--output", type=str)
    p_quant.add_argument("--type", default="q4_k_m")
    p_quant.add_argument("--no-restart", action="store_true")
    p_quant.add_argument("--port", type=int, default=8099)

    # export-onnx
    p_onnx = subparsers.add_parser("export-onnx")
    p_onnx.add_argument("--model-dir", default="ai/models/router/bert_router_best")
    p_onnx.add_argument("--output-dir", default="ai/models/router/onnx")
    p_onnx.add_argument("--test", action="store_true")

    # merge-lora
    p_merge = subparsers.add_parser("merge-lora")
    p_merge.add_argument("--lora-path", type=str)
    p_merge.add_argument("--base-path", default=os.environ.get("BASE_MODEL_PATH", "unsloth/Qwen3-8B-bnb-4bit"))
    p_merge.add_argument("--output", type=str)

    # rebuild-splits
    p_rebuild = subparsers.add_parser("rebuild-splits")
    p_rebuild.add_argument("--dry-run", action="store_true")
    p_rebuild.add_argument("--val-ratio", type=float, default=0.10)
    p_rebuild.add_argument("--test-ratio", type=float, default=0.10)

    args = parser.parse_args()
    if args.command == "train-router": train_router(args)
    elif args.command == "train-federated": train_federated(args)
    elif args.command == "quantize": quantize(args)
    elif args.command == "export-onnx": export_onnx(args)
    elif args.command == "merge-lora": merge_lora(args)
    elif args.command == "rebuild-splits": rebuild_splits(args)
    else: parser.print_help()

if __name__ == "__main__":
    main()
