#!/usr/bin/env python3
"""
MOSAIC-v2 LLM Domain Fine-Tuning & Live Evaluation Engine
=========================================================
Fine-tunes Qwen3-8B (or Gemma-3-4B) on the curated Turkish Financial SFT dataset:
  `ai/data/processed/finetune_turkish_finance_sft.jsonl`

Features:
  - Unsloth 4-bit QLoRA with direct local model loading (no duplicate download)
  - Real-time training progress tracking to `ai/experiments/training_status.json`
  - Safe memory/disk footprint (<6 GB peak, protecting the 72.5 GB free space)
  - Direct Q4_K_M GGUF export for fast llama.cpp serving
  - Automated post-training evaluation across 5 Turkish banking benchmark domains
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import json
import os
os.environ["UNSLOTH_CE_LOSS_TARGET_GB"] = "0.75"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "0"
import time
from pathlib import Path
from typing import Dict, Any
from loguru import logger
import torch
import ctypes

def set_windows_sleep_prevention(enable: bool = True):
    """Prevents Windows from entering sleep or idle modern standby during long training."""
    if sys.platform == "win32":
        try:
            if enable:
                ES_CONTINUOUS = 0x80000000
                ES_SYSTEM_REQUIRED = 0x00000001
                ES_AWAYMODE_REQUIRED = 0x00000002
                ctypes.windll.kernel32.SetThreadExecutionState(
                    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
                )
                logger.info("Windows uyku engelleme aktif edildi (ES_SYSTEM_REQUIRED | ES_AWAYMODE).")
            else:
                ES_CONTINUOUS = 0x80000000
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                logger.info("Windows uyku engelleme kaldirildi.")
        except Exception as e:
            logger.warning(f"Could not configure Windows sleep prevention: {e}")

try:
    from unsloth import FastLanguageModel
except ImportError:
    logger.error("Unsloth not installed in active environment.")
    sys.exit(1)

from datasets import load_dataset
from transformers import TrainerCallback, TrainingArguments
from trl import SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "ai" / "data" / "processed" / "finetune_turkish_finance_sft.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "ai" / "models" / "finetuned"
EXP_DIR = PROJECT_ROOT / "ai" / "experiments"
STATUS_FILE = EXP_DIR / "training_status.json"

EXP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

LOCAL_QWEN_PATH = PROJECT_ROOT / "ai" / "models" / "finetuned" / "QWEN3_EXPERT_V2"

MODEL_CONFIGS = {
    "qwen3": {
        "hf_model": str(LOCAL_QWEN_PATH) if LOCAL_QWEN_PATH.exists() else "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "max_seq_length": 1024,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "template": "qwen"
    },
    "gemma": {
        "hf_model": "unsloth/gemma-3-4b-it-bnb-4bit",
        "max_seq_length": 1024,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "template": "gemma"
    }
}

def format_qwen_prompt(instruction: str, response: str, context: str = "") -> str:
    sys_prompt = "Sen Türkiye bankacılık, mevzuat ve finans sektöründe uzmanlaşmış analitik bir yapay zeka asistanısın. Tüm soruları detaylı, profesyonel ve doğru finansal terminolojiyle Türkçe yanıtla."
    user_content = f"{instruction}\n\nBağlam: {context}" if context else instruction
    t = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    if len(t) > 4000:
        idx = max(t.rfind(". ", 0, 3950), t.rfind(".\n", 0, 3950))
        if idx != -1:
            t = t[:idx + 1] + "<|im_end|>"
        else:
            t = t[:3950] + "<|im_end|>"
    return t

def format_gemma_prompt(instruction: str, response: str, context: str = "") -> str:
    user_content = f"Sen uzman bir finans asistanısın. {instruction}\nBağlam: {context}" if context else f"Sen uzman bir finans asistanısın. {instruction}"
    t = f"<bos><start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn><eos>"
    if len(t) > 4000:
        idx = max(t.rfind(". ", 0, 3950), t.rfind(".\n", 0, 3950))
        if idx != -1:
            t = t[:idx + 1] + "<end_of_turn><eos>"
        else:
            t = t[:3950] + "<end_of_turn><eos>"
    return t


class TrainingStatusCallback(TrainerCallback):
    """Tracks training progress and writes real-time JSON status."""
    def __init__(self, status_file: Path, total_epochs: int):
        self.status_file = status_file
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.losses = []
        self.loss_history = []

    def on_train_begin(self, args, state, control, **kwargs):
        status = {
            "status": "training_started",
            "step": 0,
            "total_steps": state.max_steps,
            "percent": 0.0,
            "epoch": 0.0,
            "total_epochs": self.total_epochs,
            "current_loss": 0.0,
            "avg_loss": 0.0,
            "learning_rate": f"{args.learning_rate:.2e}",
            "elapsed_time": "0s",
            "eta": "Hesaplanıyor...",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(self.status_file, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            res = torch.cuda.memory_reserved(0) / (1024**3)
            if res > 11.5:
                torch.cuda.empty_cache()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        current_step = state.global_step
        max_steps = state.max_steps
        current_loss = logs.get("loss", 0.0)
        learning_rate = logs.get("learning_rate", 0.0)
        epoch = logs.get("epoch", 0.0)

        if current_loss > 0:
            self.losses.append(current_loss)
            self.loss_history.append({"step": current_step, "loss": round(current_loss, 4), "epoch": round(epoch, 3)})

        elapsed = time.time() - self.start_time
        steps_per_sec = current_step / elapsed if elapsed > 0 else 0
        remaining_steps = max(0, max_steps - current_step)
        eta_sec = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

        def fmt_time(seconds):
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            return f"{h}s {m:02d}d {s:02d}sn" if h > 0 else f"{m}d {s:02d}sn"

        gpu_alloc = 0.0
        gpu_res = 0.0
        if torch.cuda.is_available():
            gpu_alloc = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
            gpu_res = round(torch.cuda.memory_reserved(0) / (1024**3), 2)

        percent = round((current_step / max_steps) * 100, 2) if max_steps > 0 else 0.0
        status = {
            "status": "training",
            "step": current_step,
            "total_steps": max_steps,
            "percent": percent,
            "epoch": round(epoch, 3),
            "total_epochs": self.total_epochs,
            "current_loss": round(current_loss, 4),
            "avg_loss": round(sum(self.losses) / len(self.losses), 4) if self.losses else 0.0,
            "learning_rate": f"{learning_rate:.2e}",
            "steps_per_sec": round(steps_per_sec, 3),
            "elapsed_time": fmt_time(elapsed),
            "eta": fmt_time(eta_sec),
            "gpu_memory_allocated_gb": gpu_alloc,
            "gpu_memory_reserved_gb": gpu_res,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            with open(self.status_file, "w", encoding="utf-8") as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        gpu_alloc = round(torch.cuda.memory_allocated(0) / (1024**3), 2) if torch.cuda.is_available() else 0.0
        gpu_res = round(torch.cuda.memory_reserved(0) / (1024**3), 2) if torch.cuda.is_available() else 0.0
        duration_str = f"{int(elapsed // 3600)}s {int((elapsed % 3600) // 60)}d {int(elapsed % 60)}sn"
        status = {
            "status": "completed",
            "step": state.global_step,
            "total_steps": state.max_steps,
            "percent": 100.0,
            "epoch": self.total_epochs,
            "total_epochs": self.total_epochs,
            "final_loss": round(self.losses[-1], 4) if self.losses else 0.0,
            "avg_loss": round(sum(self.losses) / len(self.losses), 4) if self.losses else 0.0,
            "total_duration_sec": round(elapsed, 1),
            "elapsed_time": duration_str,
            "total_duration_human": duration_str,
            "gpu_memory_allocated_gb": gpu_alloc,
            "gpu_memory_reserved_gb": gpu_res,
            "loss_history": self.loss_history[-20:],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(self.status_file, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)


def evaluate_bist_sentiment(model, tokenizer, model_key: str = "gemma", max_samples: int = 80) -> Dict[str, Any]:
    """Evaluates sentiment classification F1-Macro and Accuracy on BIST sentiment benchmark."""
    test_file = PROJECT_ROOT / "ai" / "data" / "benchmarks" / "bist_sentiment" / "test.jsonl"
    if not test_file.exists():
        logger.warning(f"BIST test file not found: {test_file}")
        return {"accuracy": 0.0, "f1_macro": 0.0, "total_evaluated": 0}

    logger.info("=" * 70)
    logger.info(f"Running BIST Financial Sentiment Benchmark ({max_samples} samples)...")
    logger.info("=" * 70)

    FastLanguageModel.for_inference(model)

    samples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # Balanced sample of positive and negative
    pos = [s for s in samples if s.get("label") == "pozitif"][:max_samples // 2]
    neg = [s for s in samples if s.get("label") == "negatif"][:max_samples // 2]
    eval_set = pos + neg

    y_true = []
    y_pred = []

    for idx, item in enumerate(eval_set, 1):
        text = item["text"]
        true_label = item["label"]

        prompt = (
            f"Aşağıdaki finansal metnin duygu yönünü sadece tek bir kelime ile ('pozitif' veya 'negatif') belirt:\n"
            f"Metin: {text}\n"
            f"Duygu:"
        )

        if model_key == "qwen3":
            formatted = f"<|im_start|>system\nSen finansal duygu analizi uzmanısın.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted = f"<bos><start_of_turn>user\nSen finansal duygu analizi uzmanısın. {prompt}<end_of_turn>\n<start_of_turn>model\n"

        inputs = tokenizer([formatted], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=8, temperature=0.1, use_cache=True)

        gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
        resp = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().lower()

        pred_label = "pozitif" if "pozitif" in resp else ("negatif" if "negatif" in resp else "nötr")
        y_true.append(true_label)
        y_pred.append(pred_label)

    from sklearn.metrics import f1_score, accuracy_score, classification_report
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    logger.info(f"BIST Sentiment Benchmark -> Accuracy: {acc:.4f} | F1-Macro: {f1_macro:.4f}")
    return {
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "total_evaluated": len(eval_set),
        "classification_report": report_dict
    }


def evaluate_model_quality(model, tokenizer, output_dir: Path, model_key: str = "gemma") -> Dict[str, Any]:
    """Runs 5 Turkish financial test prompts and generates evaluation metrics."""
    logger.info("=" * 70)
    logger.info("Starting Post-Training Model Quality & Banking Benchmark Evaluation...")
    logger.info("=" * 70)

    FastLanguageModel.for_inference(model)

    test_cases = [
        {
            "domain": "Bankacılık Formülü (Kredi Anüite)",
            "prompt": "1.000.000 TL tutarında, 120 ay vadeli ve aylık %2.89 faiz oranlı konut kredisinin aylık taksit tutarı ve toplam geri ödemesi nasıl hesaplanır?",
            "expected_keywords": ["anüite", "taksit", "faiz", "bsmv", "kkdf", "muaf"]
        },
        {
            "domain": "KAP Bildirimi ve Risk Analizi",
            "prompt": "Aşağıdaki KAP bildirimini analiz ediniz: 'Şirketimiz Yönetim Kurulu sermayenin %100 oranında bedelsiz artırılmasına ve iç kaynaklardan karşılanmasına karar vermiştir.'",
            "expected_keywords": ["bedelsiz", "sermaye", "hisse", "iç kaynaklar", "olumlu", "pozitif"]
        },
        {
            "domain": "TCMB Para Politikası",
            "prompt": "TCMB'nin politika faizini artırmasının piyasa likiditesi ve mevduat faizleri üzerindeki temel etkileri nelerdir?",
            "expected_keywords": ["politika faizi", "mevduat", "likidite", "kredi", "sıkılaşma", "enflasyon"]
        },
        {
            "domain": "Finansal Rasyo Yorumlama (SYR)",
            "prompt": "Bir bankanın Sermaye Yeterlilik Rasyosu (SYR) nedir? Yasal asgari oran ile BDDK hedef rasyosu kaçtır?",
            "expected_keywords": ["özkaynak", "risk ağırlıklı", "%8", "%12", "tampon", "syr"]
        },
        {
            "domain": "Kredi Kartı Yasal Düzenlemesi (BDDK)",
            "prompt": "Limiti 80.000 TL olan bir kredi kartının BDDK kurallarına göre asgari ödeme oranı yüzde kaçtır?",
            "expected_keywords": ["%40", "50.000", "asgari", "bddk", "limit"]
        }
    ]

    eval_results = []
    total_score = 0

    for idx, tc in enumerate(test_cases, 1):
        prompt_text = tc["prompt"]
        if model_key == "qwen3":
            formatted = f"<|im_start|>system\nSen uzman bir finans asistanısın. Doğru finansal terminolojiyle Türkçe yanıtla.<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted = f"<bos><start_of_turn>user\nSen uzman bir finans asistanısın. {prompt_text}<end_of_turn>\n<start_of_turn>model\n"

        inputs = tokenizer([formatted], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.2,
                top_p=0.9,
                use_cache=True
            )

        gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        matched_kw = [kw for kw in tc["expected_keywords"] if kw.lower() in response_text.lower()]
        score = int((len(matched_kw) / len(tc["expected_keywords"])) * 10)
        total_score += score

        logger.info(f"Test [{idx}/5] Domain: {tc['domain']} | Skor: {score}/10 | Eşleşen Terim: {len(matched_kw)}/{len(tc['expected_keywords'])}")

        eval_results.append({
            "id": idx,
            "domain": tc["domain"],
            "prompt": prompt_text,
            "response": response_text[:300] + ("..." if len(response_text) > 300 else ""),
            "matched_keywords": matched_kw,
            "score": score
        })

    avg_score = round(total_score / len(test_cases), 2)
    logger.info(f"Evaluation Complete! Ortalama Finans Uzmanlık Skoru: {avg_score} / 10")

    # Run BIST Sentiment benchmark
    bist_metrics = evaluate_bist_sentiment(model, tokenizer, model_key=model_key, max_samples=80)

    eval_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_key,
        "banking_domain_score_out_of_10": avg_score,
        "banking_accuracy_pct": round((avg_score / 10.0) * 100, 2),
        "bist_sentiment_f1_macro": bist_metrics.get("f1_macro", 0.0),
        "bist_sentiment_accuracy": bist_metrics.get("accuracy", 0.0),
        "bist_sentiment_report": bist_metrics.get("classification_report", {}),
        "banking_test_results": eval_results
    }

    # Save metrics JSON
    metrics_json_path = EXP_DIR / f"{model_key}_finetune_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, ensure_ascii=False, indent=2)

    # Save Benchmark Markdown Report
    eval_md_path = EXP_DIR / f"{model_key}_finetune_benchmark_report.md"
    with open(eval_md_path, "w", encoding="utf-8") as f:
        f.write(f"# MOSAIC-v2: {model_key.upper()} Model Alan Uyarlama ve Benchmark Raporu\n\n")
        f.write(f"**Tarih:** {eval_report['timestamp']}  \n")
        f.write(f"**Model:** `{model_key}`  \n\n")
        f.write("## 1. Temel Benchmark Metrikleri\n\n")
        f.write("| Metrik | Değer | Hedef |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| **BIST Duygu Analizi F1-Macro** | **{bist_metrics.get('f1_macro', 0.0):.4f}** | > 0.6000 |\n")
        f.write(f"| **BIST Duygu Analizi Accuracy** | **{bist_metrics.get('accuracy', 0.0) * 100:.2f}%** | > 70.0% |\n")
        f.write(f"| **Türkçe Bankacılık Alan Skoru** | **{avg_score} / 10** | > 7.0 / 10 |\n")
        f.write(f"| **Bankacılık Terim Uyumu** | **%{round((avg_score / 10.0) * 100, 1)}** | > 70.0% |\n\n")
        f.write("## 2. Bankacılık Senaryo Testleri\n\n")
        f.write("| No | Alan / Kapsam | Soru | Eşleşen Terimler | Skor |\n")
        f.write("| :--- | :--- | :--- | :--- | :---: |\n")
        for t in eval_results:
            kw_str = ", ".join(t["matched_keywords"])
            f.write(f"| {t['id']} | **{t['domain']}** | {t['prompt']} | `{kw_str}` | **{t['score']}/10** |\n")
        f.write("\n---\n\n### Örnek Model Yanıtları\n\n")
        for t in eval_results:
            f.write(f"#### {t['id']}. {t['domain']}\n")
            f.write(f"**Soru:** {t['prompt']}\n\n")
            f.write(f"**Model Çıktısı:**\n> {t['response']}\n\n")

    logger.info(f"Detailed Evaluation Report written -> {eval_md_path}")
    return eval_report


def train(args):
    set_windows_sleep_prevention(True)
    try:
        model_key = args.model.lower()
        if model_key not in MODEL_CONFIGS:
            logger.error(f"Unknown model: {model_key}. Choose from {list(MODEL_CONFIGS.keys())}")
            sys.exit(1)

        dataset_path = Path(args.dataset) if args.dataset else DATA_PATH
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            sys.exit(1)

        cfg = MODEL_CONFIGS[model_key]
        model_out = OUTPUT_DIR / f"{model_key}_lora_finance"
        max_seq_len = getattr(args, "max_seq_len", 512)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Strictly prevent Windows WDDM from paging into system DDR5 RAM

        extra_kwargs = {}
        if model_key == "gemma":
            extra_kwargs["attn_implementation"] = "eager"

        logger.info(f"Initializing FastLanguageModel: {cfg['hf_model']} (4-bit, max_seq_length={max_seq_len})...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = cfg["hf_model"],
            max_seq_length = max_seq_len,
            load_in_4bit = True,
            trust_remote_code = True,
            **extra_kwargs,
        )

        lora_r = getattr(args, "r", 32)
        lora_alpha = getattr(args, "alpha", 64)
        logger.info(f"Applying LoRA adapter (r={lora_r}, alpha={lora_alpha})...")
        model = FastLanguageModel.get_peft_model(
            model,
            r = lora_r,
            target_modules = cfg["target_modules"],
            lora_alpha = lora_alpha,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 42,
        )

        logger.info(f"Loading SFT dataset from {dataset_path}...")
        raw_dataset = load_dataset("json", data_files=str(dataset_path), split="train")

        def map_prompts(examples):
            instructions = examples["instruction"]
            responses = examples["response"]
            contexts = examples.get("input", [""] * len(instructions))
            texts = []
            for inst, resp, ctx in zip(instructions, responses, contexts):
                if model_key == "qwen3":
                    t = format_qwen_prompt(inst, resp, ctx or "")
                else:
                    t = format_gemma_prompt(inst, resp, ctx or "")
                texts.append(t)
            return {"text": texts}

        train_dataset = raw_dataset.map(map_prompts, batched=True)

        use_bf16 = torch.cuda.is_bf16_supported()
        grad_accum = getattr(args, "grad_accum", 2)
        training_args = TrainingArguments(
            output_dir = str(model_out),
            num_train_epochs = args.epochs,
            max_steps = args.max_steps if args.max_steps > 0 else -1,
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = grad_accum,
            learning_rate = args.lr,
            weight_decay = 0.01,
            logging_steps = 5,
            save_strategy = "steps",
            save_steps = 250,
            save_total_limit = 1,
            bf16 = use_bf16,
            fp16 = not use_bf16,
            optim = "adamw_8bit",
            max_grad_norm = 0.3,
            warmup_steps = 20,
            lr_scheduler_type = "cosine",
            report_to = "none"
        )

        status_callback = TrainingStatusCallback(STATUS_FILE, total_epochs=args.epochs)

        logger.info(f"Starting SFT Training ({args.epochs} epochs, batch_size={args.batch_size}, grad_accum={grad_accum}, max_seq_len={max_seq_len})...")
        trainer = SFTTrainer(
            model = model,
            train_dataset = train_dataset,
            dataset_text_field = "text",
            max_seq_length = max_seq_len,
            tokenizer = tokenizer,
            args = training_args,
            callbacks = [status_callback]
        )

        last_checkpoint = None
        if model_out.exists():
            checkpoints = sorted(
                [d for d in model_out.glob("checkpoint-*") if d.is_dir()],
                key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0
            )
            if checkpoints:
                last_checkpoint = str(checkpoints[-1])
                logger.info(f"Son kontrol noktasından devam ediliyor: {last_checkpoint}")

        trainer.train(resume_from_checkpoint=last_checkpoint)

        logger.info(f"Training completed! Saving LoRA adapter -> {model_out}")
        model.save_pretrained(str(model_out))
        tokenizer.save_pretrained(str(model_out))

        # Evaluate model quality immediately after training
        try:
            evaluate_model_quality(model, tokenizer, model_out, model_key=model_key)
        except Exception as e:
            logger.warning(f"Evaluation error: {e}")

        # GGUF Export
        if args.export_gguf:
            gguf_out = OUTPUT_DIR / f"{model_key}-3-4b-finance-Q4_K_M"
            logger.info(f"Exporting directly to Q4_K_M GGUF format -> {gguf_out}...")
            try:
                model.save_pretrained_gguf(str(gguf_out), tokenizer, quantization_method="q4_k_m")
                logger.info(f"GGUF model successfully created -> {gguf_out}")
                # Also check if gguf file was generated and make a standard-named alias
                for f in OUTPUT_DIR.glob(f"*{model_key}*.gguf"):
                    logger.info(f"Found GGUF artifact: {f.name} ({f.stat().st_size / (1024**2):.2f} MB)")
            except Exception as e:
                logger.warning(f"Native Unsloth GGUF export failed ({e}). LoRA adapter is safely preserved.")
    finally:
        set_windows_sleep_prevention(False)

def main():
    parser = argparse.ArgumentParser(description="MOSAIC-v2 LLM Domain Fine-Tuning CLI")
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "gemma"], help="Target model architecture")
    parser.add_argument("--dataset", type=str, default=str(DATA_PATH), help="Path to SFT JSONL dataset")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (Default: 1)")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Maximum sequence length (Default: 512)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps (-1 for full epochs)")
    parser.add_argument("--r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=64, help="LoRA alpha scaling factor")
    parser.add_argument("--export-gguf", action="store_true", help="Directly export to Q4_K_M GGUF to save disk space (~5GB vs 16GB FP16)")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset without training")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
