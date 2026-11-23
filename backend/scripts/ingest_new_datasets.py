"""
MOSAIC-v2 New Datasets Ingestion & Harmonization Pipeline
=========================================================
Downloads and ingests open-source Turkish finance datasets:
  1. emre/finance-reasoning-turkish
  2. umarigan/turkiye_finance_qa
  3. yusufbaykaloglu/turkish-finance-dataset
  4. finansai/kap-turkish-financial-sentiment
  5. aufklarer/central-bank-communications (TCMB TR)
  6. Mustafa2735/stsb-tr-hukuk (BDDK / Banking law)
  7. AltaySec/turkish-llm-injection (Banking security eval)

Formats, normalizes, and integrates them into ai/data/raw/ and ai/data/processed/.
"""

import json
import re
import unicodedata
from pathlib import Path
from datasets import load_dataset
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "ai" / "data" / "raw"
PROC_DIR = PROJECT_ROOT / "ai" / "data" / "processed"
BENCH_DIR = PROJECT_ROOT / "ai" / "data" / "benchmarks"
SPLITS_DIR = PROJECT_ROOT / "ai" / "data" / "splits"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)
BENCH_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

TR_FIX_MAP = {
    "Ý": "İ", "ý": "ı", "Þ": "Ş", "þ": "ş",
    "Ð": "Ğ", "ð": "ğ",
    "î": "i", "â": "a", "û": "u",
    "\x00": "", "\x0b": "\n", "\x0c": "\n",
    "\u200b": "", "\u200c": "", "\u200d": "",
    "\ufeff": "",
}

def normalize_turkish(text: str) -> str:
    if not text:
        return ""
    for broken, fixed in TR_FIX_MAP.items():
        text = text.replace(broken, fixed)
    text = unicodedata.normalize("NFC", text)
    return text.strip()

def save_jsonl(records: list[dict], path: Path, name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"[{name}] {len(records):,} records saved -> {path}")

def ingest_all():
    logger.info("Starting ingestion of new Turkish Finance datasets...")

    # 1. emre/finance-reasoning-turkish
    logger.info("Ingesting emre/finance-reasoning-turkish...")
    try:
        ds1 = load_dataset("emre/finance-reasoning-turkish", split="train")
        recs1 = []
        for row in ds1:
            inp = normalize_turkish(row.get("input", ""))
            out = normalize_turkish(row.get("output", ""))
            # Extract question if enclosed in tags
            q_match = re.search(r'<user_question>(.*?)(?:</user_question>|$)', inp, re.DOTALL)
            question = q_match.group(1).strip() if q_match else inp.strip()

            # Clean thinking tags if desired or retain
            recs1.append({
                "source": "emre/finance-reasoning-turkish",
                "instruction": question,
                "input": inp,
                "response": out,
                "split": "reasoning"
            })
        save_jsonl(recs1, RAW_DIR / "hf_finance_reasoning.jsonl", "FinanceReasoning")
    except Exception as e:
        logger.error(f"Failed ds1: {e}")

    # 2. umarigan/turkiye_finance_qa
    logger.info("Ingesting umarigan/turkiye_finance_qa...")
    try:
        ds2 = load_dataset("umarigan/turkiye_finance_qa", split="train")
        recs2 = []
        for row in ds2:
            q = normalize_turkish(row.get("soru", ""))
            a = normalize_turkish(row.get("cevap", ""))
            if q and a:
                recs2.append({
                    "source": "umarigan/turkiye_finance_qa",
                    "instruction": q,
                    "response": a,
                    "split": "rag_corpus"
                })
        save_jsonl(recs2, RAW_DIR / "hf_turkiye_finance_qa.jsonl", "TurkiyeFinanceQA")
    except Exception as e:
        logger.error(f"Failed ds2: {e}")

    # 3. yusufbaykaloglu/turkish-finance-dataset
    logger.info("Ingesting yusufbaykaloglu/turkish-finance-dataset...")
    try:
        ds3 = load_dataset("yusufbaykaloglu/turkish-finance-dataset", split="train")
        recs3 = []
        for row in ds3:
            msgs = row.get("translated_messages") or []
            if isinstance(msgs, str):
                try:
                    msgs = json.loads(msgs)
                except Exception:
                    continue
            user_msg = ""
            assistant_msg = ""
            for m in msgs:
                role = m.get("role") or m.get("from")
                content = normalize_turkish(m.get("content") or m.get("value") or "")
                if role in ("user", "human"):
                    user_msg = content
                elif role in ("assistant", "gpt", "model"):
                    assistant_msg = content
            if user_msg and assistant_msg:
                recs3.append({
                    "source": "yusufbaykaloglu/turkish-finance-dataset",
                    "instruction": user_msg,
                    "response": assistant_msg,
                    "split": "chat_finance"
                })
        save_jsonl(recs3, RAW_DIR / "hf_turkish_finance_chat.jsonl", "TurkishFinanceChat")
    except Exception as e:
        logger.error(f"Failed ds3: {e}")

    # 4. finansai/kap-turkish-financial-sentiment
    logger.info("Ingesting finansai/kap-turkish-financial-sentiment...")
    try:
        ds4 = load_dataset("finansai/kap-turkish-financial-sentiment", split="train")
        recs4 = []
        for row in ds4:
            msgs = row.get("messages") or []
            user_text = ""
            asst_text = ""
            for m in msgs:
                role = m.get("role")
                cnt = normalize_turkish(m.get("content", ""))
                if role == "user":
                    user_text = cnt
                elif role == "assistant":
                    asst_text = cnt
            if user_text:
                recs4.append({
                    "source": "finansai/kap-turkish-financial-sentiment",
                    "instruction": user_text,
                    "response": asst_text,
                    "split": "kap_sentiment"
                })
        save_jsonl(recs4, RAW_DIR / "hf_kap_sentiment.jsonl", "KAPSentiment")
    except Exception as e:
        logger.error(f"Failed ds4: {e}")

    # 5. aufklarer/central-bank-communications (TCMB)
    logger.info("Ingesting aufklarer/central-bank-communications (TCMB)...")
    try:
        ds5 = load_dataset("aufklarer/central-bank-communications", split="train")
        recs5 = []
        for row in ds5:
            if row.get("language") == "tr" or str(row.get("central_bank")).lower() in ("tcmb", "cbrt"):
                txt = normalize_turkish(row.get("text", ""))
                if txt and len(txt) > 20:
                    recs5.append({
                        "source": "tcmb_communications",
                        "document_id": row.get("document_id"),
                        "meeting_date": row.get("meeting_date"),
                        "doc_type": row.get("doc_type"),
                        "sentiment": row.get("sentiment"),
                        "topic": row.get("topic"),
                        "text": txt,
                        "split": "macro_policy"
                    })
        save_jsonl(recs5, RAW_DIR / "hf_tcmb_communications.jsonl", "TCMBCommunications")
    except Exception as e:
        logger.error(f"Failed ds5: {e}")

    # 6. Mustafa2735/stsb-tr-hukuk (Banking & Loan Law)
    logger.info("Ingesting Mustafa2735/stsb-tr-hukuk...")
    try:
        ds6 = load_dataset("Mustafa2735/stsb-tr-hukuk")
        recs6 = []
        for split_name in ds6.keys():
            for row in ds6[split_name]:
                s1 = normalize_turkish(row.get("sentence1", ""))
                s2 = normalize_turkish(row.get("sentence2", ""))
                score = row.get("score", 0.0)
                topic = row.get("topic", "")
                recs6.append({
                    "source": "stsb_banking_law",
                    "sentence1": s1,
                    "sentence2": s2,
                    "score": score,
                    "topic": topic,
                    "split": split_name
                })
        save_jsonl(recs6, RAW_DIR / "hf_banking_law_stsb.jsonl", "BankingLawSTSB")
    except Exception as e:
        logger.error(f"Failed ds6: {e}")

    # 7. AltaySec/turkish-llm-injection (Banking Security Benchmark)
    logger.info("Ingesting AltaySec/turkish-llm-injection...")
    try:
        ds7 = load_dataset("AltaySec/turkish-llm-injection", split="train")
        recs7 = []
        for row in ds7:
            p = normalize_turkish(row.get("prompt", ""))
            recs7.append({
                "id": row.get("id"),
                "prompt": p,
                "category": row.get("category"),
                "subcategory": row.get("subcategory"),
                "severity": row.get("severity"),
                "expected_failure_mode": row.get("expected_failure_mode"),
                "source": "altaysec_banking_injection"
            })
        save_jsonl(recs7, BENCH_DIR / "banking_security_eval.jsonl", "BankingSecurityBench")
    except Exception as e:
        logger.error(f"Failed ds7: {e}")

    logger.info("All new datasets successfully downloaded and structured in ai/data/raw/!")

if __name__ == "__main__":
    ingest_all()
