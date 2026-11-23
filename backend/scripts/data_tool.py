import argparse
import json
import os
import random
import re
import sys
import time
import hashlib
from pathlib import Path
import urllib.request

import numpy as np
from loguru import logger

# --- ROOT PATHS ---
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "ai" / "data" / "raw"
PROC_DIR = ROOT / "ai" / "data" / "processed"
BENCHMARKS_DIR = ROOT / "ai" / "data" / "benchmarks"
SPLITS_DIR = ROOT / "ai" / "data" / "splits"
CHROMA_DIR = ROOT / "ai" / "data" / "chromadb"
BRANCH1_DIR = SPLITS_DIR / "branch1"
BRANCH2_DIR = SPLITS_DIR / "branch2"
HQ_DIR = SPLITS_DIR / "hq"

# Ensure directories exist
for d in [RAW_DIR, PROC_DIR, BENCHMARKS_DIR, SPLITS_DIR, CHROMA_DIR, BRANCH1_DIR, BRANCH2_DIR, HQ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- LOGGING ---
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {level} | {message}")

# --- SHARED UTILITIES ---
TR_FIX_MAP = {
    "Ý": "İ", "ý": "ı", "Þ": "Ş", "þ": "ş",
    "Ð": "Ğ", "ð": "ğ",
    "î": "i", "â": "a", "û": "u",
    "\x00": "", "\x0b": "\n", "\x0c": "\n",
    "\u200b": "", "\u200c": "", "\u200d": "",
    "\ufeff": "",  # BOM
}

def normalize_turkish(text: str) -> str:
    """Fix common Turkish encoding issues and normalize to NFC."""
    if not text:
        return ""
    for broken, fixed in TR_FIX_MAP.items():
        text = text.replace(broken, fixed)
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    return text.strip()

def save_jsonl(records: list[dict], output_path: Path, tag: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"[{tag}] {len(records):,} records -> {output_path}")

def server_available(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False

# --- SUBCOMMAND: collect-hf ---
def cmd_collect_hf(args):
    """Collect datasets from HuggingFace."""
    from datasets import load_dataset

    random.seed(42)
    np.random.seed(42)

    _HF_CACHE = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

    # 1. Turkish Finance Instruction
    logger.info("Collecting Turkish Finance Instruction...")
    try:
        ds = load_dataset("Dbmaxwell/turkish-finance-instruction-dataset", split="train", cache_dir=_HF_CACHE)
        finance_records = []
        for row in ds:
            instruction = row.get("instruction") or row.get("input") or row.get("question") or ""
            response = row.get("response") or row.get("output") or row.get("answer") or ""
            context = row.get("context") or row.get("input_context") or ""
            if instruction and response:
                finance_records.append({
                    "source": "turkish_finance_instruction",
                    "instruction": instruction.strip(),
                    "context": context.strip(),
                    "response": response.strip(),
                    "split": "rag_corpus"
                })
        save_jsonl(finance_records, RAW_DIR / "finance_instructions.jsonl", "TurkishFinance")
    except Exception as e:
        logger.error(f"Failed to load Turkish Finance: {e}")

    # 2. MASSIVE Router
    logger.info("Collecting MASSIVE Router...")
    MASSIVE_LABEL_MAP = {
        "alarm": 0, "audio_volume_down": 0, "audio_volume_mute": 0, "audio_volume_other": 0, "audio_volume_up": 0,
        "calendar_query": 0, "calendar_remove": 0, "calendar_set": 0, "datetime_convert": 0, "datetime_query": 0,
        "email_addcontact": 0, "email_querycontact": 0, "general_greet": 0, "general_joke": 0, "general_quirky": 0,
        "music_dislikeness": 0, "music_likeness": 0, "music_query": 0, "music_settings": 0, "play_audiobook": 0,
        "play_game": 0, "play_music": 0, "play_podcasts": 0, "play_radio": 0, "social_post": 0, "social_query": 0,
        "takeaway_order": 0, "transport_taxi": 0, "transport_ticket": 0, "transport_traffic": 0, "weather_query": 0,
        "banking": 1, "bill_payment": 1, "cooking_recipe": 1, "currency": 1, "definition": 1, "email_query": 1,
        "email_sendemail": 1, "insurance_query": 1, "iot_cleaning": 1, "iot_coffee": 1, "lists_createoradd": 1,
        "lists_query": 1, "lists_remove": 1, "news_query": 1, "recommendation_events": 1, "recommendation_locations": 1,
        "recommendation_movies": 1, "shopping_list": 1, "shopping_list_update": 1, "shopping_query": 1, "transport_query": 1,
    }
    try:
        ds = load_dataset("AmazonScience/massive", "tr-TR", cache_dir=_HF_CACHE)
        router_records = []
        for split_name in ["train", "validation", "test"]:
            for row in ds[split_name]:
                intent = row.get("intent") or row.get("scenario") or ""
                utterance = row.get("utt") or row.get("text") or ""
                label = MASSIVE_LABEL_MAP.get(intent)
                if label is not None:
                    router_records.append({
                        "source": "massive",
                        "text": utterance.strip(),
                        "label": label,
                        "label_name": ["chitchat", "simple", "complex"][label],
                        "intent": intent
                    })
        save_jsonl(router_records, RAW_DIR / "router_massive.jsonl", "MASSIVE")
    except Exception as e:
        logger.warning(f"Failed to load MASSIVE: {e}. Consider running clean after collect to use fallbacks.")

    # 3. Pretraining Corpus
    logger.info("Collecting Pretraining Corpus...")
    all_texts = []
    try:
        ds = load_dataset("wikiann", "tr", split="train", cache_dir=_HF_CACHE)
        all_texts.extend([" ".join(row["tokens"]) for row in ds if row.get("tokens")][:50_000])
        ds = load_dataset("opus100", "en-tr", split="train", cache_dir=_HF_CACHE)
        all_texts.extend([row["translation"]["tr"] for row in ds if row.get("translation", {}).get("tr")][:50_000])
        ds = load_dataset("nickmuchi/financial-classification", split="train", cache_dir=_HF_CACHE)
        all_texts.extend([row["text"] for row in ds if row.get("text")])
    except Exception as e:
        logger.warning(f"Error collecting pretraining: {e}")

    random.shuffle(all_texts)
    pretrain_path = RAW_DIR / "pretraining_corpus.txt"
    with open(pretrain_path, "w", encoding="utf-8") as f:
        f.write("\n".join(t.strip() for t in all_texts if t.strip()))
    logger.info(f"[Pretraining] {len(all_texts):,} lines -> {pretrain_path}")

# --- SUBCOMMAND: collect-kap ---
def cmd_collect_kap(args):
    """Collect PDF disclosures from KAP."""
    import requests
    KAP_SEARCH_URL = "https://www.kap.org.tr/tr/api/disclosures"
    KAP_PDF_BASE   = "https://www.kap.org.tr"
    REQUEST_DELAY = 2.0

    def _safe_get(url):
        time.sleep(REQUEST_DELAY)
        headers = {"User-Agent": "MosaicResearch/1.0 (academic research)"}
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp
        except Exception as e:
            logger.warning(f"GET failed [{url}]: {e}")
            return None

    logger.info(f"Collecting KAP disclosures for year={args.year}, max_pages={args.max_pages}")
    all_items = []
    for page in range(1, args.max_pages + 1):
        resp = _safe_get(f"{KAP_SEARCH_URL}?year={args.year}&pageIndex={page}&pageSize=50")
        if not resp: break
        try:
            data = resp.json()
        except Exception as e:
            logger.warning(f"Invalid JSON from KAP page {page}: {e}")
            break
        items = data if isinstance(data, list) else data.get("data", data.get("results", []))
        if not items: break
        all_items.extend(items)
        logger.info(f"Page {page}: {len(items)} items")

    records = []
    try:
        from pdfminer.high_level import extract_text
        import io
    except ImportError:
        logger.error("pdfminer.six not installed. Run: pip install pdfminer.six")
        return

    for item in all_items:
        pdf_url = item.get("pdfUrl") or item.get("dosyaUrl", "")
        if not pdf_url: continue
        if not pdf_url.startswith("http"): pdf_url = KAP_PDF_BASE + pdf_url

        resp = _safe_get(pdf_url)
        if not resp: continue
        try:
            text = extract_text(io.BytesIO(resp.content)).strip()
            if len(text) < 100: continue
            records.append({
                "source": "kap",
                "title": item.get("title", ""),
                "company": item.get("companyName", ""),
                "year": args.year,
                "pdf_url": pdf_url,
                "text": text[:20_000],
                "char_count": len(text)
            })
            logger.info(f"   {item.get('companyName', '')[:30]} - {len(text):,} chars")
        except Exception as e:
            logger.warning(f"PDF error [{pdf_url}]: {e}")

    save_jsonl(records, RAW_DIR / f"kap_{args.year}.jsonl", "KAP")

# --- SUBCOMMAND: collect-news ---
def cmd_collect_news(args):
    """Collect finance news using trafilatura."""
    import requests
    import xml.etree.ElementTree as ET
    try:
        import trafilatura
    except ImportError:
        logger.error("trafilatura not installed. Run: pip install trafilatura")
        return

    SITES = [
        {"name": "bloomberght", "sitemap": "https://www.bloomberght.com/sitemap.xml"},
        {"name": "dunya", "sitemap": "https://www.dunya.com/sitemap.xml"},
    ]

    all_articles = []
    complex_examples = []

    for site in SITES:
        logger.info(f"Scanning sitemap: {site['name']}")
        urls = []
        try:
            resp = requests.get(site["sitemap"], timeout=10, headers={"User-Agent": "MosaicResearch/1.0"})
            root = ET.fromstring(resp.content)
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            for loc in root.findall(".//sm:url/sm:loc", ns):
                if loc.text: urls.append(loc.text)
        except Exception as e:
            logger.warning(f"Sitemap error [{site['name']}]: {e}")
            continue

        random.shuffle(urls)
        collected = 0
        for url in urls[:args.articles_per_site * 2]:
            if collected >= args.articles_per_site: break
            time.sleep(1.5)
            try:
                downloaded = trafilatura.fetch_url(url)
                text = trafilatura.extract(downloaded, favor_precision=True)
                if not text or len(text) < 200: continue

                title = text.split("\n")[0][:200]
                kw_patterns = [r'\b(?:faiz|enflasyon|döviz|kur|borsa|hisse|tahvil|bono|kredi|yatırım|portföy)\b', r'\b(?:TL|USD|EUR|BIST|TCMB|SPK|BDDK|KAP)\b']
                keywords = []
                for p in kw_patterns: keywords.extend(re.findall(p, text, re.I))
                keywords = list(set(keywords))[:10]

                all_articles.append({"source": site["name"], "url": url, "title": title, "text": text[:10_000], "char_count": len(text)})

                # COMPLEX router example
                if keywords:
                    q = f"{random.choice(keywords)} gelişmesi piyasaları nasıl etkiler?"
                    complex_examples.append({"source": f"fin_news_{site['name']}", "text": q, "label": 2, "label_name": "complex"})

                collected += 1
                if collected % 20 == 0: logger.info(f"  {site['name']}: {collected}")
            except Exception: continue

    save_jsonl(all_articles, RAW_DIR / "fin_news.jsonl", "News")
    save_jsonl(complex_examples, RAW_DIR / "router_complex.jsonl", "NewsComplex")

# --- SUBCOMMAND: collect-bench ---
def cmd_collect_bench(args):
    """Collect benchmark datasets (TR-MMLU, BIST Sentiment)."""
    from datasets import load_dataset, get_dataset_config_names

    TR_MMLU_DIR = BENCHMARKS_DIR / "tr_mmlu"
    SENTIMENT_DIR = BENCHMARKS_DIR / "bist_sentiment"

    # 1. TR-MMLU
    TR_MMLU_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Collecting TR-MMLU...")
    all_mmlu = []
    try:
        configs = get_dataset_config_names("malhajar/mmlu-tr")
        for cfg in configs:
            if cfg in ("all", "default"): continue
            try:
                ds = load_dataset("malhajar/mmlu-tr", cfg, split="test")
                for row in ds:
                    q = normalize_turkish(str(row.get("question", "")))
                    c_list = row.get("choices", [])
                    if len(c_list) >= 4:
                        choices = {
                            "A": normalize_turkish(str(c_list[0])),
                            "B": normalize_turkish(str(c_list[1])),
                            "C": normalize_turkish(str(c_list[2])),
                            "D": normalize_turkish(str(c_list[3])),
                        }
                        ans_idx = row.get("answer")
                        if q and all(choices.values()) and ans_idx is not None and 0 <= ans_idx < 4:
                            ans_letter = ["A", "B", "C", "D"][ans_idx]
                            all_mmlu.append({"subject": cfg, "question": q, "choices": choices, "answer": ans_letter})
            except Exception: continue
    except Exception as e:
        logger.error(f"TR-MMLU error: {e}")

    random.shuffle(all_mmlu)
    save_jsonl(all_mmlu[:args.max_mmlu_test], TR_MMLU_DIR / "test.jsonl", "MMLU_Test")
    save_jsonl(all_mmlu[args.max_mmlu_test:args.max_mmlu_test+500], TR_MMLU_DIR / "dev.jsonl", "MMLU_Dev")

    # 2. BIST Sentiment
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Collecting BIST Sentiment...")
    all_sent = []
    try:
        ds = load_dataset("winvoker/turkish-sentiment-analysis-dataset", split="train")
        for i, row in enumerate(ds):
            text = normalize_turkish(str(row.get("text") or row.get("Text") or ""))
            label_val = row.get("label") or row.get("Label")
            if str(label_val).lower() == "positive": label = "pozitif"
            elif str(label_val).lower() == "negative": label = "negatif"
            elif str(label_val).lower() == "neutral": label = "nötr"
            else: label = {0: "negatif", 1: "pozitif", 2: "nötr"}.get(label_val)
            if text and label and len(text) > 10:
                all_sent.append({"text": text, "label": label, "source": "winvoker"})
    except Exception as e:
        logger.error(f"Sentiment error: {e}")

    random.shuffle(all_sent)
    save_jsonl(all_sent[:args.max_sentiment_test], SENTIMENT_DIR / "test.jsonl", "Sentiment_Test")
    save_jsonl(all_sent[args.max_sentiment_test:args.max_sentiment_test+args.max_sentiment_train], SENTIMENT_DIR / "train.jsonl", "Sentiment_Train")

# --- SUBCOMMAND: clean ---
def cmd_clean(args):
    """Clean and deduplicate data."""
    PDF_ARTIFACTS = re.compile(r"(\bSayfa\s+\d+\s*/\s*\d+\b)|(\bPage\s+\d+\s+of\s+\d+\b)|(\b\d{1,3}\s*$)|(^[\s\-_=]{10,}$)|(www\.\S+|http\S+)|(\S+@\S+\.\S+)", re.M | re.X)

    def clean_text(text):
        text = normalize_turkish(text)
        text = PDF_ARTIFACTS.sub(" ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def is_valid(text):
        if len(text) < 100: return False
        tr_letters = len(re.findall(r"[a-zğüşıöç]", text.lower()))
        non_space = len([c for c in text if not c.isspace()])
        return tr_letters / (non_space + 1) >= 0.3

    def dedup(records, key="text", threshold=0.85):
        try:
            from datasketch import MinHash, MinHashLSH
            lsh = MinHashLSH(threshold=threshold, num_perm=128)
            unique = []
            for i, rec in enumerate(records):
                text = rec.get(key, "").lower()
                m = MinHash(num_perm=128)
                for sh in {text[j:j+6].encode("utf-8") for j in range(len(text)-5)}: m.update(sh)
                if not lsh.query(m):
                    lsh.insert(f"d{i}", m)
                    unique.append(rec)
            return unique
        except ImportError:
            logger.warning("datasketch not found, using exact dedup.")
            seen = set()
            unique = []
            for r in records:
                h = hashlib.md5(r.get(key, "").encode()).hexdigest()
                if h not in seen:
                    seen.add(h); unique.append(r)
            return unique

    # Processing
    logger.info("Cleaning Finance Instructions...")
    if (RAW_DIR / "finance_instructions.jsonl").exists():
        recs = []
        for line in open(RAW_DIR / "finance_instructions.jsonl", encoding="utf-8"):
            r = json.loads(line)
            r["response"] = clean_text(r["response"])
            if is_valid(r["response"]): recs.append(r)
        recs = dedup(recs, "response")
        random.shuffle(recs)
        save_jsonl(recs, PROC_DIR / "finance_rag_train.jsonl", "CleanFinance")

    logger.info("Cleaning News...")
    if (RAW_DIR / "fin_news.jsonl").exists():
        recs = []
        for line in open(RAW_DIR / "fin_news.jsonl", encoding="utf-8"):
            r = json.loads(line); r["text"] = clean_text(r["text"])
            if is_valid(r["text"]): recs.append(r)
        recs = dedup(recs, "text")
        save_jsonl(recs, PROC_DIR / "fin_news_train.jsonl", "CleanNews")

    # Router logic
    logger.info("Combining Router Datasets...")
    router_all = []
    for src in ["router_massive.jsonl", "router_complex.jsonl", "router_synth_complex.jsonl"]:
        p = RAW_DIR / src
        if p.exists():
            for line in open(p, encoding="utf-8"):
                r = json.loads(line); r["text"] = clean_text(r["text"])
                if len(r["text"]) > 10: router_all.append(r)
    router_all = dedup(router_all, "text")
    random.shuffle(router_all)
    n = len(router_all)
    save_jsonl(router_all[:int(n*0.8)], PROC_DIR / "router_train.jsonl", "RouterTrain")
    save_jsonl(router_all[int(n*0.8):int(n*0.9)], PROC_DIR / "router_val.jsonl", "RouterVal")
    save_jsonl(router_all[int(n*0.9):], PROC_DIR / "router_test.jsonl", "RouterTest")

# --- SUBCOMMAND: synth-qa ---
def cmd_synth_qa(args):
    """Generate synthetic Q&A pairs using LLM server."""
    if not server_available(args.llm_url):
        logger.error(f"LLM server not found at {args.llm_url}. Start it with scripts/_llm_server.py")
        return

    logger.info(f"Generating {args.num_qa} synthetic Q&A pairs...")
    chunks = []
    for src in [RAW_DIR / "finance_instructions.jsonl", RAW_DIR / "fin_news.jsonl"]:
        if src.exists():
            for line in open(src, encoding="utf-8"):
                r = json.loads(line)
                t = r.get("text") or r.get("response") or ""
                if len(t) > 300: chunks.append(t[:2000])
    random.shuffle(chunks)

    qa_records, complex_records = [], []
    generated = 0
    for context in chunks[:args.num_qa * 2]:
        if generated >= args.num_qa: break
        prompt = "<|im_start|>system\nProduce one complex Turkish finance Q&A pair in JSON: " + '{"soru": "...", "cevap": "..."}\n<|im_end|>\n<|im_start|>user\nContext: ' + context + "\n<|im_end|>\n<|im_start|>assistant\n"
        try:
            req = urllib.request.Request(f"{args.llm_url}/completion", data=json.dumps({"prompt": prompt, "n_predict": 512, "temperature": 0.7, "stop": ["<|im_end|>"]}).encode(), headers={"Content-Type": "application/json"})
            resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
            raw = resp.get("content", "")
            match = re.search(r'\{.*"soru".*:.*"cevap".*:.*\}', raw, re.S)
            if match:
                data = json.loads(match.group())
                qa_records.append({"question": data["soru"], "answer": data["cevap"], "source": "synthetic"})
                complex_records.append({"text": data["soru"], "label": 2, "label_name": "complex"})
                generated += 1
                if generated % 10 == 0: logger.info(f"Generated: {generated}")
        except Exception: continue

    save_jsonl(qa_records, RAW_DIR / "synthetic_qa.jsonl", "SynthQA")
    save_jsonl(complex_records, RAW_DIR / "router_synth_complex.jsonl", "SynthComplex")

# --- SUBCOMMAND: augment-bench ---
def cmd_augment_bench(args):
    """Augment finetune dataset with benchmark data."""
    logger.info("Augmenting finetune data...")
    new_data = []

    # MMLU dev
    p = BENCHMARKS_DIR / "tr_mmlu" / "dev.jsonl"
    if p.exists():
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            inst = f"Soru: {r['question']}\nA) {r['choices']['A']}\nB) {r['choices']['B']}\nC) {r['choices']['C']}\nD) {r['choices']['D']}\nCevap:"
            new_data.append({"source": "bench/mmlu", "instruction": inst, "response": r["answer"]})

    # Sentiment train
    p = BENCHMARKS_DIR / "bist_sentiment" / "train.jsonl"
    if p.exists():
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            inst = f"Metnin duygusunu analiz et (pozitif/negatif/nötr):\n{r['text']}"
            new_data.append({"source": "bench/sentiment", "instruction": inst, "response": r["label"]})

    # Load existing
    existing = []
    base_p = PROC_DIR / "finance_rag_train.jsonl"
    if base_p.exists():
        for line in open(base_p, encoding="utf-8"): existing.append(json.loads(line))

    combined = existing + new_data
    random.shuffle(combined)
    save_jsonl(combined, PROC_DIR / "finetune_combined_v2.jsonl", "AugmentedFinetune")

# --- SUBCOMMAND: augment-router ---
def cmd_augment_router(args):
    """Augment router training data with templates and external sources."""
    logger.info(f"Augmenting router data to {args.target_per_class} per class...")

    COMPLEX_TEMPLATES = [
        "{enstruman} ile {enstruman2} arasındaki {konu} farkını analiz et.",
        "{firma} için DCF modeli kur, {yil} projeksiyonu yap.",
        "{endeks} son {sure} performansını {kriter} açısından değerlendir.",
        "{enstruman} portföyüne {strateji} stratejisi nasıl uygulanır?",
        "Türkiye'nin {konu} politikasının {etki} üzerindeki etkilerini açıkla.",
        "{firma} bilançosunda EBITDA marjını {yil} verileriyle hesapla.",
        "{sektor} sektöründe faaliyet gösteren {firma} için WACC hesapla.",
        "Monte Carlo simülasyonu ile {enstruman} risk analizi yap.",
        "{konu} şoklarının banka sermaye yeterliliğine yansımasını incele.",
        "FedProx ile FedAvg'nin {konu} senaryosunda neden farklı performans gösterdiğini açıkla.",
        "Yield curve inversiyonu ile resesyon arasındaki korelasyonu {yil} verileriyle analiz et.",
        "{firma} için {konu} bazlı senaryo analizi oluştur.",
        "Türk lirası volatilitesinin ihracatçı {sektor} şirket değerlemesine etkisi.",
        "Enflasyon beklentisini {hesap_yontemi} ile hesapla ve politika faizine yansımasını yorumla.",
        "Uzun vadeli Eurobond getirisi ile CDS spreadleri arasındaki arbitraj fırsatlarını değerlendir.",
    ]

    SIMPLE_TEMPLATES = [
        "{enstruman} nedir?",
        "{enstruman} nasıl çalışır?",
        "{kavram} ne demektir?",
        "{banka} konut kredisi faizi ne kadar?",
        "{banka} mevduat faizi kaç puan?",
        "Repo ile mevduat arasındaki fark nedir?",
        "{enstruman} ne zaman alınır?",
        "{kavram} hesabı nasıl açılır?",
        "{kavram} için vergi oranı nedir?",
        "BIST 100 endeksi ne anlama gelir?",
        "Döviz kuru nasıl okunur?",
        "{kavram} hesaplama formülü nedir?",
        "{banka} kredi başvurusu nasıl yapılır?",
        "Vadeli mevduat ne kadar süre için açılır?",
        "Devlet tahvili ve bono arasındaki fark nedir?",
    ]

    CHITCHAT_TEMPLATES = [
        "Merhaba!", "Nasılsın?", "İyi günler.", "Teşekkürler.", "Günaydın.",
        "Anlamadım, tekrar açıklar mısın?", "Tamam, anladım.", "Harika bir gün geçiriyorum.",
        "Seni çok yordum galiba.", "Biraz dinlen.", "Hangi konu hakkında yardım edebilirsin?",
        "Neler yapabilirsin?", "Çok teşekkür ederim yardımın için.", "Görüşmek üzere!",
    ]

    FILLS = {
        "enstruman": ["hisse senedi", "tahvil", "bono", "eurobond", "opsiyon", "vadeli işlem", "döviz", "altın", "fon", "BIST 30 endeksi"],
        "enstruman2": ["altın", "döviz", "tahvil", "bono", "repo", "özel sektör tahvili", "hazine bonosu"],
        "konu": ["faiz riski", "kur riski", "getiri", "likidite", "volatilite", "kredi riski", "enflasyon", "büyüme", "cari açık"],
        "firma": ["Garanti BBVA", "İş Bankası", "Ziraat Bankası", "Akbank", "Yapı Kredi", "Halkbank", "Vakıfbank", "Sabancı Holding", "Koç Holding", "Turkcell", "TTKOM", "BİM"],
        "yil": ["2019", "2020", "2021", "2022", "2023", "2024"],
        "sure": ["6 aylık", "1 yıllık", "3 yıllık", "5 yıllık"],
        "kriter": ["Sharpe oranı", "Sortino oranı", "beta değeri", "alpha", "maksimum drawdown"],
        "strateji": ["momentum", "değer yatırımı", "büyüme odaklı", "hedge", "çeşitlendirme"],
        "endeks": ["BIST 100", "BIST 30", "BIST Mali", "S&P 500", "MSCI EM"],
        "sektor": ["bankacılık", "enerji", "teknoloji", "perakende", "sanayi", "gıda", "inşaat"],
        "etki": ["enflasyon", "döviz kuru", "büyüme", "reel ücretler", "ihracatçı sektörler"],
        "hesap_yontemi": ["breakeven enflasyon yöntemi", "ankete dayalı tahmin", "VAR modeli", "DSGE modeli"],
        "kavram": ["faiz", "mevduat", "yatırım hesabı", "reeskont", "politika faizi", "zorunlu karşılık", "Eurobond", "interbank faizi", "kaldıraç oranı"],
        "banka": ["Ziraat Bankası", "Halkbank", "Vakıfbank", "İş Bankası", "Garanti BBVA", "Akbank", "Yapı Kredi", "QNB Finansbank"],
    }

    def render(template):
        for key, choices in FILLS.items():
            placeholder = "{" + key + "}"
            if placeholder in template:
                template = template.replace(placeholder, random.choice(choices), 1)
        if "{enstruman2}" in template:
            template = template.replace("{enstruman2}", random.choice(FILLS["enstruman"]), 1)
        return template

    existing = []
    if (PROC_DIR / "router_train.jsonl").exists():
        for line in open(PROC_DIR / "router_train.jsonl", encoding="utf-8"):
            existing.append(json.loads(line))

    by_label = {0: [], 1: [], 2: []}
    for r in existing: by_label[r["label"]].append(r)

    configs = [(0, CHITCHAT_TEMPLATES), (1, SIMPLE_TEMPLATES), (2, COMPLEX_TEMPLATES)]
    label_names = {0: "chitchat", 1: "simple", 2: "complex"}

    for label, templates in configs:
        while len(by_label[label]) < args.target_per_class:
            t = render(random.choice(templates))
            by_label[label].append({
                "source": "synthetic_template",
                "text": t,
                "label": label,
                "label_name": label_names[label]
            })

    combined = by_label[0] + by_label[1] + by_label[2]
    random.shuffle(combined)
    (PROC_DIR / "router_augmented").mkdir(exist_ok=True)
    save_jsonl(combined, PROC_DIR / "router_augmented" / "router_train_aug.jsonl", "AugmentedRouter")

# --- SUBCOMMAND: split ---
def cmd_split(args):
    """Split RAG corpus into Branch-1 and Branch-2 (Non-IID)."""
    logger.info(f"Splitting RAG corpus with overlap_ratio={args.overlap_ratio}...")

    B1_KWS = {"kredi", "bireysel", "konut", "mevduat", "kart", "limit", "faiz"}
    B2_KWS = {"kurumsal", "yatırım", "hisse", "borsa", "sermaye", "döviz", "kur"}

    all_recs = []
    for p in [PROC_DIR / "finance_rag_train.jsonl", PROC_DIR / "fin_news_train.jsonl"]:
        if p.exists():
            for line in open(p, encoding="utf-8"):
                r = json.loads(line); r["_text"] = (r.get("text") or r.get("response") or "").lower()
                all_recs.append(r)

    b1_pool, b2_pool, shared_pool = [], [], []
    for r in all_recs:
        s1 = sum(1 for kw in B1_KWS if kw in r["_text"])
        s2 = sum(1 for kw in B2_KWS if kw in r["_text"])
        if s1 > s2: b1_pool.append(r)
        elif s2 > s1: b2_pool.append(r)
        else: shared_pool.append(r)

    random.shuffle(shared_pool)
    n_overlap = int(len(shared_pool) * args.overlap_ratio)
    overlap = shared_pool[:n_overlap]
    remaining = shared_pool[n_overlap:]

    b1_final = b1_pool + remaining[:len(remaining)//2] + overlap
    b2_final = b2_pool + remaining[len(remaining)//2:] + overlap

    for d, name, data in [(BRANCH1_DIR, "corpus", b1_final), (BRANCH2_DIR, "corpus", b2_final), (HQ_DIR, "corpus", all_recs)]:
        d.mkdir(parents=True, exist_ok=True)
        save_jsonl([{k:v for k,v in r.items() if k != "_text"} for r in data], d / f"{name}.jsonl", name)

# --- SUBCOMMAND: index ---
def cmd_index(args):
    """Build ChromaDB indexes."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("chromadb or sentence-transformers not installed.")
        return

    device = "cuda" if args.use_gpu else "cpu"
    logger.info(f"Building ChromaDB indexes on {device}...")
    model = SentenceTransformer(args.embed_model, device=device, trust_remote_code=True)
    model.max_seq_length = 256

    def build(path, name, file_path):
        client = chromadb.PersistentClient(path=str(path))
        try: client.delete_collection(name)
        except Exception: pass
        coll = client.create_collection(name, metadata={"hnsw:space": "cosine"})

        recs = []
        if file_path.exists():
            for line in open(file_path, encoding="utf-8"): recs.append(json.loads(line))

        texts = [r.get("text") or r.get("response") or "" for r in recs]
        prefixed = [f"passage: {t}" for t in texts]

        batch_size = 64
        for i in range(0, len(prefixed), batch_size):
            b_texts = texts[i:i+batch_size]
            b_pref = prefixed[i:i+batch_size]
            b_ids = [f"doc_{i+j}" for j in range(len(b_texts))]
            b_metas = [{"source": "rag"} for _ in b_texts]
            embeds = model.encode(b_pref, normalize_embeddings=True).tolist()
            coll.add(documents=b_texts, embeddings=embeds, ids=b_ids, metadatas=b_metas)
            if (i + batch_size) % 256 == 0: logger.info(f"  {name}: {i+batch_size}/{len(texts)}")
        logger.info(f"  {name} complete: {len(texts)} docs")

    build(CHROMA_DIR / "global", "finance_global", HQ_DIR / "corpus.jsonl")
    build(CHROMA_DIR / "branch1", "finance_branch1", BRANCH1_DIR / "corpus.jsonl")
    build(CHROMA_DIR / "branch2", "finance_branch2", BRANCH2_DIR / "corpus.jsonl")

# --- MAIN CLI ---
def main():
    parser = argparse.ArgumentParser(description="MOSAIC-v2 Data Engineering Tool")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # collect-hf
    subparsers.add_parser("collect-hf", help="Collect datasets from HuggingFace")

    # collect-kap
    p_kap = subparsers.add_parser("collect-kap", help="Collect KAP disclosures")
    p_kap.add_argument("--year", type=int, default=2024)
    p_kap.add_argument("--max-pages", type=int, default=10)

    # collect-news
    p_news = subparsers.add_parser("collect-news", help="Collect finance news")
    p_news.add_argument("--articles-per-site", type=int, default=200)

    # collect-bench
    p_bench = subparsers.add_parser("collect-bench", help="Collect benchmark datasets")
    p_bench.add_argument("--max-mmlu-test", type=int, default=2000)
    p_bench.add_argument("--max-sentiment-test", type=int, default=1500)
    p_bench.add_argument("--max-sentiment-train", type=int, default=1000)

    # clean
    subparsers.add_parser("clean", help="Clean and preprocess all raw data")

    # synth-qa
    p_sqa = subparsers.add_parser("synth-qa", help="Generate synthetic Q&A")
    p_sqa.add_argument("--num-qa", type=int, default=300)
    p_sqa.add_argument("--llm-url", default="http://127.0.0.1:8099")

    # augment-bench
    subparsers.add_parser("augment-bench", help="Augment finetune data with benchmarks")

    # augment-router
    p_ar = subparsers.add_parser("augment-router", help="Augment router data")
    p_ar.add_argument("--target-per-class", type=int, default=10000)

    # split
    p_split = subparsers.add_parser("split", help="Non-IID split of RAG corpus")
    p_split.add_argument("--overlap-ratio", type=float, default=0.25)

    # index
    p_idx = subparsers.add_parser("index", help="Build ChromaDB indexes")
    p_idx.add_argument("--embed-model", default="newmindai/TurkEmbed4Retrieval")
    p_idx.add_argument("--use-gpu", action="store_true")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cmd_map = {
        "collect-hf": cmd_collect_hf,
        "collect-kap": cmd_collect_kap,
        "collect-news": cmd_collect_news,
        "collect-bench": cmd_collect_bench,
        "clean": cmd_clean,
        "synth-qa": cmd_synth_qa,
        "augment-bench": cmd_augment_bench,
        "augment-router": cmd_augment_router,
        "split": cmd_split,
        "index": cmd_index,
    }

    start_time = time.time()
    logger.info(f"Starting {args.command}...")
    cmd_map[args.command](args)
    logger.info(f"Finished {args.command} in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
