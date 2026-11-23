"""Sıfır-örnek kıyaslama, modelin KENDİ sohbet şablonuyla (/v1/chat/completions).
eval_tool.py bench istemleri her modele Qwen ChatML şablonunu ham /completion ile gönderiyor;
bu betik Gemma gibi farklı şablonlu modeller için adil karşılaştırma sağlar.
Duygu: veri kümesi yalnızca pozitif/negatif içerdiğinden İKİLİ istem kullanılır (alan-nötr).
Kullanım: python bench_chat_native.py --url http://127.0.0.1:8099 --name gemma
"""
import argparse, json, re, time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[3]
B = ROOT / "ai" / "data" / "benchmarks"
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--n", type=int, default=500)
a = ap.parse_args()

def chat(cl, system, user, max_tokens):
    t0 = time.perf_counter()
    r = cl.post(f"{a.url.rstrip('/')}/v1/chat/completions", json={
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False}}, timeout=120)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"] or ""
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip(), time.perf_counter() - t0

def letter(s):
    s = s.strip()
    if s and s[0].upper() in "ABCD": return s[0].upper()
    m = re.search(r"\b([ABCD])\b", s)
    return m.group(1).upper() if m else None

def senti(s):
    s = s.lower()
    p, n = ("pozitif" in s or "olumlu" in s or "positive" in s), ("negatif" in s or "olumsuz" in s or "negative" in s)
    return "pozitif" if p and not n else "negatif" if n and not p else None

out = {"model": a.name, "url": a.url, "template": "model-native (/v1/chat/completions)"}
with httpx.Client() as cl:
    recs = [json.loads(l) for l in open(B / "tr_mmlu" / "test.jsonl", encoding="utf-8")][:a.n]
    sysm = "Sen Türkçe akademik sınavlar konusunda uzmanlaşmış bir yapay zeka asistanısın. Her soru için dört seçenekten (A, B, C, D) birini seç ve SADECE o harfi yaz."
    corr = unp = 0; lat = 0.0
    for i, r in enumerate(recs):
        c = r["choices"]
        u = f"{r['question']}\n\nA) {c['A']}\nB) {c['B']}\nC) {c['C']}\nD) {c['D']}\n\nCevap (A/B/C/D):"
        try: ans, dt = chat(cl, sysm, u, 16)
        except Exception as e: ans, dt = "", 0.0
        p = letter(ans); lat += dt
        if p is None: unp += 1
        corr += int(p == r.get("answer", r.get("target")))
        if (i + 1) % 100 == 0: print(f"[{a.name}] MMLU {i+1}/{len(recs)} acc={corr/(i+1):.3f}", flush=True)
    out["tr_mmlu"] = {"total": len(recs), "correct": corr, "accuracy": round(corr / len(recs), 4),
                      "unparseable": unp, "avg_latency_s": round(lat / len(recs), 4)}

    recs = [json.loads(l) for l in open(B / "bist_sentiment" / "test.jsonl", encoding="utf-8")][:a.n]
    syss = "Verilen Türkçe metnin duygusunu sınıflandır. SADECE tek kelime yaz: pozitif veya negatif."
    yt, yp, unp, lat = [], [], 0, 0.0
    for r in recs:
        try: ans, dt = chat(cl, syss, f"Metin: {r['text'][:500]}\n\nEtiket:", 8)
        except Exception: ans, dt = "", 0.0
        p = senti(ans); lat += dt
        if p is None: unp += 1; p = "pozitif" if r["label"] == "negatif" else "negatif"  # ayrıştırılamayan = yanlış
        yt.append(r["label"]); yp.append(p)
    from sklearn.metrics import f1_score, accuracy_score
    out["sentiment_binary"] = {"total": len(recs), "labels": {l: yt.count(l) for l in set(yt)},
        "macro_f1": round(f1_score(yt, yp, average="macro"), 4), "accuracy": round(accuracy_score(yt, yp), 4),
        "unparseable_counted_wrong": unp, "avg_latency_s": round(lat / len(recs), 4),
        "majority_baseline_acc": round(max(yt.count(l) for l in set(yt)) / len(yt), 4)}
print(json.dumps(out, ensure_ascii=False, indent=1))
p = ROOT / "ai" / "experiments" / "results" / f"bench_native_{a.name}.json"
p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
