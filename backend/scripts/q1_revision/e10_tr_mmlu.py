"""E10 — TR-MMLU (Bayram vd., 2025; 6.200 soru, 62 bölüm) sıfır-örnek değerlendirmesi.

Önceki MMLU-TR (malhajar/mmlu-tr, çeviri MMLU) ölçümünü TR-MMLU ile tamamlar.
İki istem koşulu, MMLU-TR deneyiyle birebir aynı:
  chatml : Qwen ChatML şablonu, ham /completion (n_predict 32, sıcaklık 0)
  native : modelin kendi sohbet şablonu, /v1/chat/completions (--jinja ile açılan sunucu)
Seçenek sayısı veri setinden okunur (A–E); istem, seçenek harflerini buna göre yazar.

Kullanım:
  python e10_tr_mmlu.py --data ai/data/benchmarks/tr_mmlu_bayram/test.parquet \
      --url http://127.0.0.1:8098 --name qwen --mode chatml
Çıktı: ai/experiments/results/q1/e10_tr_mmlu_<name>_<mode>.json
"""
import argparse, json, re, random, time, sys
from collections import defaultdict
from pathlib import Path
import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "ai" / "experiments" / "results" / "q1"
LET = "ABCDEFG"

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--url", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--mode", choices=["chatml", "native"], required=True)
ap.add_argument("--n", type=int, default=0, help="0 = tümü")
ap.add_argument("--inspect", action="store_true", help="yalnızca şemayı yazdır")
a = ap.parse_args()

df = pd.read_parquet(a.data)
cols = {c.lower(): c for c in df.columns}

def pick(*names):
    for n in names:
        if n in cols: return cols[n]
    return None

C_Q = pick("soru", "question", "prompt")
C_OPT = pick("secenekler", "seçenekler", "choices", "options")
C_ANS = pick("cevap", "answer", "dogru_cevap", "label", "target")
C_SEC = pick("bolum", "bölüm", "section", "subject", "ders", "category")
if a.inspect or not (C_Q and C_ANS):
    print("sütunlar:", list(df.columns), "satır:", len(df))
    print(df.head(3).T.to_string()[:3000])
    print("eşleme:", dict(q=C_Q, opt=C_OPT, ans=C_ANS, sec=C_SEC))
    sys.exit(0 if a.inspect else 2)

def options(row):
    if C_OPT is not None:
        v = row[C_OPT]
        if isinstance(v, str):
            try: v = json.loads(v)
            except Exception: v = [s for s in re.split(r"\n+", v) if s.strip()]
        if isinstance(v, dict): v = [v[k] for k in sorted(v)]
        return [re.sub(r"^\s*[A-Ga-g][\)\.:]\s*", "", str(x)).strip() for x in list(v)]
    out = []
    for L in LET:
        c = pick(L.lower())
        if c is None: break
        if pd.notna(row[c]) and str(row[c]).strip(): out.append(str(row[c]).strip())
    return out

def gold(row, opts):
    v = row[C_ANS]
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        i = int(v); return LET[i] if 0 <= i < len(opts) else None
    s = str(v).strip()
    if len(s) == 1 and s.upper() in LET[:len(opts)]: return s.upper()
    m = re.match(r"^\s*([A-Ga-g])[\)\.:\s]", s + " ")
    if m and m.group(1).upper() in LET[:len(opts)]: return m.group(1).upper()
    for i, o in enumerate(opts):
        if s == o: return LET[i]
    return None

recs = []
for _, row in df.iterrows():
    opts = options(row); g = gold(row, opts)
    if len(opts) < 2 or g is None: continue
    recs.append(dict(q=str(row[C_Q]).strip(), opts=opts, gold=g,
                     sec=str(row[C_SEC]) if C_SEC else "tümü"))
skipped = len(df) - len(recs)
if a.n: recs = recs[:a.n]

def system_msg(k):
    ls = ", ".join(LET[:k])
    return (f"Sen Türkçe akademik sınavlar konusunda uzmanlaşmış bir yapay zeka asistanısın. "
            f"Her soru için {k} seçenekten ({ls}) birini seç ve SADECE o harfi yaz.")

def user_msg(r):
    k = len(r["opts"])
    body = "\n".join(f"{LET[i]}) {o}" for i, o in enumerate(r["opts"]))
    return f"{r['q']}\n\n{body}\n\nCevap ({'/'.join(LET[:k])}):"

STOP = ["<|im_end|>", "<|im_start|>"]

def ask(cl, r):
    sysm, usr = system_msg(len(r["opts"])), user_msg(r)
    t0 = time.perf_counter()
    if a.mode == "chatml":
        prompt = (f"<|im_start|>system\n{sysm}<|im_end|>\n"
                  f"<|im_start|>user\n{usr} /no_think<|im_end|>\n<|im_start|>assistant\n")
        x = cl.post(f"{a.url}/completion", json={"prompt": prompt, "n_predict": 32, "temperature": 0.0,
                    "top_p": 0.95, "repeat_penalty": 1.1, "stream": False, "stop": STOP}, timeout=120)
        x.raise_for_status(); txt = x.json().get("content", "")
    else:
        x = cl.post(f"{a.url}/v1/chat/completions", json={"messages": [
                    {"role": "system", "content": sysm}, {"role": "user", "content": usr}],
                    "max_tokens": 16, "temperature": 0.0,
                    "chat_template_kwargs": {"enable_thinking": False}}, timeout=120)
        x.raise_for_status(); txt = x.json()["choices"][0]["message"]["content"] or ""
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip(), time.perf_counter() - t0

def letter(s, k):
    ok = LET[:k]; s = s.strip()
    line = s.split("\n")[0].strip() if s else ""
    if line and line[0].upper() in ok and (len(line) == 1 or not line[1].isalpha()): return line[0].upper()
    m = re.search(r"(?:CEVAP|YANIT|ANSWER)[:\s]+([A-G])", s, re.I)
    if m and m.group(1).upper() in ok: return m.group(1).upper()
    m = re.search(rf"\b([{ok}])\b", s)
    return m.group(1).upper() if m else None

def boot_ci(x, B=2000, seed=7):
    rnd = random.Random(seed); n = len(x); ms = []
    for _ in range(B):
        ms.append(sum(x[rnd.randrange(n)] for _ in range(n)) / n)
    ms.sort(); return [round(ms[int(.025 * B)], 4), round(ms[int(.975 * B) - 1], 4)]

hits, unp, lat, per = [], 0, 0.0, defaultdict(lambda: [0, 0])
rows = []
with httpx.Client() as cl:
    for i, r in enumerate(recs):
        try: ans, dt = ask(cl, r)
        except Exception as e: ans, dt = "", 0.0
        p = letter(ans, len(r["opts"])); lat += dt
        if p is None: unp += 1
        h = int(p == r["gold"]); hits.append(h)
        per[r["sec"]][0] += h; per[r["sec"]][1] += 1
        rows.append(dict(i=i, sec=r["sec"], gold=r["gold"], pred=p, raw=ans[:40]))
        if (i + 1) % 250 == 0:
            print(f"[{a.name}/{a.mode}] {i+1}/{len(recs)} acc={sum(hits)/len(hits):.4f} unp={unp}", flush=True)

k_counts = defaultdict(int)
for r in recs: k_counts[len(r["opts"])] += 1
chance = sum(c / k for k, c in k_counts.items()) / len(recs)
out = dict(model=a.name, mode=a.mode, url=a.url, data=str(a.data),
           schema=dict(q=C_Q, opt=C_OPT, ans=C_ANS, sec=C_SEC), skipped_rows=skipped,
           n=len(recs), option_counts=dict(k_counts), chance_acc=round(chance, 4),
           correct=sum(hits), accuracy=round(sum(hits) / len(recs), 4), ci95=boot_ci(hits),
           unparseable=unp, avg_latency_s=round(lat / len(recs), 4),
           per_section={s: dict(n=v[1], acc=round(v[0] / v[1], 4)) for s, v in sorted(per.items())})
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"e10_tr_mmlu_{a.name}_{a.mode}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
(OUT / f"e10_tr_mmlu_{a.name}_{a.mode}_preds.jsonl").write_text(
    "\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "per_section"}, ensure_ascii=False, indent=1))
