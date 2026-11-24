"""
E18 — Yönlendiricinin rapor dilindeki sorularla yeniden eğitilmesi (yönlendirici v2).

E9, ilk yönlendiricinin tek bir değeri soran çapraz şube sorularının %99.7'sini karmaşık sınıfa attığını gösterdi:
basit sınıf kısa müşteri hizmetleri şablonlarından, karmaşık sınıf ise şirket adı, yıl ve tutar içeren analitik
şablonlardan üretildiği için model sorunun gerektirdiği çabayı değil ifade biçimini öğrenmişti.

Bu betik:
 1) Değerlendirme sorularının kaynak parçalarıyla örtüşmeyen şube parçalarından (E12a'da kullanılan 7.113 parça)
    rapor dilinde iki tür soru toplar: E12a'nın doğal soruları ve boşluk doldurma sorguları (tek bir bilgiyi sorar ->
    basit) ve Qwen3-8B'nin aynı parçalar için yazdığı, birden fazla bilgiyi birleştirmeyi, hesaplamayı ya da
    karşılaştırmayı gerektiren sorular (-> karmaşık; parçada en az iki sayı varsa hesaplama, yoksa çok adımlı
    akıl yürütme istenir). Böylece iki sınıf da rapor dilinde örnek içerir ve ifade biçimi sınıfı ele vermez.
 2) Etiket denetimi: Qwen3-8B her soruyu, sınıf tanımlarından habersiz, "tek bilgi yeterli (A)" ya da "birden fazla
    bilgi/hesaplama/çok adımlı akıl yürütme gerekir (B)" olarak ayrıca sınıflandırır; üretim amacıyla bu kararın
    çeliştiği sorular atılır.
 3) Parçalar %80/%20 bölünür; eğitim parçalarının soruları özgün eğitim kümesine eklenir (v2), test parçalarının
    soruları eğitimde hiç görülmeyen rapor dili test kümelerini oluşturur.
 4) v2, ilk yönlendiriciyle aynı ayarlarla (önceden eğitilmiş BERTurk, 5 dönem, parti 32, öğrenme oranı 2e-5,
    en fazla 128 token) üç tohumla eğitilir; TF-IDF + lojistik regresyon tabanı da iki veri kümesiyle eğitilir.
 5) Ölçülen kümeler: özgün şablon testi (3.000), rapor dili basit/karmaşık testleri, E9'daki 600 kalıp ve 419
    yeniden ifade edilmiş çapraz şube sorusu (beklenen: basit). Canlı sistemdeki yönlendirici (bert_router_best)
    aynı kümelerde karşılaştırılır; kural katmanı (Bölüm 2.6) ham kararlara ayrıca uygulanır.
Çıktılar: ai/data/e18/complex_queries.jsonl (korpustan türetilmiş, dağıtılmaz), ai/models/router/bert_router_v2
          (tohum 0; canlı sisteme konuşlandırılmaz), ai/experiments/results/q1/e18_router_v2.json
"""
import argparse
import json
import random
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx
import torch

from q1_common import OUT_DIR, ROOT, questions, save
from e4_router_baselines_fl import BASE, DEV, load, scores, train_bert

E12 = ROOT / "ai" / "data" / "e12" / "train_queries.jsonl"
E18 = ROOT / "ai" / "data" / "e18"
NUMS = re.compile(r"\d+(?:[.,]\d+)*")
NAMES = {0: "chitchat", 1: "simple", 2: "complex"}

PROMPT = (
    "Görev: Aşağıdaki finans/bankacılık metin parçasını okuyan bir banka analistinin yazacağı, cevabı için parçadaki "
    "EN AZ İKİ sayısal bilgiyi birlikte kullanmayı gerektiren TEK bir soru yaz.\n"
    "Kurallar: (1) Soru oran, fark, yüzde değişim, büyüme, pay, karşılaştırma ya da bir hesaplama gerektirmeli; tek bir "
    "değeri doğrudan sormamalı. (2) Soru parçayı görmeyen biri için de anlaşılır olmalı; varsa kurum veya şirket adını "
    "ve dönemi içermeli. (3) Parçadan beş sözcükten uzun bir diziyi aynen kopyalama. (4) Cevabı yazma. "
    "(5) Başına 'Soru:' gibi bir ön ek koyma, yalnızca soruyu yaz.\n\nParça:\n\"\"\"\n{chunk}\n\"\"\"\n/no_think"
)
PROMPT_REASON = (
    "Görev: Aşağıdaki finans/bankacılık metin parçasını okuyan bir banka analistinin yazacağı, cevabı için parçadaki "
    "BİRDEN FAZLA bilgiyi birleştirmeyi ve çok adımlı akıl yürütmeyi (neden-sonuç, karşılaştırma, etki değerlendirmesi "
    "ya da hesaplama) gerektiren TEK bir soru yaz.\n"
    "Kurallar: (1) Soru tek bir bilgiyi doğrudan sormamalı. (2) Soru parçayı görmeyen biri için de anlaşılır olmalı; "
    "varsa kurum veya şirket adını içermeli. (3) Parçadan beş sözcükten uzun bir diziyi aynen kopyalama. (4) Cevabı "
    "yazma. (5) Başına 'Soru:' gibi bir ön ek koyma, yalnızca soruyu yaz.\n\nParça:\n\"\"\"\n{chunk}\n\"\"\"\n/no_think"
)
JUDGE = (
    "Aşağıdaki bankacılık sorusunu sınıflandır.\n"
    "A: Cevap, bir belgede yazan TEK bir bilgiyi (bir değer, tarih, ad, kural ya da tanım) bulmakla verilebilir.\n"
    "B: Cevap için birden fazla bilgiyi birleştirmek, hesaplama yapmak, karşılaştırmak ya da çok adımlı akıl yürütmek gerekir.\n"
    "Soru: {q}\nYalnızca A veya B yaz. /no_think"
)


def clean(txt):
    t = (txt or "").strip().strip('"').strip()
    t = re.sub(r"^(soru|question)\s*[:：]\s*", "", t, flags=re.I).strip().strip('"').strip()
    return t.split("\n")[0].strip().strip('"').strip()


def valid(q):
    return 20 < len(q) < 300 and q.endswith("?") and "yaz" not in q.lower()[:15] and "___" not in q


def gen_complex(docs, picks, url, workers, max_minutes):
    out_path = E18 / "complex_queries.jsonl"
    done = {}
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            r = json.loads(line); done[r["doc"]] = r
    # geçersiz kalan önceki üretimler (ör. kesilmiş yanıtlar) bir kez yeniden denenir
    todo = [i for i in picks if i not in done or (not valid(clean(done[i]["query"])) and done[i].get("try", 1) < 2)]
    print(f"karmaşık soru üretimi: hazır {len(done)}, üretilecek {len(todo)}", flush=True)
    t0 = time.time()

    def ask(cl, i):
        try:
            pr = PROMPT if len(set(NUMS.findall(docs[i]))) >= 2 else PROMPT_REASON
            r = cl.post(url, json={"messages": [{"role": "user", "content": pr.format(chunk=docs[i][:1400])}],
                                   "max_tokens": 160, "temperature": 0.3})
            txt = r.json()["choices"][0]["message"]["content"].split("</think>")[-1].strip().strip('"')
            return i, clean(txt)
        except Exception:  # tek bir isteğin hatası üretimi durdurmasın
            return i, ""

    E18.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=180) as cl, out_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        for s in range(0, len(todo), workers * 8):
            if (time.time() - t0) / 60 > max_minutes:
                print("süre sınırı", flush=True); break
            for i, txt in ex.map(lambda i: ask(cl, i), todo[s:s + workers * 8]):
                rec = {"doc": i, "query": txt, "try": (done[i].get("try", 1) + 1) if i in done else 1}
                done[i] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (s // (workers * 8)) % 10 == 0:
                print(min(s + workers * 8, len(todo)), "/", len(todo), f"{(time.time() - t0) / 60:.1f} dk", flush=True)
    for r in done.values():
        r["query"] = clean(r["query"]); r["ok"] = valid(r["query"])
    return done


def judge(items, url, workers, max_minutes):
    """items: list of (key, question). Returns {key: 'A'|'B'|None}; cached in ai/data/e18/judge.jsonl."""
    path = E18 / "judge.jsonl"
    done = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            r = json.loads(line); done[r["key"]] = r["v"]
    todo = [(k, q) for k, q in items if k not in done]
    print(f"etiket denetimi: hazır {len(done)}, sorulacak {len(todo)}", flush=True)
    t0 = time.time()

    def ask(cl, kq):
        k, q = kq
        try:
            r = cl.post(url, json={"messages": [{"role": "user", "content": JUDGE.format(q=q)}], "max_tokens": 8, "temperature": 0.0})
            t = r.json()["choices"][0]["message"]["content"].split("</think>")[-1].strip().upper()
            return k, ("A" if t.startswith("A") else "B" if t.startswith("B") else None)
        except Exception:
            return k, None

    with httpx.Client(timeout=120) as cl, path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        for s in range(0, len(todo), workers * 16):
            if (time.time() - t0) / 60 > max_minutes:
                print("süre sınırı (denetim)", flush=True); break
            for k, v in ex.map(lambda kq: ask(cl, kq), todo[s:s + workers * 16]):
                done[k] = v; f.write(json.dumps({"key": k, "v": v}, ensure_ascii=False) + "\n")
            if (s // (workers * 16)) % 10 == 0:
                print("denetim", min(s + workers * 16, len(todo)), "/", len(todo), f"{(time.time() - t0) / 60:.1f} dk", flush=True)
    return done


def predict_proba(m, tok, T):
    m.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(T), 128):
            e = tok(T[i:i + 128], max_length=128, padding=True, truncation=True, return_tensors="pt").to(DEV)
            P += torch.softmax(m(**e).logits.float(), -1).cpu().tolist()
    return P


def decide(texts, probs, rules):
    import sys
    sys.path.insert(0, str(ROOT / "backend"))
    from ai_engine import _apply_router_overrides
    raw = [max(range(3), key=lambda k: p[k]) for p in probs]
    if not rules:
        return raw
    return [_apply_router_overrides(t, {"label": l, "confidence": p[l]})["label"] for t, l, p in zip(texts, raw, probs)]


def evaluate(pred_fn, sets):
    res = {}
    for name, (T, Y) in sets.items():
        P = pred_fn(T)
        r = {"n": len(Y), "acc": round(sum(int(a == b) for a, b in zip(P, Y)) / len(Y), 4),
             "share": {NAMES[k]: round(v / len(P), 4) for k, v in sorted(Counter(P).items())}}
        if len(set(Y)) > 1:
            r.update(scores(Y, P))
        res[name] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=5000)
    ap.add_argument("--url", default="http://127.0.0.1:8098/v1/chat/completions")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-gen-minutes", type=float, default=30)
    ap.add_argument("--max-judge-minutes", type=float, default=25)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--gen-only", action="store_true")
    a = ap.parse_args()

    # ---- 1) rapor dilinde sorular
    qs, docs, _ = questions(600)
    nat, clz = {}, {}
    for line in E12.open(encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] == "natural":
            nat.setdefault(r["doc"], r["query"])
        elif r["kind"] == "cloze":
            clz.setdefault(r["doc"], r["query"].replace("___", "kaç").strip().rstrip(".") + "?")
    cand = sorted(nat)
    rnd = random.Random(18); rnd.shuffle(cand)
    picks = cand[:a.n_docs]
    gen = gen_complex(docs, picks, a.url, a.workers, a.max_gen_minutes)
    items = [(f"s{i}", nat[i]) for i in picks] + [(f"c{i}", gen[i]["query"]) for i in picks if i in gen and gen[i]["ok"]]
    jd = judge(items, a.url, a.workers, a.max_judge_minutes)
    simple = {i: nat[i] for i in picks if jd.get(f"s{i}") == "A"}
    complex_ = {i: gen[i]["query"] for i in picks if i in gen and gen[i]["ok"] and jd.get(f"c{i}") == "B"}
    stats = {"docs": len(picks), "generated_valid": sum(1 for i in picks if i in gen and gen[i]["ok"]),
             "simple_kept": len(simple), "simple_judged_B": sum(1 for i in picks if jd.get(f"s{i}") == "B"),
             "complex_kept": len(complex_), "complex_judged_A": sum(1 for i in picks if jd.get(f"c{i}") == "A"),
             "cloze": len(clz)}
    print("veri:", stats, flush=True)
    if a.gen_only:
        return
    held = lambda i: i % 5 == 0
    trS = [q for i, q in simple.items() if not held(i)] + [q for i, q in clz.items() if not held(i)]
    trC = [q for i, q in complex_.items() if not held(i)]
    # denge: rapor dilinde eklenen basit ve karmaşık soru sayısı eşitlenir, ifade biçimi bir sınıfı kayırmasın
    nb = min(len(trS), len(trC)); rb = random.Random(7); rb.shuffle(trS); rb.shuffle(trC)
    trS, trC = trS[:nb], trC[:nb]

    # ---- 2) veri kümeleri
    trT, trY = load("router_train.jsonl"); teT, teY = load("router_test.jsonl")
    v2T = trT + trS + trC
    v2Y = trY + [1] * len(trS) + [2] * len(trC)
    para = {r["reference"]: r["natural"] for r in json.loads((OUT_DIR / "e0_paraphrases.json").read_text(encoding="utf-8"))["kayitlar"] if r["natural"]}
    cloze = [q["search_query"].replace("___", "kaç").strip().rstrip(".") + "?" for q in qs]
    paraq = [para[q["reference"]] for q in qs if q["reference"] in para]
    teS = [q for i, q in simple.items() if held(i)]; teC = [q for i, q in complex_.items() if held(i)]
    teZ = [q for i, q in clz.items() if held(i)]
    sets = {
        "template_test": (teT, teY),
        "report_simple_heldout": (teS, [1] * len(teS)),
        "report_complex_heldout": (teC, [2] * len(teC)),
        "report_cloze_heldout": (teZ, [1] * len(teZ)),
        "crossbranch_cloze": (cloze, [1] * len(cloze)),
        "crossbranch_paraphrased": (paraq, [1] * len(paraq)),
    }
    out = {"data": {**stats, "original_train": len(trY), "added_simple_train": len(trS), "added_complex_train": len(trC),
                    "v2_train": len(v2Y), "v2_train_classes": dict(Counter(v2Y)),
                    "split": "by chunk (doc id mod 5 == 0 held out); held-out chunks never seen in training; the 600 evaluation questions' source chunks were excluded in E12a",
                    "label_check": "Qwen3-8B A/B judgement; generated questions kept only when it agrees with the intended class",
                    "examples": {"simple": teS[:5], "complex": teC[:5]}},
           "eval_sets": {k: len(v[1]) for k, v in sets.items()}, "models": {}}

    # ---- 3) TF-IDF tabanları
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union
    def tfidf(T, Y):
        return make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
                             LogisticRegression(max_iter=2000, C=4.0)).fit(T, Y)
    for name, (T, Y) in (("tfidf_v1_data", (trT, trY)), ("tfidf_v2_data", (v2T, v2Y))):
        m = tfidf(T, Y)
        out["models"][name] = evaluate(lambda X: m.predict(X).tolist(), sets)
        print(name, json.dumps({k: v["acc"] for k, v in out["models"][name].items()}), flush=True)

    # ---- 4) canlı sistemdeki yönlendirici (v1)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    v1p = ROOT / "ai" / "models" / "router" / "bert_router_best"
    tok1 = AutoTokenizer.from_pretrained(str(v1p))
    m1 = AutoModelForSequenceClassification.from_pretrained(str(v1p)).to(DEV)
    for rules in (False, True):
        k = "bert_v1_live" + ("_with_rules" if rules else "")
        out["models"][k] = evaluate(lambda X: decide(X, predict_proba(m1, tok1, X), rules), sets)
        print(k, json.dumps({s: v["acc"] for s, v in out["models"][k].items()}), flush=True)
    del m1; torch.cuda.empty_cache()

    # ---- 5) v2, üç tohum
    tok = AutoTokenizer.from_pretrained(BASE)
    runs = {"raw": [], "with_rules": []}
    for seed in [int(s) for s in a.seeds.split(",")]:
        t0 = time.time()
        m = train_bert(tok, None, v2T, v2Y, a.epochs, seed)
        print(f"v2 tohum {seed}: {(time.time() - t0) / 60:.1f} dk", flush=True)
        for rules in (False, True):
            r = evaluate(lambda X: decide(X, predict_proba(m, tok, X), rules), sets)
            runs["with_rules" if rules else "raw"].append(r)
            print("v2", seed, "kurallı" if rules else "ham", json.dumps({s: v["acc"] for s, v in r.items()}), flush=True)
        if seed == 0:
            dst = ROOT / "ai" / "models" / "router" / "bert_router_v2"
            m.save_pretrained(str(dst)); tok.save_pretrained(str(dst))
        del m; torch.cuda.empty_cache()
        out["models"]["bert_v2_runs"] = runs
        save("e18_router_v2.json", out)
    for kind, rs in runs.items():
        summ = {}
        for s in sets:
            accs = [r[s]["acc"] for r in rs]
            summ[s] = {"acc_mean": round(statistics.mean(accs), 4), "acc_sd": round(statistics.pstdev(accs), 4),
                       "share_mean": {c: round(statistics.mean(r[s]["share"].get(c, 0) for r in rs), 4) for c in NAMES.values()}}
        out["models"]["bert_v2_" + kind] = summ
    save("e18_router_v2.json", out)
    print("özet", json.dumps({k: {s: v.get("acc", v.get("acc_mean")) for s, v in out["models"][k].items()} for k in out["models"] if k != "bert_v2_runs"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
