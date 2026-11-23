"""
E9 — Yönlendiricinin şablon dışı sorulardaki davranışı.

Yönlendirici test kümesi şablonlardan üretildiği için TF-IDF + lojistik regresyon bile %100 doğruluğa ulaşmaktadır (E4).
Bu betik, şablon dışı gerçekçi sorgularda (E1'deki 600 soru 'kaç' kalıbıyla ve 419 yeniden ifade edilmiş soru) üç
sınıflandırıcının ve kural katmanının hangi rotayı seçtiğini ölçer. Bu sorular tek bir olgusal değerin sorulduğu
aramalardır; sınıf tanımına göre beklenen rota 'basit bilgi'dir (etiket 1).
Çıktı: ai/experiments/results/q1/e9_router_ood.json
"""
import json
from collections import Counter

import httpx

from q1_common import OUT_DIR, ROOT, save
from e4_router_baselines_fl import load


def main():
    import sys
    sys.path.insert(0, str(ROOT / "backend"))
    from ai_engine import _apply_router_overrides
    from q1_common import questions
    qs, _, _ = questions(600)
    para = {r["reference"]: r["natural"] for r in json.loads((OUT_DIR / "e0_paraphrases.json").read_text(encoding="utf-8"))["kayitlar"] if r["natural"]}
    sets = {"cloze_question": [q["search_query"].replace("___", "kaç").strip().rstrip(".") + "?" for q in qs],
            "paraphrased": [para[q["reference"]] for q in qs if q["reference"] in para]}
    trT, trY = load("router_train.jsonl")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union
    tf = make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                  TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
                       LogisticRegression(max_iter=2000, C=4.0)).fit(trT, trY)
    names = {0: "chitchat", 1: "simple", 2: "complex"}
    out = {"expected": "simple (1)", "sets": {}}
    with httpx.Client(timeout=120) as cl:
        for s, texts in sets.items():
            raw = []
            for i in range(0, len(texts), 100):
                raw += cl.post("http://127.0.0.1:8000/predict/batch", json=texts[i:i + 100]).json()
            final = [_apply_router_overrides(t, {"label": int(r["label"]), "confidence": float(r.get("confidence", 1.0))})["label"] for t, r in zip(texts, raw)]
            tfp = tf.predict(texts).tolist()
            n = len(texts)
            out["sets"][s] = {
                "n": n,
                "bert_raw": {names[k]: round(v / n, 4) for k, v in Counter(int(r["label"]) for r in raw).items()},
                "bert_with_rules": {names[k]: round(v / n, 4) for k, v in Counter(final).items()},
                "tfidf_logreg": {names[k]: round(v / n, 4) for k, v in Counter(tfp).items()},
                "bert_mean_confidence": round(sum(float(r.get("confidence", 0)) for r in raw) / n, 4),
                "rule_overrides": sum(int(a != int(r["label"])) for a, r in zip(final, raw)),
            }
            print(s, json.dumps(out["sets"][s], ensure_ascii=False), flush=True)
    save("e9_router_ood.json", out)


if __name__ == "__main__":
    main()
