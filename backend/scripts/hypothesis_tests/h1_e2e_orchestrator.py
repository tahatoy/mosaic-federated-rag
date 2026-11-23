"""
Uçtan uca çapraz şube deneyi (savunma sonrası iyileştirme ölçümü).

H1 deneyindeki sorular (h1_federated_rag_eval.build_questions: yalnızca bir şubede bulunan cümlelerden
üretilen sayısal boşluk doldurma soruları, yanıtı İÇERMEYEN şubeden sorulur) bu kez doğrudan canlı
orkestratöre (/query) gönderilir: yönlendirici, HQ koleksiyonu, şube servisleri (mTLS), yeniden sıralama
ve dil modeli dahil tüm hat ölçülür. Kullanıcı arayüzündeki gibi doğal soru sorulur: boşluklu cümlede
"___" yerine "kaç" konur ve soru işaretiyle bitirilir (talimat kalıbı aramayı bozduğundan kullanılmaz).

Ölçülenler: doğruluk (altın sayı yanıtta mı), rota dağılımı, bağlama giren parçaların kaynağı
(sahip şube / diğer şube / HQ koleksiyonu).
Kullanım: python h1_e2e_orchestrator.py --n 100 --tag once
Çıktı: ai/experiments/results/h1_e2e_<tag>.json
"""
import argparse, collections, json, sys, time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h1_federated_rag_eval import build_questions, norm_num  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100); ap.add_argument("--tag", required=True)
ap.add_argument("--orch", default="http://127.0.0.1:8004/query")
a = ap.parse_args()


def origin(branch):
    b = str(branch or "").lower()
    if b in ("b1", "branch_01", "1"): return 1
    if b in ("b2", "branch_02", "2"): return 2
    return "hq"


qs = build_questions(a.n)
res = {"n": len(qs), "dogru": 0, "rota": collections.Counter(), "kaynak_yok": 0,
       "dogru_ve_sahip_sube_kaynakta": 0, "kaynaklarda_hq_koleksiyonu": 0, "sure_ms": [], "kayitlar": []}
t0 = time.time()
with httpx.Client(timeout=300) as cl:
    for i, q in enumerate(qs, 1):
        soru = q["search_query"].replace("___", "kaç").strip().rstrip(".") + "?"
        d = cl.post(a.orch, json={"query": soru, "branch_id": q["asked_branch"]}).json()
        ans = d.get("answer", "")
        ok = int(q["gold_norm"] in norm_num(ans))
        origins = [origin(s.get("branch")) for s in d.get("sources", [])]
        res["dogru"] += ok; res["rota"][d.get("route")] += 1; res["sure_ms"].append(d.get("elapsed_ms", 0))
        res["kaynak_yok"] += int(not origins)
        res["kaynaklarda_hq_koleksiyonu"] += int("hq" in origins)
        res["dogru_ve_sahip_sube_kaynakta"] += int(ok and q["owner_branch"] in origins)
        res["kayitlar"].append({"soru": soru, "altin": q["gold"], "sahip": q["owner_branch"], "sorulan": q["asked_branch"],
                                "rota": d.get("route"), "kaynaklar": origins, "dogru": ok, "yanit": ans[:300]})
        if i % 20 == 0: print(i, "doğruluk", round(res["dogru"] / i, 3), flush=True)
n = len(qs)
res["dogruluk"] = round(res["dogru"] / n, 4)
res["rota"] = dict(res["rota"])
res["ortalama_sure_ms"] = round(sum(res["sure_ms"]) / n, 1); del res["sure_ms"]
res["toplam_sure_s"] = round(time.time() - t0, 1)
out = ROOT / "ai" / "experiments" / "results" / f"h1_e2e_{a.tag}.json"
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "kayitlar"}, ensure_ascii=False, indent=1))
