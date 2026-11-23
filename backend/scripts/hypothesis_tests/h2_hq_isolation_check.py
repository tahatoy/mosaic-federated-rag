"""
H2 ek ölçümü (savunma sonrası iyileştirme): genel merkezde şubeye özgü veri var mı?

HQ ChromaDB koleksiyonundaki her parça metni, şube korpuslarıyla (ai/data/splits/branch1|2) karşılaştırılır:
  * yalnız_şube1 / yalnız_şube2 : yalnızca bir şubede bulunması gereken parçaların HQ'daki kopya sayısı
  * ortak                       : iki şubede birebir bulunan genel bilgi (HQ'da tutulabilir)
Ayrıca HQ denetim izi ve orkestratör günlüğünde şubeye özgü parça metni geçip geçmediği taranır.
Kullanım: python h2_hq_isolation_check.py --hq-path ai/data/chromadb/hq_ortak --tag sonra
Çıktı: ai/experiments/results/h2_hq_izolasyon_<tag>.json
"""
import argparse, json, sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser(); ap.add_argument("--hq-path", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--collection", default="finance_global"); a = ap.parse_args()


def load(name):
    with open(ROOT / "ai" / "data" / "splits" / name / "corpus.jsonl", encoding="utf-8") as f:
        return {json.loads(l)["text"] for l in f if l.strip()}


import chromadb  # noqa: E402

t1, t2 = load("branch1"), load("branch2")
only1, only2, ortak = t1 - t2, t2 - t1, t1 & t2
docs = chromadb.PersistentClient(path=str(ROOT / a.hq_path)).get_collection(a.collection).get(include=["documents"])["documents"]
hq = set(docs)
res = {"hq_koleksiyonu": a.hq_path, "hq_parca": len(docs),
       "hq_yalniz_sube1_kopyasi": len(hq & only1), "hq_yalniz_sube2_kopyasi": len(hq & only2),
       "hq_ortak_parca": len(hq & ortak), "hq_diger": len(hq - t1 - t2),
       "sube1_yalniz_toplam": len(only1), "sube2_yalniz_toplam": len(only2), "ortak_toplam": len(ortak)}
res["hq_subeye_ozgu_toplam"] = res["hq_yalniz_sube1_kopyasi"] + res["hq_yalniz_sube2_kopyasi"]

# Günlük/denetim izi taraması: şubeye özgü parçaların ayırt edici ilk 80 karakteri HQ dosyalarında geçiyor mu?
probes = [t[:80] for t in list(only1)[:300] + list(only2)[:300] if len(t) >= 80]
scan = {}
for f in [ROOT / "logs" / "orch.log", *sorted((ROOT / "logs" / "audit").glob("*"))]:
    if f.is_file():
        txt = f.read_text(encoding="utf-8", errors="ignore")
        scan[str(f.relative_to(ROOT))] = sum(1 for p in probes if p in txt)
res["hq_gunluk_taramasi"] = {"denenen_parca": len(probes), "dosya_basina_eslesme": scan}
out = ROOT / "ai" / "experiments" / "results" / f"h2_hq_izolasyon_{a.tag}.json"
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
