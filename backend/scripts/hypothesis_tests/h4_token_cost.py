"""Öneri formu H4'ün token/hesaplama maliyeti bileşeni.
Karşılaştırılanlar (şube korpusundan rastgele seçilen Türkçe parçalar, aynı dbmdz BERT WordPiece sözlüğü):
  genel     : ham metin -> BERT tokenizer (genel amaçlı)
  kok       : Zemberek kök ayrıştırma (analyze_and_disambiguate + get_stem; 4.7.4'teki H4 deneyiyle aynı yöntem) -> BERT
  morfem    : Zemberek morfem bölütleme (kök + ekler ayrı; öneri formundaki hibrit tasarım) -> BERT
  turkembed : gömme modelinin kendi tokenizer'ı (ham metin)
Ayrıca ai/tokenizer/hybrid_tokenizer.py'nin ZemberekAnalyzer yolu kontrol edilir (analiz gerçekten uygulanıyor mu?).
Çıktı: ai/experiments/results/h4_token_maliyeti.json"""
import json, random, sys, time, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "ai"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer
from zemberek import TurkishMorphology
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rows = []
for b in ("branch1", "branch2"):
    with open(ROOT / "ai/data/splits" / b / "corpus.jsonl", encoding="utf-8") as f:
        rows += [json.loads(l)["text"] for l in f if l.strip()]
tr = [t for t in rows if sum(c in "çğışöüÇĞİŞÖÜ" for c in t) >= 5]
random.Random(2026).shuffle(tr); texts = [t[:1000] for t in tr[:N]]
words = sum(len(t.split()) for t in texts)
bert = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
morph = TurkishMorphology.create_with_defaults()

def analyses(text):
    try:
        return morph.analyze_and_disambiguate(text).best_analysis()
    except Exception:
        return []
def kok(text):
    out = []
    for a in analyses(text):
        try:
            s = a.get_stem(); w = a.item.lemma if hasattr(a, "item") else None
            out.append(s if s and s != "UNK" and len(s) > 1 else (a.surface_form() if hasattr(a, "surface_form") else s))
        except Exception:
            pass
    return " ".join(x for x in out if x)
def morfem(text):
    out = []
    for a in analyses(text):
        try:
            parts = [m.surface for m in a.morpheme_data_list if getattr(m, "surface", "")]
            out.extend(parts if parts else [a.surface_form()])
        except Exception:
            pass
    return " ".join(x for x in out if x)
sample = texts[0][:120]
diag = {"ornek": sample, "kok": kok(sample), "morfem": morfem(sample)}
try:
    from tokenizer.zemberek_wrapper import ZemberekAnalyzer
    za = ZemberekAnalyzer(use_cache=False); ws = sample.split()[:8]
    diag["hybrid_tokenizer_ZemberekAnalyzer_degisiklik"] = sum(za.analyze(w) != [w] for w in ws)
    diag["hybrid_tokenizer_ornek"] = [za.analyze(w) for w in ws]
except Exception as e:
    diag["hybrid_tokenizer_hata"] = repr(e)
print(json.dumps(diag, ensure_ascii=False), flush=True)
out = {"n_parca": len(texts), "kelime": words, "tani": diag, "tokenizerlar": {}}
def run(name, pre, tk):
    t0 = time.perf_counter(); ids = [tk.encode(pre(t), add_special_tokens=False) for t in texts]; dt = time.perf_counter() - t0
    tot = sum(len(x) for x in ids); unk = sum(x.count(tk.unk_token_id) for x in ids)
    out["tokenizerlar"][name] = {"toplam_token": tot, "kelime_basina_token": round(tot / words, 4),
                                 "parca_basina_token": round(tot / len(texts), 2), "parca_basina_ms": round(dt / len(texts) * 1000, 3),
                                 "unk_orani": round(unk / tot, 5) if tot else None}
    print(name, out["tokenizerlar"][name], flush=True)
run("genel_bert", lambda t: t, bert)
run("zemberek_kok+bert", kok, bert)
run("zemberek_morfem+bert", morfem, bert)
emb = ROOT / "ai/models/embeddings/turkembed_finetuned"
if emb.exists(): run("turkembed", lambda t: t, AutoTokenizer.from_pretrained(str(emb)))
g = out["tokenizerlar"]["genel_bert"]
out["genel_bert_e_gore"] = {k: {"token_degisim_orani": round(v["toplam_token"] / g["toplam_token"] - 1, 4),
                               "sure_kat": round(v["parca_basina_ms"] / g["parca_basina_ms"], 2)} for k, v in out["tokenizerlar"].items() if k != "genel_bert"}
(ROOT / "ai/experiments/results/h4_token_maliyeti.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out["genel_bert_e_gore"], ensure_ascii=False))
