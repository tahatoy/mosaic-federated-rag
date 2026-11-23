"""
H2 deneyi (1/3): PII maskeleme geri çağırma (recall) ve yanlış pozitif testi.

Amaç: backend/ai_engine.py içindeki `_filter_pii` işlevinin sentetik ama GEÇERLİ
(sağlama basamağı doğru) kişisel verileri ne ölçüde maskelediğini ölçmek.
Ağır bağımlılıkları (torch vb.) içe aktarmamak için işlev, ast ile kaynaktan çıkarılır.

Çıktı: ai/experiments/results/h2_pii_leakage.json
Çalıştırma: python backend/scripts/hypothesis_tests/h2_pii_leakage_test.py
"""
import ast, json, random, re, os, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = (ROOT / os.getenv("PII_SRC", "backend/ai_engine.py")).read_text(encoding="utf-8")
random.seed(42)

def load_filter():
    tree = ast.parse(SRC)
    # _filter_pii ve (v2'de) yardımcıları: _pii_* işlevleri ve _PII_* sabitleri
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and (n.name == "_filter_pii" or n.name.startswith("_pii_")))
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "").startswith("_PII_") for t in n.targets))]
    mod = ast.Module(body=keep, type_ignores=[])
    ns = {"re": re, "os": os, "logger": types.SimpleNamespace(info=lambda *a, **k: None)}
    exec(compile(mod, "ai_engine._filter_pii", "exec"), ns)
    return ns["_filter_pii"]

_filter_pii = load_filter()

# ---- geçerli sentetik kimlik üreticileri ---------------------------------
def tc_no():
    d = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(8)]
    d10 = ((sum(d[0::2]) * 7) - sum(d[1::2])) % 10
    d.append(d10)
    d.append(sum(d) % 10)
    return "".join(map(str, d))

def iban_tr():
    bank = "".join(str(random.randint(0, 9)) for _ in range(5))
    acct = "".join(str(random.randint(0, 9)) for _ in range(17))
    bban = bank + "0" + acct  # 5 + 1 + 17 = 23? TR BBAN = 22 hane
    bban = bban[:22]
    tmp = bban + "2927" + "00"  # T=29, R=27
    chk = 98 - (int(tmp) % 97)
    return f"TR{chk:02d}{bban}"

def luhn(n15):
    s = 0
    for i, ch in enumerate(reversed(n15)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9: d -= 9
        s += d
    return str((10 - s % 10) % 10)

def card():
    base = "5" + "".join(str(random.randint(0, 9)) for _ in range(14))
    return base + luhn(base)

FIRST = ["Ahmet","Mehmet","Ayşe","Fatma","Mustafa","Zeynep","Emre","Elif","Hüseyin","Özlem","İbrahim","Şeyma","Çağlar","Ömer","Gül"]
LAST = ["Yılmaz","Kaya","Demir","Çelik","Şahin","Yıldız","Öztürk","Aydın","Arslan","Doğan","Kılıç","Aslan","Çetin","Koç","Güneş"]

def name_variants():
    f, l = random.choice(FIRST), random.choice(LAST)
    return {
        "ad_soyad_baş_harf_büyük": f"{f} {l}",
        "üç_parçalı_ad": f"{f} {random.choice(FIRST)} {l}",
        "hepsi_büyük_harf": f"{f} {l}".upper().replace("İ","İ"),
        "hepsi_küçük_harf": f"{f} {l}".lower(),
        "soyad_ad_sırası_virgüllü": f"{l}, {f}",
        "unvanlı_tek_isim": f"{f} Bey",
    }

def phone_variants():
    a = random.choice(["530","532","541","555","505","542"])
    b, c, d = f"{random.randint(100,999)}", f"{random.randint(0,99):02d}", f"{random.randint(0,99):02d}"
    return {
        "0_5xx_xxx_xx_xx": f"0{a} {b} {c} {d}",
        "bitişik_05xxxxxxxxx": f"0{a}{b}{c}{d}",
        "tireli": f"0{a}-{b}-{c}-{d}",
        "uluslararası_+90": f"+90 {a} {b} {c} {d}",
        "parantezli": f"0 ({a}) {b} {c} {d}",
    }

def card_variants():
    n = card()
    return {
        "bitişik_16": n,
        "4lü_boşluklu": " ".join(n[i:i+4] for i in range(0, 16, 4)),
        "4lü_tireli": "-".join(n[i:i+4] for i in range(0, 16, 4)),
    }

def iban_variants():
    i = iban_tr()
    return {
        "bitişik": i,
        "4lü_boşluklu": " ".join(i[k:k+4] for k in range(0, len(i), 4)),
        "küçük_harf": i.lower(),
    }

_TR_ASCII = {ord(a): b for a, b in zip("\u015f\u0131\u011f\u00fc\u00f6\u00e7\u0130\u015e\u011e\u00dc\u00d6\u00c7", "sigugoc" + "ISGUOC")}

def _ascii(x):
    return x.translate(_TR_ASCII).lower()

def email_variants():
    u = _ascii(random.choice(FIRST))
    l = _ascii(random.choice(LAST))
    return {"standart": f"{u}.{l}{random.randint(1,99)}@example.com", "alt_çizgili": f"{u}_{l}@banka.com.tr"}

def tc_variants():
    t = tc_no()
    return {"bitişik_11": t, "3lü_boşluklu": f"{t[:3]} {t[3:6]} {t[6:9]} {t[9:]}", "noktalı": f"{t[:3]}.{t[3:6]}.{t[6:9]}.{t[9:]}"}

TEMPLATES = [
    "Müşteri bilgisi: {v}. Lütfen işlemi tamamlayın.",
    "{v} numaralı kayıt için limit artışı talep edildi.",
    "Dosyada {v} ifadesi geçmektedir, kontrol ediniz.",
]

def evaluate(category, gen, n=200):
    per_variant = {}
    for _ in range(n):
        for vname, val in gen().items():
            text = random.choice(TEMPLATES).format(v=val)
            out = _filter_pii(text, {})
            leaked = val in out  # değerin birebir sızması
            # kısmi sızıntı: değerin rakam/harf çekirdeği çıktıda kalıyor mu
            core = re.sub(r"[^0-9A-Za-zÇĞİÖŞÜçğıöşü@._]", "", val)
            out_core = re.sub(r"[^0-9A-Za-zÇĞİÖŞÜçğıöşü@._]", "", out)
            partial = core in out_core
            d = per_variant.setdefault(vname, {"n": 0, "masked": 0, "partial_leak": 0})
            d["n"] += 1
            d["masked"] += int(not leaked and not partial)
            d["partial_leak"] += int(partial and not leaked)
    for v in per_variant.values():
        v["recall"] = round(v["masked"] / v["n"], 4)
    tot_n = sum(v["n"] for v in per_variant.values()); tot_m = sum(v["masked"] for v in per_variant.values())
    return {"variants": per_variant, "overall_recall": round(tot_m / tot_n, 4), "n": tot_n}

def false_positive_rate(sample=3000):
    path = ROOT / "ai" / "data" / "splits" / "hq" / "corpus.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    random.shuffle(rows)
    rows = rows[:sample]
    changed = 0; examples = []; placeholders = {}
    for r in rows:
        txt = r.get("text") or r.get("content") or ""
        m = {}
        out = _filter_pii(txt, m)
        if m:
            changed += 1
            for ph, orig in m.items():
                cat = re.match(r"\[([A-Z_]+)_\d+\]", ph).group(1)
                placeholders[cat] = placeholders.get(cat, 0) + 1
                if len(examples) < 25:
                    examples.append({"kategori": cat, "maskelenen": orig})
    return {"örnek_sayısı": len(rows), "en_az_bir_maskelenen_chunk": changed,
            "oran": round(changed / len(rows), 4), "kategori_sayıları": placeholders, "örnekler": examples}

if __name__ == "__main__":
    res = {
        "TC_kimlik_no": evaluate("tc", tc_variants),
        "IBAN": evaluate("iban", iban_variants),
        "telefon": evaluate("tel", phone_variants),
        "kredi_kartı": evaluate("kk", card_variants),
        "e_posta": evaluate("mail", email_variants),
        "ad_soyad": evaluate("ad", name_variants),
        "gerçek_metinde_yanlış_pozitif": false_positive_rate(),
    }
    out = ROOT / "ai" / "experiments" / "results" / "h2_pii_leakage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in res.items():
        if "overall_recall" in v:
            print(f"{k:28s} recall={v['overall_recall']:.3f}  " + ", ".join(f"{a}:{b['recall']:.2f}" for a, b in v["variants"].items()))
        else:
            print(k, {a: b for a, b in v.items() if a != "örnekler"})
    print("->", out)
