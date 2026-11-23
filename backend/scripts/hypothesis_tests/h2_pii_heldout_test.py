"""
H2 – kişisel veri maskeleme: AYRI TUTULMUŞ (held-out) test kümesi.
Bu betik, maskeleme iyileştirmesinden ÖNCE yazılmıştır (2026-10-14) ve ilk testten
farklı tohum, farklı ad/soyad listeleri, farklı cümle kalıpları ve ek biçim varyantları kullanır.
Ayrıca sayısal ama kişisel olmayan finansal ifadelerle yanlış alarm (negatif) testi yapar.

Kullanım: python h2_pii_heldout_test.py [--src backend/ai_engine.py] [--fn _filter_pii] [--tag v1]
Çıktı:   ai/experiments/results/h2_pii_heldout_<tag>.json
"""
import argparse, ast, json, os, random, re, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="backend/ai_engine.py"); ap.add_argument("--fn", default="_filter_pii")
ap.add_argument("--tag", default="v1"); ap.add_argument("--n", type=int, default=300)
ap.add_argument("--set", default="heldout", choices=["heldout", "confirm", "confirm2"])
a = ap.parse_args()

def load_filter(src, name):
    code = (ROOT / src).read_text(encoding="utf-8")
    tree = ast.parse(code)
    keep = [n for n in tree.body if (isinstance(n, (ast.FunctionDef, ast.Assign)) and
            (getattr(n, "name", None) == name or getattr(n, "name", "").startswith("_pii_") or
             any(getattr(t, "id", "").startswith("_PII_") for t in getattr(n, "targets", []))))]
    ns = {"re": re, "os": os, "logger": types.SimpleNamespace(info=lambda *x, **k: None, debug=lambda *x, **k: None)}
    exec(compile(ast.Module(body=keep, type_ignores=[]), src, "exec"), ns)
    return ns[name]

f = load_filter(a.src, a.fn)
rnd = random.Random({"heldout": 2026, "confirm": 777, "confirm2": 31337}[a.set])

FIRST = ["Burak","Selin","Kerem","Derya","Onur","Tuğba","Serkan","Esra","Hakan","Merve","Yusuf","Büşra",
         "Cem","Pınar","Kaan","Gizem","Volkan","Ebru","Tolga","Nur","Barış","Sevgi","Uğur","İrem"]
LAST = ["Aksoy","Karaca","Polat","Ateş","Erdem","Bulut","Tekin","Ünal","Korkmaz","Özdemir","Kurt","Bozkurt",
        "Aydoğan","Sarı","Turan","Işık","Güler","Ekinci","Çakır","Şimşek"]
if a.set == "confirm":
    # doğrulama kümesi: kod dondurulduktan SONRA tanımlandı; adların bir kısmı ad sözlüğünde YOK
    FIRST = ["Doğukan","Ilgaz","Nehir","Taylan","Bora","Ezgi","Tuana","Alp","Ada","Çınar","Mert","Elif","Ayşe","Kemal","Lale","Nazım"]
    LAST = ["Yücel","Kalkan","Başaran","Kocaman","Soylu","Akın","Tunç","Duman","Keskin","Uysal","Karakaya","Öcal"]
if a.set == "confirm2":
    # doğrulama-2: ad iyileştirmesinden ÖNCE tanımlandı (2026-10-14 21:45). Adların çoğu sözlükte YOK; form etiketleri ve serbest metin karışık.
    FIRST = ["Mirza","Şenay","Tamer","Belgin","Kutay","Oya","Rüstem","Zerrin","Arif","Feride","Doğan","Nesrin","Haluk","Pelin","Yaşar","Gülten"]
    LAST = ["Erkoç","Tezcan","Sevinç","Dündar","Atalay","Kılınç","Özbek","Çolak","Yazıcı","Ekşi","Coşkun","Varol"]
CTX_CONFIRM2 = ["Adı Soyadı: {v}\nŞube: Merkez", "Hesap Sahibi: {v} (vadesiz TL)", "Müşteri Adı : {v}", "Sayın {v}, talebiniz alındı.",
                "Ad-Soyad: {v} | Segment: Bireysel", "{v} isimli müşteri şubeye geldi.", "Kefil {v} ile görüşüldü.", "Alıcı: {v}"]
CTX_CONFIRM = ["Hesap sahibi {v} için ekstre hazırlandı.", "Not: {v} — şube ziyaretinde paylaşıldı.",
               "Müşteri formunda {v} yazıyordu; teyit edildi.", "{v}\n(şube 1 kaydı)", "Talep sahibi: {v}; durum: açık."]
CTX = ["Sayın {v}, başvurunuz alınmıştır.", "Gönderen: {v}\nKonu: hesap güncelleme",
       "{v} adlı müşterinin talebi işleme alındı.", "Lütfen {v} bilgisini doğrulayınız.",
       "Kayıt notu ({v}) sistemde güncellendi.", "İletişim için: {v}"]

def tc():
    d = [rnd.randint(1, 9)] + [rnd.randint(0, 9) for _ in range(8)]
    d.append(((sum(d[0::2]) * 7) - sum(d[1::2])) % 10); d.append(sum(d) % 10)
    return "".join(map(str, d))
def iban():
    bban = "".join(str(rnd.randint(0, 9)) for _ in range(5)) + "0" + "".join(str(rnd.randint(0, 9)) for _ in range(16))
    chk = 98 - int(bban + "292700") % 97
    return f"TR{chk:02d}{bban}"
def luhn_card():
    base = rnd.choice(["4", "5"]) + "".join(str(rnd.randint(0, 9)) for _ in range(14))
    s = sum((d * 2 - 9 if d * 2 > 9 else d * 2) if i % 2 == 0 else d for i, d in enumerate(int(c) for c in reversed(base)))
    return base + str((10 - s % 10) % 10)
def ascii_(x): return x.translate(str.maketrans("şığüöçŞİĞÜÖÇ", "siguocSIGUOC")).lower()

def gens():
    t, i, c = tc(), iban(), luhn_card()
    a3, b3, x2, y2 = rnd.choice(["533", "544", "506", "551", "538"]), f"{rnd.randint(100,999)}", f"{rnd.randint(0,99):02d}", f"{rnd.randint(0,99):02d}"
    fn, ln = rnd.choice(FIRST), rnd.choice(LAST)
    return {
        "TC": {"etiketli": f"T.C. Kimlik No: {t}", "tireli": f"{t[:3]}-{t[3:6]}-{t[6:9]}-{t[9:]}",
               "ikili_boşluk": f"{t[:2]} {t[2:5]} {t[5:8]} {t[8:]}", "bitişik": t},
        "IBAN": {"TR_boşluklu_iki": f"TR {i[2:4]} " + " ".join(i[k:k+4] for k in range(4, 26, 4)),
                 "tireli": "-".join(i[k:k+4] for k in range(0, 26, 4)), "küçük_bitişik": i.lower(), "standart": i},
        "TELEFON": {"noktalı": f"0{a3}.{b3}.{x2}.{y2}", "parantez_bitişik": f"0({a3}){b3}{x2}{y2}",
                    "+90_tireli": f"+90-{a3}-{b3}-{x2}{y2}", "boşluksuz_+90": f"+90{a3}{b3}{x2}{y2}", "standart": f"0{a3} {b3} {x2} {y2}"},
        "KART": {"noktalı": ".".join(c[k:k+4] for k in range(0, 16, 4)), "tireli": "-".join(c[k:k+4] for k in range(0, 16, 4)),
                 "boşluklu": " ".join(c[k:k+4] for k in range(0, 16, 4))},
        "EPOSTA": {"büyük_harfli": f"{ascii_(fn).capitalize()}.{ascii_(ln).capitalize()}@Ornek.com.tr",
                   "rakamlı": f"{ascii_(fn)}{rnd.randint(70,99)}@mail.com"},
        "AD": {"ad_soyad": f"{fn} {ln}", "ek_almış": f"{fn} {ln}'ın", "büyük_soyad": f"{fn} {ln.upper()}",
               "soyad_önce": f"{ln.upper()} {fn}"},
    }

NEG = ["Kredi tutarı 1.250.000 TL olarak onaylandı.", "Toplam aktifler 2.4 trilyon TL seviyesine ulaştı.",
       "İşlem tarihi 12.03.2024 olarak kaydedildi.", "Faiz oranı %3,50 ve vade 36 ay.",
       "Vergi kimlik numarası 1234567890 olan şirket.", "Referans no 20240312-0045 ile takip edin.",
       "Merkez Bankası politika faizini 500 baz puan artırdı.", "Borsa İstanbul BIST 100 endeksi 9.875 puandan kapandı.",
       "Hisse başı kâr 12,75 TL; temettü verimi %4,2.", "Sermaye Piyasası Kurulu yeni tebliği yayımladı.",
       "Kamuyu Aydınlatma Platformu üzerinden açıklama yapıldı.", "Türkiye Cumhuriyet Merkez Bankası rezervleri arttı.",
       "Garanti BBVA ve Akbank birinci çeyrek sonuçlarını açıkladı.", "Swift kodu TGBATRIS olan banka.",
       "Müşteri numarası 4471 olan şube kaydı.", "Sipariş 5551234 adet olarak girildi."]

res = {"impl": f"{a.src}:{a.fn}", "set": a.set, "seed": {"heldout": 2026, "confirm": 777, "confirm2": 31337}[a.set], "n_per_variant": a.n, "kategoriler": {}}
for _ in range(a.n):
    for cat, vs in gens().items():
        for vn, val in vs.items():
            text = rnd.choice({"heldout": CTX, "confirm": CTX_CONFIRM, "confirm2": CTX_CONFIRM2}[a.set]).format(v=val)
            out = f(text, {})
            core = re.sub(r"[^0-9A-Za-zÇĞİÖŞÜçğıöşü@]", "", val.split(": ")[-1].replace("'ın", ""))
            leak = core.lower() in re.sub(r"[^0-9A-Za-zÇĞİÖŞÜçğıöşü@]", "", out).lower()
            d = res["kategoriler"].setdefault(cat, {}).setdefault(vn, {"n": 0, "maskelendi": 0})
            d["n"] += 1; d["maskelendi"] += int(not leak)
for cat, vs in res["kategoriler"].items():
    for d in vs.values(): d["recall"] = round(d["maskelendi"] / d["n"], 4)
    tn = sum(d["n"] for d in vs.values()); tm = sum(d["maskelendi"] for d in vs.values())
    vs["_toplam_recall"] = round(tm / tn, 4)
alln = sum(d["n"] for vs in res["kategoriler"].values() for k, d in vs.items() if not k.startswith("_"))
allm = sum(d["maskelendi"] for vs in res["kategoriler"].values() for k, d in vs.items() if not k.startswith("_"))
res["genel_recall"] = round(allm / alln, 4)
fp = []
for s in NEG:
    m = {}; f(s, m)
    if m: fp.append({"cümle": s, "maskelenen": list(m.values())})
res["negatif_cümleler"] = {"n": len(NEG), "yanlış_alarm": len(fp), "oran": round(len(fp) / len(NEG), 4), "örnekler": fp}
# gerçek korpusta yanlış alarm (ilk testle aynı yöntem, farklı örneklem)
rows = [json.loads(l) for l in open(ROOT / "ai/data/splits/hq/corpus.jsonl", encoding="utf-8") if l.strip()]
rnd.shuffle(rows); rows = rows[:3000]; ch = 0; cats = {}
for r in rows:
    m = {}; f(r.get("text", ""), m)
    if m:
        ch += 1
        for ph in m: c = re.match(r"\[([A-Z_]+)_\d+\]", ph).group(1); cats[c] = cats.get(c, 0) + 1
res["gerçek_korpus_yanlış_alarm"] = {"n": len(rows), "maskelenen_chunk": ch, "oran": round(ch / len(rows), 4), "kategori": cats}
out = ROOT / "ai/experiments/results" / f"h2_pii_{a.set}_{a.tag}.json"
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v.get("_toplam_recall") for k, v in res["kategoriler"].items()}, ensure_ascii=False),
      "genel", res["genel_recall"], "negatif_FP", res["negatif_cümleler"]["oran"], "korpus_FP", res["gerçek_korpus_yanlış_alarm"]["oran"])
