"""
E0 — Çapraz şube sorularının doğal dil (yeniden ifade edilmiş) sürümleri.

Boşluk doldurma (cloze) soruları kaynak cümleyi neredeyse aynen içerdiği için sözcük örtüşmesi yüksektir
ve erişimi kolaylaştırır. Bu betik her soru için kaynak cümleyi kopyalamayan, eş anlamlılar kullanan kısa
bir doğal soru üretir (genel merkezdeki Qwen3-8B, düşünme kapalı, sıcaklık 0). Altın sayıyı içeren veya
kaynak cümleyle ardışık 6+ sözcük paylaşan çıktılar reddedilir; bu durumda soru 'natural' kümesine alınmaz.
Sürüm 2: iki örnekli (few-shot) istem; önce Qwen3-8B, reddedilirse Gemma-3-4B denenir; ham çıktılar saklanır.

Çıktı: ai/experiments/results/q1/e0_paraphrases.json
"""
import argparse
import re

import httpx

from q1_common import questions, norm_num, save, tr_tokens


def ngram_overlap(a, b, n=6):
    ta, tb = tr_tokens(a), tr_tokens(b)
    ga = {tuple(ta[i:i + n]) for i in range(len(ta) - n + 1)}
    return any(tuple(tb[i:i + n]) in ga for i in range(len(tb) - n + 1))


FS = ("Görev: Verilen cümledeki ___ boşluğunu soran bir soru yaz. Soru, bir banka çalışanının aklına gelen bilgiyi kendi sözcükleriyle sorması gibi olmalı.\n"
      "Kurallar: (1) Cümlenin sözcük dizisini kopyalama; sözcük sırasını değiştir ve en az üç ifadeyi eş anlamlısıyla değiştir. (2) Kurum/şirket adını koru. "
      "(3) Hiç sayı yazma. (4) Yalnızca soruyu yaz.\n\n"
      "Örnek cümle: \"Şirketin 2021 yılı net kârı bir önceki yıla göre artarak ___ milyon TL olmuştur.\"\n"
      "Örnek soru: Şirket geçen yılın ardından 2021'de kaç milyon liralık net kâr elde etti?\n\n"
      "Örnek cümle: \"Banka, konut kredilerinde aylık faiz oranını ___ seviyesine düşürdüğünü açıkladı.\"\n"
      "Örnek soru: Bankanın açıkladığına göre konut kredisi aylık faizi hangi orana indirildi?\n\n")


def entities(s):
    return {w.strip("()'’.,") for w in s.split() if len(w) >= 3 and w[0].isupper()} - {"Ek", "Bununla", "Ayrıca", "Bu", "Söz", "Geçen", "Şirket", "Şirketin"}


def accept(txt, q):
    if not (10 < len(txt) < 250 and txt.endswith("?")):
        return False
    if re.search(r"\d", txt) or "___" in txt or ngram_overlap(txt, q["reference"]):
        return False
    ents = entities(q["reference"])
    return not ents or any(e in txt for e in ents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    qs, _, _ = questions(a.n)
    gens = [("qwen3-8b", "http://127.0.0.1:8098/v1/chat/completions", " /no_think"), ("gemma-3-4b", "http://127.0.0.1:8099/v1/chat/completions", "")]
    out, rej = [], 0
    with httpx.Client(timeout=120) as cl:
        for i, q in enumerate(qs):
            rec = {"i": i, "reference": q["reference"], "natural": None, "generator": None, "raw": []}
            for name, url, suf in gens:
                msg = FS + "Cümle: \"" + q["search_query"] + "\"\nSoru:" + suf
                r = cl.post(url, json={"messages": [{"role": "user", "content": msg}], "max_tokens": 80, "temperature": 0.0})
                txt = r.json()["choices"][0]["message"]["content"].split("</think>")[-1].strip().strip('"').split("\n")[0].strip()
                rec["raw"].append(txt)
                if accept(txt, q):
                    rec["natural"], rec["generator"] = txt, name
                    break
            rej += int(rec["natural"] is None)
            out.append(rec)
            if (i + 1) % 50 == 0:
                print(i + 1, "reddedilen:", rej, flush=True)
    from collections import Counter
    save("e0_paraphrases.json", {"n": len(qs), "kabul": len(qs) - rej, "red": rej,
                                 "uretec": dict(Counter(r["generator"] for r in out if r["generator"])),
                                 "filtre": "rakam yok, '?' ile biter, kaynak cümleyle 6-gram örtüşme yok, kaynak cümledeki özel adlardan en az biri korunur",
                                 "kayitlar": out})


if __name__ == "__main__":
    main()
