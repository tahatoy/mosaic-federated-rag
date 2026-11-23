"""
E12a — Training queries for a retrieval-oriented TurkEmbed (TurkEmbed v2).

The first TurkEmbed model was fine-tuned on instruction-response pairs, which taught it to match a question with an
answer rather than with the passage that contains the answer, and it generalised worse than its base model. This
script builds query -> source-chunk pairs from the branch corpora instead:

  * natural query : Qwen3-8B (thinking off) writes one question that the chunk answers, in a bank employee's words
  * cloze query   : a sentence of the chunk with a single figure is blanked, as in the H1 question generator

Leakage control: every chunk that contains (part of) the source sentence of any of the 600 cross-branch test
questions is excluded, together with its gold chunks, so the test questions and their paraphrases never share a
passage with the training data.

Output: ai/data/e12/train_queries.jsonl (corpus-derived, not redistributed) and a summary in results/q1.
"""
import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from q1_common import ROOT, norm_num, questions, save, tr_tokens  # also puts hypothesis_tests on sys.path
from h1_federated_rag_eval import NUM  # noqa: E402

OUT = ROOT / "ai" / "data" / "e12"

PROMPT = (
    "Görev: Aşağıdaki finans/bankacılık metin parçasını okuyan bir banka çalışanının, cevabı bu parçada bulunan "
    "bir bilgiyi öğrenmek için yazacağı TEK bir soru yaz.\n"
    "Kurallar: (1) Soru parçadaki belirli bir bilgiyi (oran, tutar, tarih, kural, kurum, kavram) sormalı ve parçayı "
    "görmeyen biri için de anlaşılır olmalı; varsa kurum veya şirket adını içermeli. (2) Parçadan beş sözcükten uzun "
    "bir diziyi aynen kopyalama, kendi sözcüklerinle sor. (3) Cevabı soruya yazma. (4) Yalnızca soruyu yaz.\n\n"
    "Parça:\n\"\"\"\n{chunk}\n\"\"\"\nSoru: /no_think"
)


def ngram_overlap(a, b, n=6):
    ta, tb = tr_tokens(a), tr_tokens(b)
    ga = {tuple(ta[i:i + n]) for i in range(len(ta) - n + 1)}
    return any(tuple(tb[i:i + n]) in ga for i in range(len(tb) - n + 1))


def usable(text):
    letters = sum(ch.isalpha() for ch in text)
    return len(text) >= 200 and letters / max(1, len(text)) > 0.6


def cloze_of(text):
    body = text.split("\n\n", 1)[-1]
    for sent in re.split(r"(?<=[.!?])\s+", body):
        ms = list(NUM.finditer(sent))
        if 70 < len(sent) < 300 and len(ms) == 1 and len(norm_num(ms[0].group(1))) >= 2:
            m = ms[0]
            blank = sent[:m.start()] + "___" + sent[m.end():]
            if not blank.lstrip().startswith(("___", ":", "%")):
                return blank
    return None


def ask(cl, url, chunk):
    r = cl.post(url, json={"messages": [{"role": "user", "content": PROMPT.format(chunk=chunk[:1400])}],
                           "max_tokens": 80, "temperature": 0.3})
    txt = r.json()["choices"][0]["message"]["content"].split("</think>")[-1].strip().strip('"')
    return txt.split("\n")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000, help="number of chunks to sample")
    ap.add_argument("--url", default="http://127.0.0.1:8098/v1/chat/completions")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-minutes", type=float, default=110)
    ap.add_argument("--offset", type=int, default=0, help="skip the first chunks of the fixed shuffled order")
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()

    qs, docs, owner = questions(600)
    test_refs = [q["reference"] for q in qs]
    probes = [r[:60] for r in test_refs] + [r[-60:] for r in test_refs]
    gold = {i for q in qs for i in q["gold_idx"]}
    excluded = {i for i, t in enumerate(docs) if i in gold or any(p in t for p in probes)}
    eligible = [i for i in range(len(docs)) if i not in excluded and usable(docs[i])]
    rnd = random.Random(12)
    rnd.shuffle(eligible)
    pick = eligible[a.offset:a.offset + a.n]
    print(f"chunks: {len(docs)}, excluded for test overlap: {len(excluded)}, eligible: {len(eligible)}, sampled: {len(pick)}",
          flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "train_queries.jsonl"
    t0, stats = time.time(), {"natural_ok": 0, "natural_rejected": 0, "cloze": 0}
    with httpx.Client(timeout=180) as cl, out_path.open("a" if a.append else "w", encoding="utf-8") as f, \
            ThreadPoolExecutor(a.workers) as ex:
        for start in range(0, len(pick), a.workers * 8):
            if (time.time() - t0) / 60 > a.max_minutes:
                print("time limit reached", flush=True)
                break
            batch = pick[start:start + a.workers * 8]
            outs = list(ex.map(lambda i: (i, ask(cl, a.url, docs[i])), batch))
            for i, txt in outs:
                ok = 15 < len(txt) < 250 and txt.endswith("?") and "___" not in txt and not ngram_overlap(txt, docs[i])
                stats["natural_ok" if ok else "natural_rejected"] += 1
                if ok:
                    f.write(json.dumps({"doc": i, "kind": "natural", "query": txt}, ensure_ascii=False) + "\n")
                cz = cloze_of(docs[i])
                if cz:
                    stats["cloze"] += 1
                    f.write(json.dumps({"doc": i, "kind": "cloze", "query": cz}, ensure_ascii=False) + "\n")
            done = start + len(batch)
            if done % 400 < a.workers * 8:
                print(done, stats, f"{(time.time() - t0) / 60:.1f} min", flush=True)
    save(f"e12a_train_queries_summary{'_off' + str(a.offset) if a.offset else ''}.json", {"offset": a.offset, "n_docs": len(docs), "excluded_test_overlap": len(excluded),
                                             "eligible": len(eligible), "sampled": len(pick), **stats,
                                             "minutes": round((time.time() - t0) / 60, 1)})


if __name__ == "__main__":
    main()
