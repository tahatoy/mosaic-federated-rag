"""
E12b — Fine-tune multilingual-e5-small into TurkEmbed v2 with query -> source-chunk pairs.

Differences from the first TurkEmbed fine-tuning:
  * positives are the chunks that answer the query (E12a), not generated answers
  * the e5 prefixes "query: " / "passage: " are used in training exactly as at inference
  * one mined hard negative per query (rank 3-30 under the base model, near-duplicates of the positive removed,
    because neighbouring chunks overlap by 120 characters)
  * the checkpoint is selected on a development split of held-out chunks; the cross-branch test questions are
    never used for training or model selection

Output: ai/models/embeddings/turkembed_v2 and results/q1/e12b_train.json
"""
import argparse
import json
import random

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

try:  # keeps the same chunk from appearing twice in a batch (in-batch false negatives)
    from sentence_transformers.datasets import NoDuplicatesDataLoader
except ImportError:  # pragma: no cover
    NoDuplicatesDataLoader = None

from q1_common import ROOT, load_corpora, mean_ci, rank_metrics, save

BASE = "intfloat/multilingual-e5-small"
OUT_MODEL = ROOT / "ai" / "models" / "embeddings" / "turkembed_v2"
QFILE = ROOT / "ai" / "data" / "e12" / "train_queries.jsonl"


def encode(model, texts, prefix, bs=128):
    return model.encode([prefix + t for t in texts], batch_size=bs, normalize_embeddings=True,
                        convert_to_tensor=True, show_progress_bar=False)


def dev_mrr(model, docs, dev):
    E = encode(model, docs, "passage: ")
    Q = encode(model, [d["query"] for d in dev], "query: ")
    top = torch.topk((Q @ E.T).float(), 10, dim=1).indices.cpu().tolist()
    per = {"natural": [], "cloze": []}
    for r, d in zip(top, dev):
        per[d["kind"]].append(rank_metrics(r, [d["doc"]])["mrr@10"])
    return {k: mean_ci(v) for k, v in per.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--out", default="turkembed_v2")
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    docs, _ = load_corpora()
    rows = [json.loads(line) for line in QFILE.open(encoding="utf-8") if line.strip()]
    chunks = sorted({r["doc"] for r in rows})
    rnd = random.Random(a.seed)
    rnd.shuffle(chunks)
    dev_chunks = set(chunks[: max(200, len(chunks) // 10)])
    train = [r for r in rows if r["doc"] not in dev_chunks]
    dev = [r for r in rows if r["doc"] in dev_chunks]
    print(f"queries: {len(rows)} (train {len(train)}, dev {len(dev)}), chunks: {len(chunks)}", flush=True)

    dev_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(BASE, device=dev_device)
    model.max_seq_length = 256

    # hard negatives from the base model
    E = encode(model, docs, "passage: ")
    Q = encode(model, [r["query"] for r in train], "query: ")
    top = torch.topk((Q @ E.T).float(), 30, dim=1).indices.cpu().tolist()
    examples, no_neg = [], 0
    for r, cand in zip(train, top):
        pos = r["doc"]
        options = [c for c in cand[2:] if c != pos and float(E[c] @ E[pos]) < 0.92]
        if options:
            neg = rnd.choice(options[:10])
            examples.append(InputExample(texts=["query: " + r["query"], "passage: " + docs[pos], "passage: " + docs[neg]]))
        else:
            no_neg += 1
            examples.append(InputExample(texts=["query: " + r["query"], "passage: " + docs[pos], "passage: " + docs[cand[-1]]]))
    del E, Q
    torch.cuda.empty_cache()

    history = {"base": dev_mrr(model, docs, dev)}
    print("base dev:", history["base"], flush=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    best, best_ep = None, 0
    steps = len(examples) // a.batch
    for ep in range(1, a.epochs + 1):
        if NoDuplicatesDataLoader is not None:
            loader = NoDuplicatesDataLoader(examples, batch_size=a.batch)
        else:
            loader = DataLoader(examples, shuffle=True, batch_size=a.batch, drop_last=True)
        model.fit(train_objectives=[(loader, loss)], epochs=1, warmup_steps=int(0.1 * steps) if ep == 1 else 0,
                  optimizer_params={"lr": a.lr}, show_progress_bar=False, use_amp=True)
        history[f"epoch{ep}"] = dev_mrr(model, docs, dev)
        score = sum(v[0] for v in history[f"epoch{ep}"].values())
        print(f"epoch {ep} dev:", history[f"epoch{ep}"], flush=True)
        if best is None or score > best:
            best, best_ep = score, ep
            model.save(str(OUT_MODEL.parent / a.out))
    save(f"e12b_train_{a.out}.json", {"output": a.out, "base_model": BASE, "n_queries": len(rows), "n_train": len(train), "n_dev": len(dev),
                             "dev_chunks": len(dev_chunks), "no_hard_negative": no_neg, "epochs": a.epochs,
                             "lr": a.lr, "batch": a.batch, "max_seq_length": 256, "selected_epoch": best_ep,
                             "dev_mrr10": history})


if __name__ == "__main__":
    main()
