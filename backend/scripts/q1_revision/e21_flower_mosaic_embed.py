"""
E21 — MOSAIC-Embed'in Flower üzerinden gerçek federe eğitimi (taban modeli geçecek tarifle) ve çapraz şube değerlendirmesi.

E7'deki Flower koşusu (LoRA r=8, 600 çift, toplu 16, sert negatif yok, 3 tur) taban modelin gerisinde kalmıştı.
Bu deney MOSAIC-Embed'in merkezi tarifini (E12b / turkembed_v3) federe olarak uygular:
  * veri: ai/data/e12/train_queries.jsonl (MOSAIC-Embed ile aynı sorgu–kaynak parça çiftleri), aynı geliştirme bölmesi
    (tohum 12, parçaların %10'u); eğitim çiftleri parçanın sahibi olan şubeye gider, iki şubede ortak parçalar sıra
    numarasının teklik/çiftliğine göre bölünür (E3 ile aynı kural);
  * her şube sert negatifini YALNIZCA kendi korpusundan (kendi parçaları + ortak parçalar) taban modelle çıkarır
    (sıra 3–30, pozitife kosinüs < 0.92) — başka şubenin metni hiçbir aşamada istemciye girmez;
  * yerel eğitim: MultipleNegativesRankingLoss ile aynı kayıp (ölçek 20, toplu içi negatifler + bir sert negatif),
    toplu 64, aynı parçanın toplu içinde iki kez yer almaması, öğrenme oranı 2e-5, ilk turda %10 ısınma, her turda
    doğrusal azalma (sentence-transformers fit() ile aynı çizelge), en fazla 256 belirteç;
  * paylaşılan güncelleme: sözcük gömme tablosu (250k × 384) dondurulur, kalan ~21 M parametre (≈ 85 MB, fp32)
    Flower (gRPC + TLS) ile taşınır, FedAvg (örnek sayısıyla ağırlıklı) ya da FedProx (μ = 0.01);
  * tur seçimi federe değerlendirmeyle: her istemci kendi geliştirme sorgularında, kendi korpusu üzerinde MRR@10 hesaplar,
    sunucu bunları örnek sayısıyla ağırlıklı ortalar; en iyi turun modeli kaydedilir. Çapraz şube test kümeleri (600 kalıp,
    419 yeniden ifade) her turda yalnızca raporlama için HQ'da ölçülür, seçimde kullanılmaz.

Alt komutlar
  prep     şube dosyalarını hazırlar (ai/data/e21/branch{1,2}.jsonl + dev; sert negatifler şube içinde)
  server   Flower sunucusu (HQ)
  client   Flower istemcisi (HQ'da ayrı süreç ya da şube VM'i)
  central  denetim: aynı döngü ve donuk sözcük gömmesiyle merkezi eğitim (federasyonun maliyetini ayırmak için)
  eval     son değerlendirme: taban e5-small, MOSAIC-Embed, merkezi-donuk denetim, Flower modelleri; tek aşama ve
           çapraz kodlayıcıyla (ilk 20); eşleştirilmiş bootstrap GA. Çıktı: results/q1/e21_flower_mosaic_embed.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE = "intfloat/multilingual-e5-small"
SCALE = 20.0
MAXLEN = 256

try:  # HQ'da proje kökü; VM'de yalnızca istemci kipinde gerekmez
    from q1_common import OUT_DIR, ROOT, load_corpora, mean_ci, paired_diff_ci, questions, rank_metrics, save
except Exception:  # pragma: no cover - şube VM'i
    OUT_DIR = ROOT = None

DATA_DIR = (ROOT / "ai" / "data" / "e21") if ROOT else Path("q1fl21")
MODEL_DIR = (ROOT / "ai" / "models" / "embeddings") if ROOT else Path("q1fl21")


def dev_of():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ───────────────────────── model yardımcıları ─────────────────────────
def load_st(path=BASE, device=None):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(path, device=device or dev_of())
    m.max_seq_length = MAXLEN
    return m


def freeze_word_embeddings(m):
    """Sözcük gömme tablosu dondurulur; dönen liste, Flower'da taşınan tensörlerin sabit sırasıdır."""
    for n, p in m.named_parameters():
        p.requires_grad = "word_embeddings" not in n
    return [(n, p) for n, p in m.named_parameters() if p.requires_grad]


def get_arrays(params):
    return [p.detach().float().cpu().numpy() for _, p in params]


def set_arrays(params, arrs):
    with torch.no_grad():
        for (_, p), a in zip(params, arrs):
            p.copy_(torch.from_numpy(np.asarray(a)).to(p.device, dtype=p.dtype))


def embed(m, texts):
    f = m.tokenize(texts)
    f = {k: (v.to(m.device) if hasattr(v, "to") else v) for k, v in f.items()}  # eski sentence-transformers sözlüğe metin alanı da koyar
    return F.normalize(m(f)["sentence_embedding"], dim=-1)


def encode(m, texts, prefix, bs=256):
    return m.encode([prefix + t for t in texts], batch_size=bs, normalize_embeddings=True,
                    convert_to_tensor=True, show_progress_bar=False)


def batches_no_dup(rows, bs, rnd):
    """NoDuplicatesDataLoader eşdeğeri: aynı pozitif/negatif parça bir toplu içinde iki kez yer almaz."""
    pool = rows[:]
    rnd.shuffle(pool)
    out = []
    while pool:
        b, used, rest = [], set(), []
        for r in pool:
            keys = {r["pos"], r["neg"]}
            if len(b) < bs and not (keys & used):
                b.append(r); used |= keys
            else:
                rest.append(r)
        if len(b) < 2:
            break
        out.append(b)
        pool = rest
    return out


def local_train(m, params, rows, texts, *, lr, bs, rnd, warmup_frac=0.0, mu=0.0, anchor=None, log=None):
    """Bir yerel dönem. rows: {"q", "pos", "neg"}; texts: parça kimliği -> metin (yalnızca bu istemcinin korpusu)."""
    m.train()
    opt = torch.optim.AdamW([p for _, p in params], lr=lr, weight_decay=0.01)
    bl = batches_no_dup(rows, bs, rnd)
    total = len(bl); warm = int(warmup_frac * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    use_amp = m.device.type == "cuda"
    loss_sum = 0.0
    for b in bl:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            q = embed(m, ["query: " + r["q"] for r in b])
            d = embed(m, ["passage: " + texts[r["pos"]] for r in b] + ["passage: " + texts[r["neg"]] for r in b])
            loss = F.cross_entropy(SCALE * q.float() @ d.float().T, torch.arange(len(b), device=m.device))
        if mu > 0 and anchor is not None:
            loss = loss + mu / 2 * sum(((p - a) ** 2).sum() for (_, p), a in zip(params, anchor))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0)
        opt.step(); sched.step()
        loss_sum += float(loss.detach())
    m.eval()
    if log is not None:
        log.update(steps=total, loss=round(loss_sum / max(1, total), 4))
    return total


def dev_mrr(m, texts, dev_rows, pool=0):
    """İstemcinin kendi geliştirme sorguları, kendi korpusu üzerinde MRR@10 (federe değerlendirme).
    pool > 0: CPU'lu şube VM'inde süreyi sınırlamak için korpus yerine geliştirme pozitifleri + sabit tohumlu `pool` parça."""
    if not dev_rows:
        return 0.0
    ids = sorted(texts)
    if pool and pool < len(ids):
        keep = {r["pos"] for r in dev_rows}
        rest = [i for i in ids if i not in keep]
        ids = sorted(keep | set(random.Random(7).sample(rest, max(0, pool - len(keep)))))
    pos_of = {d: i for i, d in enumerate(ids)}
    with torch.no_grad():
        E = encode(m, [texts[i] for i in ids], "passage: ")
        Q = encode(m, [r["q"] for r in dev_rows], "query: ")
        top = torch.topk((Q @ E.T).float(), 10, dim=1).indices.cpu().tolist()
    s = 0.0
    for t, r in zip(top, dev_rows):
        g = pos_of[r["pos"]]
        s += next((1.0 / k for k, x in enumerate(t, 1) if x == g), 0.0)
    return s / len(dev_rows)


def read_branch_file(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    texts = {int(k): v for k, v in obj["texts"].items()}
    return obj["train"], obj["dev"], texts, obj.get("meta", {})


# ───────────────────────── prep ─────────────────────────
def cmd_prep(a):
    random.seed(a.seed); torch.manual_seed(a.seed)
    docs, owner = load_corpora()
    rows = [json.loads(l) for l in (ROOT / "ai/data/e12/train_queries.jsonl").open(encoding="utf-8") if l.strip()]
    chunks = sorted({r["doc"] for r in rows})
    rnd = random.Random(a.seed)  # E12b ile aynı geliştirme bölmesi
    rnd.shuffle(chunks)
    dev_chunks = set(chunks[: max(200, len(chunks) // 10)])

    def branch_of(d):
        o = owner[d]
        return o if o in (1, 2) else (1 if d % 2 == 0 else 2)

    m = load_st(BASE)
    meta = {"seed": a.seed, "n_queries": len(rows), "dev_chunks": len(dev_chunks), "branches": {}}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for b in (1, 2):
        corpus = [i for i in range(len(docs)) if owner[i] in (0, b)]  # şubenin gördüğü korpus: kendi + ortak parçalar
        cset = set(corpus)
        mine = [r for r in rows if branch_of(int(r["doc"])) == b and int(r["doc"]) in cset]
        tr = [r for r in mine if r["doc"] not in dev_chunks]
        dv = [r for r in mine if r["doc"] in dev_chunks]
        E = encode(m, [docs[i] for i in corpus], "passage: ", bs=128)
        Q = encode(m, [r["query"] for r in tr], "query: ", bs=128)
        pos_of = {d: k for k, d in enumerate(corpus)}
        top = torch.topk((Q @ E.T).float(), 30, dim=1).indices.cpu().tolist()
        train, no_neg = [], 0
        rb = random.Random(a.seed * 10 + b)
        for r, cand in zip(tr, top):
            p = pos_of[int(r["doc"])]
            opts = [c for c in cand[2:] if c != p and float(E[c] @ E[p]) < 0.92]
            if not opts:
                no_neg += 1
            neg = rb.choice(opts[:10]) if opts else cand[-1]
            train.append({"q": r["query"], "pos": int(r["doc"]), "neg": corpus[neg], "kind": r.get("kind")})
        dev = [{"q": r["query"], "pos": int(r["doc"]), "kind": r.get("kind")} for r in dv]
        need = {x for t in train for x in (t["pos"], t["neg"])} | {d["pos"] for d in dev}
        # geliştirme değerlendirmesi şubenin tüm korpusu üzerinde yapılır; dosyaya şube korpusunun tamamı yazılır
        texts = {i: docs[i] for i in corpus}
        assert need <= set(texts)
        obj = {"branch": b, "train": train, "dev": dev, "texts": {str(k): v for k, v in texts.items()},
               "meta": {"corpus_chunks": len(corpus), "train_pairs": len(train), "dev_queries": len(dev), "no_hard_negative": no_neg}}
        (DATA_DIR / f"branch{b}.json").write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        meta["branches"][b] = obj["meta"]
        print(f"şube {b}:", obj["meta"], flush=True)
        del E, Q
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


# ───────────────────────── Flower istemcisi ─────────────────────────
def cmd_client(a):
    import flwr as fl
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    train, dev, texts, meta = read_branch_file(a.data)
    if a.max_pairs:
        train = train[: a.max_pairs]
    device = a.device or dev_of()
    m = load_st(a.model, device=device)
    params = freeze_word_embeddings(m)
    if a.grad_ckpt:  # CPU'lu şube VM'inde toplu 64 × 3 dizi × 256 belirteçlik etkinleşme belleğini sınırlar; sonucu değiştirmez
        m[0].auto_model.gradient_checkpointing_enable()
    n_tr = int(sum(p.numel() for _, p in params))
    print(json.dumps({"branch": a.branch, "device": device, "train_pairs": len(train), "dev_queries": len(dev),
                      "corpus_chunks": len(texts), "trainable_params": n_tr, "bytes_per_update": n_tr * 4, "tls": bool(a.ca)}), flush=True)

    class Client(fl.client.NumPyClient):
        def get_parameters(self, config):
            return get_arrays(params)

        def fit(self, parameters, config):
            t0 = time.time()
            set_arrays(params, parameters)
            rnd_i = int(config.get("round", 1))
            mu = float(config.get("proximal_mu", 0.0))
            anchor = [p.detach().clone() for _, p in params] if mu > 0 else None
            log = {}
            local_train(m, params, train, texts, lr=float(config.get("lr", 2e-5)), bs=int(config.get("batch", 64)),
                        rnd=random.Random(1000 * a.branch + rnd_i), warmup_frac=0.1 if rnd_i == 1 else 0.0,
                        mu=mu, anchor=anchor, log=log)
            out = get_arrays(params)
            info = {"round": rnd_i, "branch": a.branch, "fit_s": round(time.time() - t0, 1), **log,
                    "bytes": int(sum(x.nbytes for x in out))}
            print(json.dumps(info), flush=True)
            return out, len(train), {k: float(v) if isinstance(v, (int, float)) else 0.0 for k, v in info.items()}

        def evaluate(self, parameters, config):
            t0 = time.time()
            set_arrays(params, parameters)
            s = dev_mrr(m, texts, dev, pool=a.dev_pool)
            print(json.dumps({"eval_round": config.get("round"), "branch": a.branch, "dev_mrr10": round(s, 4),
                              "eval_s": round(time.time() - t0, 1)}), flush=True)
            return 1.0 - s, len(dev), {"dev_mrr10": float(s), "branch": float(a.branch)}

    kw = {"root_certificates": Path(a.ca).read_bytes()} if a.ca else {"insecure": True}
    fl.client.start_client(server_address=a.server, client=Client().to_client(),
                           grpc_max_message_length=1024 * 1024 * 1024, **kw)


def cmd_selftest(a):
    """Şube VM'inde kütüphane uyumluluğu: 16 çiftle iki yerel adım (gradyan denetim noktasıyla) ve geliştirme değerlendirmesi."""
    import flwr, sentence_transformers, transformers
    torch.set_num_threads(a.threads)
    print(json.dumps({"torch": torch.__version__, "flwr": flwr.__version__, "st": sentence_transformers.__version__, "tf": transformers.__version__}), flush=True)
    train, dev, texts, _ = read_branch_file(a.data)
    m = load_st(a.model, device="cpu"); params = freeze_word_embeddings(m)
    m[0].auto_model.gradient_checkpointing_enable()
    t0 = time.time(); lg = {}
    local_train(m, params, train[:16], texts, lr=2e-5, bs=8, rnd=random.Random(0), log=lg)
    print(json.dumps({"selftest_train": lg, "sec": round(time.time() - t0, 1),
                      "dev_mrr10_pool300": round(dev_mrr(m, texts, dev[:20], pool=300), 4)}), flush=True)


# ───────────────────────── Flower sunucusu ─────────────────────────
def test_sets():
    qs, docs, owner = questions(600)
    para = {r["reference"]: r["natural"] for r in json.loads((OUT_DIR / "e0_paraphrases.json").read_text(encoding="utf-8"))["kayitlar"] if r["natural"]}
    nq = [q for q in qs if q["reference"] in para]
    return docs, owner, {"cloze": (qs, [q["search_query"] for q in qs]), "natural": (nq, [para[q["reference"]] for q in nq])}


def retrieval_eval(m, docs, sets, per_query=False):
    res, perq = {}, {}
    with torch.no_grad():
        E = encode(m, docs, "passage: ")
        for s, (sq, qt) in sets.items():
            Q = encode(m, qt, "query: ")
            top = torch.topk((Q @ E.T).float(), 100, dim=1).indices.cpu().tolist()
            per = [rank_metrics(t, q["gold_idx"]) for t, q in zip(top, sq)]
            res[f"{s}_mrr@10"] = round(float(np.mean([p["mrr@10"] for p in per])), 4)
            res[f"{s}_hit@10"] = round(float(np.mean([p["hit@10"] for p in per])), 4)
            perq[s] = {"per": per, "top": top}
    return (res, perq) if per_query else res


def cmd_server(a):
    import flwr as fl
    device = dev_of()
    m = load_st(a.model, device=device)
    params = freeze_word_embeddings(m)
    docs, _, sets = test_sets()
    log = {"strategy": a.strategy, "rounds": a.rounds, "tls": bool(a.tls_dir), "where": a.where,
           "trainable_params": int(sum(p.numel() for _, p in params)), "eval": [], "fed_dev": [], "fit": [], "t_round_end": []}
    log["bytes_per_update"] = log["trainable_params"] * 4
    t_start = time.time()
    best = {"score": -1.0, "round": None}
    out_model = MODEL_DIR / a.out_model
    tag = a.tag or f"{a.where}_{a.strategy}"
    cur = {"arrays": None}

    def central_eval(server_round, parameters, config):
        set_arrays(params, fl.common.parameters_to_ndarrays(parameters) if hasattr(parameters, "tensors") else parameters)
        cur["arrays"] = get_arrays(params)
        r = {"round": server_round, "t": round(time.time() - t_start, 1), **retrieval_eval(m, docs, sets)}
        log["eval"].append(r)
        print("TEST (yalnızca rapor)", json.dumps(r), flush=True)
        save(f"e21_flower_{tag}.json", log)
        return 0.0, r

    def eval_agg(metrics):
        n = sum(k for k, _ in metrics)
        s = sum(k * mm["dev_mrr10"] for k, mm in metrics) / max(1, n)
        rnd_i = len(log["fed_dev"]) + 1
        log["fed_dev"].append({"round": rnd_i, "dev_mrr10": round(s, 4),
                               "per_client": {int(mm["branch"]): round(mm["dev_mrr10"], 4) for _, mm in metrics}})
        print("FEDERE GELİŞTİRME", json.dumps(log["fed_dev"][-1]), flush=True)
        if s > best["score"] and cur["arrays"] is not None:
            best.update(score=s, round=rnd_i)
            set_arrays(params, cur["arrays"])
            m.save(str(out_model))
            log["selected_round"] = rnd_i; log["selected_dev_mrr10"] = round(s, 4)
            print(f"en iyi tur {rnd_i} kaydedildi -> {out_model}", flush=True)
        save(f"e21_flower_{tag}.json", log)
        return {"dev_mrr10": s}

    def fit_cfg(rnd):
        return {"round": rnd, "lr": a.lr, "batch": a.batch, "proximal_mu": a.mu if a.strategy == "fedprox" else 0.0}

    def fit_agg(metrics):
        log["fit"].append([{"n": n, **mm} for n, mm in metrics])
        log["t_round_end"].append(round(time.time() - t_start, 1))
        return {"fit_s_max": max(mm["fit_s"] for _, mm in metrics)}

    init = fl.common.ndarrays_to_parameters(get_arrays(params))
    kw = dict(fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=a.clients, min_available_clients=a.clients,
              min_evaluate_clients=a.clients, initial_parameters=init, evaluate_fn=central_eval,
              on_fit_config_fn=fit_cfg, on_evaluate_config_fn=lambda r: {"round": r},
              fit_metrics_aggregation_fn=fit_agg, evaluate_metrics_aggregation_fn=eval_agg)
    strat = fl.server.strategy.FedProx(proximal_mu=a.mu, **kw) if a.strategy == "fedprox" else fl.server.strategy.FedAvg(**kw)
    cert = None
    if a.tls_dir:
        t = Path(a.tls_dir)
        cert = ((t / "ca.crt").read_bytes(), (t / "fl_server.crt").read_bytes(), (t / "fl_server.key").read_bytes())
    fl.server.start_server(server_address=f"0.0.0.0:{a.port}", config=fl.server.ServerConfig(num_rounds=a.rounds, round_timeout=a.round_timeout),
                           strategy=strat, certificates=cert, grpc_max_message_length=1024 * 1024 * 1024)
    log["total_s"] = round(time.time() - t_start, 1)
    log["out_model"] = str(out_model)
    save(f"e21_flower_{tag}.json", log)


# ───────────────────────── merkezi denetim ─────────────────────────
def cmd_central(a):
    torch.manual_seed(a.seed)
    tr_all, dv_all, texts_all = [], [], {}
    for b in (1, 2):
        tr, dv, tx, _ = read_branch_file(DATA_DIR / f"branch{b}.json")
        tr_all += tr; dv_all += dv; texts_all.update(tx)
    m = load_st(BASE)
    params = freeze_word_embeddings(m) if a.freeze else [(n, p) for n, p in m.named_parameters()]
    hist, best = [], -1.0
    out_model = MODEL_DIR / a.out_model
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); lg = {}
        local_train(m, params, tr_all, texts_all, lr=a.lr, bs=a.batch, rnd=random.Random(a.seed * 100 + ep),
                    warmup_frac=0.1 if ep == 1 else 0.0, log=lg)
        s = dev_mrr(m, texts_all, dv_all)
        hist.append({"epoch": ep, "dev_mrr10": round(s, 4), "sec": round(time.time() - t0, 1), **lg})
        print(json.dumps(hist[-1]), flush=True)
        if s > best:
            best = s; m.save(str(out_model))
    save(f"e21_central_{a.out_model}.json", {"freeze_word_embeddings": a.freeze, "epochs": a.epochs, "lr": a.lr, "batch": a.batch,
                                             "history": hist, "selected_epoch": max(hist, key=lambda h: h["dev_mrr10"])["epoch"]})


# ───────────────────────── son değerlendirme ─────────────────────────
def cmd_eval(a):
    from sentence_transformers import CrossEncoder
    docs, owner, sets = test_sets()
    cands = [("base_e5_small", BASE), ("mosaic_embed_central", str(MODEL_DIR / "turkembed_v3"))]
    for nm in a.models.split(","):
        nm = nm.strip()
        if nm and (MODEL_DIR / nm).exists():
            cands.append((nm, str(MODEL_DIR / nm)))
        elif nm:
            print("atlandı (yok):", nm, flush=True)
    ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device=dev_of())
    out = {"n": {s: len(v[0]) for s, v in sets.items()}, "models": {}, "diffs": {}}
    per = {}
    for nm, path in cands:
        m = load_st(path)
        m.max_seq_length = 512  # Tablo 4 (E1) ile aynı değerlendirme uzunluğu
        res, pq = retrieval_eval(m, docs, sets, per_query=True)
        row = {}
        for s, (sq, qt) in sets.items():
            mrr = [p["mrr@10"] for p in pq[s]["per"]]
            row[s] = {"mrr@10": mean_ci(mrr), "hit@10": mean_ci([p["hit@10"] for p in pq[s]["per"]])}
            # deploy edilen hatla aynı: ilk 20 yoğun aday + çapraz kodlayıcı
            rr = []
            for q, text, top in zip(sq, qt, pq[s]["top"]):
                c = top[:20]
                sc = ce.predict([[text, docs[i]] for i in c], show_progress_bar=False)
                rr.append(rank_metrics([c[j] for j in np.argsort(-np.asarray(sc))], q["gold_idx"])["mrr@10"])
            row[s]["mrr@10_rerank"] = mean_ci(rr)
            per[(nm, s)] = mrr; per[(nm, s, "rr")] = rr
        out["models"][nm] = row
        print(nm, json.dumps(row), flush=True)
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    names = [n for n, _ in cands]
    for nm in names[2:]:
        for ref in ("base_e5_small", "mosaic_embed_central"):
            for s in sets:
                out["diffs"][f"{nm}_vs_{ref}_{s}"] = paired_diff_ci(per[(nm, s)], per[(ref, s)])
                out["diffs"][f"{nm}_vs_{ref}_{s}_rerank"] = paired_diff_ci(per[(nm, s, "rr")], per[(ref, s, "rr")])
    for f in sorted(OUT_DIR.glob("e21_flower_*.json")):
        try:
            lg = json.loads(f.read_text(encoding="utf-8"))
            out.setdefault("flower_runs", {})[f.stem] = {k: lg.get(k) for k in ("where", "strategy", "rounds", "tls", "trainable_params",
                                                                                    "bytes_per_update", "selected_round", "selected_dev_mrr10",
                                                                                    "total_s", "t_round_end")}
            out["flower_runs"][f.stem]["fit_s"] = [[round(c.get("fit_s", 0), 1) for c in r] for r in lg.get("fit", [])]
            out["flower_runs"][f.stem]["test_by_round"] = lg.get("eval")
            out["flower_runs"][f.stem]["fed_dev_by_round"] = lg.get("fed_dev")
        except Exception as e:  # pragma: no cover
            print("günlük okunamadı", f, e)
    save("e21_flower_mosaic_embed.json", out)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep"); p.add_argument("--seed", type=int, default=12)
    p = sp.add_parser("selftest"); p.add_argument("--data", required=True); p.add_argument("--model", default=BASE); p.add_argument("--threads", type=int, default=2)
    p = sp.add_parser("client")
    p.add_argument("--server", required=True); p.add_argument("--branch", type=int, required=True)
    p.add_argument("--data", required=True); p.add_argument("--model", default=BASE); p.add_argument("--ca", default="")
    p.add_argument("--device", default=""); p.add_argument("--threads", type=int, default=0); p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--dev-pool", type=int, default=0); p.add_argument("--grad-ckpt", type=int, default=0)
    p = sp.add_parser("server")
    p.add_argument("--port", type=int, default=9092); p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--strategy", default="fedavg", choices=["fedavg", "fedprox"]); p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--model", default=BASE); p.add_argument("--tls-dir", default=""); p.add_argument("--clients", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5); p.add_argument("--batch", type=int, default=64)
    p.add_argument("--round-timeout", type=float, default=7200.0); p.add_argument("--where", default="hq"); p.add_argument("--tag", default=""); p.add_argument("--out-model", default="mosaic_embed_flower")
    p = sp.add_parser("central")
    p.add_argument("--epochs", type=int, default=6); p.add_argument("--lr", type=float, default=2e-5); p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=12); p.add_argument("--freeze", type=int, default=1); p.add_argument("--out-model", default="mosaic_embed_central_frozen")
    p = sp.add_parser("eval"); p.add_argument("--models", default="mosaic_embed_central_frozen,mosaic_embed_flower,mosaic_embed_flower_vm")
    a = ap.parse_args()
    {"selftest": cmd_selftest, "prep": cmd_prep, "client": cmd_client, "server": cmd_server, "central": cmd_central, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
