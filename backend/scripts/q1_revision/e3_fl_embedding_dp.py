"""
E3 — Gömme modelinin federe eğitimi ve diferansiyel gizlilik: fayda-gizlilik eğrisi.

Tezde federe katmanın gömme düzlemi tasarlanmış, fakat (i) federe eğitim yalnızca yönlendiricide ve bir
simülasyonda koşulmuş, (ii) DP yalnızca tek bir ayarda (ε=1/tur, istemcide gürültü, C=1) analitik olarak
incelenmişti. Bu deney aynı algoritmaları (FedAvg, FedProx μ=0.01, istemci güncellemesinde global L2 kırpma +
Gauss gürültüsü) gerçek eğitimle koşar ve sonucu çapraz şube erişim görevinde ölçer.

Veri: her şubenin kendi korpusundaki (talimat, yanıt) çiftleri; test sorularının (E1, n=600) altın parçaları
eğitimden çıkarılır. İstemciler: K=2 (iki şube) ve K=10 (her şube 5 alt istemciye bölünür; non-IID korunur).
Değerlendirme: multilingual-e5-small tabanından başlayan modelin tüm şube parçaları üzerinde MRR@10 / Hit@10.

Gizlilik muhasebesi: tam katılımlı Gauss mekanizması için Rényi DP, tur başına α/(2z²), R tur bileşimi ve
ε(δ) = min_α [R·α/(2z²) + ln(1/δ)/(α−1)], δ = 1e-5 (istemci düzeyi, ekleme/çıkarma komşuluğu).
  merkezi DP (güvenli birleştirme varsayımı): ortalamaya bir kez N(0, (zC/K)²) eklenir
  yerel DP (tezdeki tasarım): her istemci kendi güncellemesine N(0, (zC)²) ekler -> ortalamada zC/√K
Çıktı: ai/experiments/results/q1/e3_fl_embedding.json
"""
import argparse
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from q1_common import OUT_DIR, load_branch, questions, rank_metrics, save

BASE = "intfloat/multilingual-e5-small"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def rdp_epsilon(z, rounds, delta=1e-5):
    if z <= 0:
        return float("inf")
    best = float("inf")
    for a in [1 + x / 10 for x in range(1, 100)] + list(range(11, 257)):
        best = min(best, rounds * a / (2 * z * z) + math.log(1 / delta) / (a - 1))
    return best


def build_pairs(test_texts, per_branch, seed):
    rnd = random.Random(seed)
    out = {}
    for b in (1, 2):
        rows = [r for r in load_branch(f"branch{b}") if r["text"] not in test_texts]
        seen, pairs = set(), []
        for r in rows:
            q, p = (r.get("instruction") or "").strip(), (r.get("response") or "").strip()
            if len(q) > 10 and len(p) > 40 and q not in seen:
                seen.add(q); pairs.append((q, p[:600]))
        rnd.shuffle(pairs)
        out[b] = pairs[:per_branch]
    return out


def make_clients(pairs, K, seed):
    if K == 2:
        return [pairs[1], pairs[2]]
    rnd = random.Random(seed)
    cl = []
    for b in (1, 2):
        p = pairs[b][:]; rnd.shuffle(p)
        h = K // 2
        cl += [p[i::h] for i in range(h)]
    return cl


def load_model(lora):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(BASE, device=DEV)
    m.max_seq_length = 256
    if lora:
        from peft import LoraConfig, get_peft_model
        m[0].auto_model = get_peft_model(m[0].auto_model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["query", "value"]))
    return m


def tparams(m):
    return [p for p in m.parameters() if p.requires_grad]


def get_vec(m):
    return [p.detach().clone() for p in tparams(m)]


def set_vec(m, v):
    with torch.no_grad():
        for p, x in zip(tparams(m), v):
            p.copy_(x)


def embed(m, texts):
    f = m.tokenize(texts)
    f = {k: v.to(DEV) for k, v in f.items()}
    return F.normalize(m(f)["sentence_embedding"], dim=-1)


def local_train(m, pairs, epochs, lr, seed, mu=0.0, anchor=None, bs=32):
    m.train()
    opt = torch.optim.AdamW(tparams(m), lr=lr)
    rnd = random.Random(seed)
    for _ in range(epochs):
        p = pairs[:]; rnd.shuffle(p)
        for i in range(0, len(p) - 1, bs):
            b = p[i:i + bs]
            if len(b) < 2:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                q = embed(m, ["query: " + x for x, _ in b]); d = embed(m, ["passage: " + y for _, y in b])
                loss = F.cross_entropy(20.0 * q @ d.T, torch.arange(len(b), device=DEV))
            if mu > 0:
                loss = loss + mu / 2 * sum(((w - a) ** 2).sum() for w, a in zip(tparams(m), anchor))
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()


def fl_run(clients, rounds, epochs, lr, seed, lora, mu=0.0, dp=None, log_norms=None):
    torch.manual_seed(seed)
    m = load_model(lora)
    g = get_vec(m)
    K = len(clients)
    gen = torch.Generator(device=DEV).manual_seed(seed + 99)
    for r in range(rounds):
        acc = [torch.zeros_like(x) for x in g]
        wsum = 0.0
        for k, data in enumerate(clients):
            set_vec(m, g)
            local_train(m, data, epochs, lr, seed * 1000 + r * 10 + k, mu=mu, anchor=g if mu > 0 else None)
            delta = [x - y for x, y in zip(get_vec(m), g)]
            nrm = math.sqrt(sum(float((d.float() ** 2).sum()) for d in delta))
            if log_norms is not None:
                log_norms.append(nrm)
            if dp:
                s = min(1.0, dp["C"] / (nrm + 1e-12))
                delta = [d * s for d in delta]
                if dp["mode"] == "local":
                    delta = [d + torch.randn(d.shape, device=DEV, generator=gen) * dp["z"] * dp["C"] for d in delta]
                w = 1.0
            else:
                w = float(len(data))
            for a, d in zip(acc, delta):
                a.add_(d, alpha=w)
            wsum += w
        if dp and dp["mode"] == "central":
            acc = [a + torch.randn(a.shape, device=DEV, generator=gen) * dp["z"] * dp["C"] for a in acc]
        g = [x + a / wsum for x, a in zip(g, acc)]
    set_vec(m, g)
    return m


def evaluate(m, docs, sets):
    m.eval()
    with torch.no_grad():
        E = m.encode(["passage: " + d for d in docs], batch_size=256, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        res = {}
        for name, (qs, qt) in sets.items():
            Q = m.encode(["query: " + q for q in qt], batch_size=256, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
            top = torch.topk((Q @ E.T).float(), 100, dim=1).indices.cpu().tolist()
            per = [rank_metrics(t, q["gold_idx"]) for t, q in zip(top, qs)]
            res[name] = {k: round(float(np.mean([p[k] for p in per])), 4) for k in ("mrr@10", "hit@1", "hit@10", "ndcg@10")}
            # sahip şubeye göre ayrım (yerel modelin diğer şubenin sorularındaki başarımı için)
            for b in (1, 2):
                sel = [p["mrr@10"] for p, q in zip(per, qs) if q["owner_branch"] == b]
                res[name][f"mrr@10_owner{b}"] = round(float(np.mean(sel)), 4) if sel else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--per-branch", type=int, default=2500)
    ap.add_argument("--zs", default="0.1,0.3,1.0,3.0")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    qs, docs, owner = questions(600)
    para = {}
    pp = OUT_DIR / "e0_paraphrases.json"
    if pp.exists():
        para = {r["reference"]: r["natural"] for r in json.loads(pp.read_text(encoding="utf-8"))["kayitlar"] if r["natural"]}
    sets = {"cloze": (qs, [q["search_query"] for q in qs])}
    nq = [q for q in qs if q["reference"] in para]
    if nq:
        sets["natural"] = (nq, [para[q["reference"]] for q in nq])
    test_texts = {docs[i] for q in qs for i in q["gold_idx"]}
    pairs = build_pairs(test_texts, a.per_branch, seed=0)
    print("çift:", {b: len(p) for b, p in pairs.items()}, "test:", {k: len(v[0]) for k, v in sets.items()}, flush=True)
    R, E = a.rounds, 1
    LR_FULL, LR_LORA = 2e-5, 2e-4
    out = {"config": {"rounds": R, "local_epochs": E, "per_branch": a.per_branch, "seeds": seeds, "lr_full": LR_FULL, "lr_lora": LR_LORA,
                      "fedprox_mu": 0.01, "delta": 1e-5, "lora": "r=8, alpha=16, query/value"}, "runs": []}

    def record(name, seed, m, t0, **kw):
        ev = evaluate(m, docs, sets)
        row = {"name": name, "seed": seed, "sec": round(time.time() - t0, 1), **kw, **{f"{s}_{k}": v for s, d in ev.items() for k, v in d.items()}}
        out["runs"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        save("e3_fl_embedding.json", out)
        del m; torch.cuda.empty_cache()

    t0 = time.time(); record("base_e5_small", 0, load_model(False), t0)
    for seed in seeds:
        allp = pairs[1] + pairs[2]
        for nm, data, lora in (("central_full", allp, False), ("local_b1_full", pairs[1], False), ("local_b2_full", pairs[2], False)):
            t0 = time.time(); m = load_model(lora); torch.manual_seed(seed)
            local_train(m, data, R * E, LR_FULL, seed); record(nm, seed, m, t0)
        for K in (2, 10):
            cl = make_clients(pairs, K, seed)
            for alg, mu in (("fedavg", 0.0), ("fedprox", 0.01)):
                t0 = time.time(); record(f"{alg}_K{K}_full", seed, fl_run(cl, R, E, LR_FULL, seed, False, mu=mu), t0, K=K)
        # LoRA (DP deneyleri için düşük boyutlu güncelleme)
        t0 = time.time(); m = load_model(True); torch.manual_seed(seed); local_train(m, allp, R * E, LR_LORA, seed); record("central_lora", seed, m, t0)
        cl = make_clients(pairs, 10, seed)
        norms = []
        t0 = time.time(); record("fedavg_K10_lora", seed, fl_run(cl, R, E, LR_LORA, seed, True, log_norms=norms), t0, K=10,
                                 update_norm_median=round(float(np.median(norms)), 4))
        C = float(np.median(norms))
        for z in [float(x) for x in a.zs.split(",")]:
            eps = rdp_epsilon(z, R)
            for mode in ("central", "local"):
                t0 = time.time()
                record(f"dp_{mode}_K10_lora", seed, fl_run(cl, R, E, LR_LORA, seed, True, dp={"z": z, "C": C, "mode": mode}), t0,
                       K=10, z=z, C=round(C, 4), epsilon=round(eps, 2))
        if a.quick:
            break
    # tezdeki ayar: ε=1/tur (klasik Gauss, z=4.8448), C=1, istemcide gürültü, R tur -> RDP ile toplam ε
    out["thesis_setting"] = {"z": 4.8448, "C": 1.0, "rdp_epsilon_R": round(rdp_epsilon(4.8448, R), 3), "basic_composition_epsilon_R": float(R)}
    save("e3_fl_embedding.json", out)


if __name__ == "__main__":
    main()
