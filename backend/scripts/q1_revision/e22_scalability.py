"""
E22 — 2 → 100 şube ölçeklenebilirlik simülasyonu (Q1 hakem sorusu: "iki şubeyle gösterilen düzen onlarca şubede çalışır mı?").

Üç bağımsız eksen ölçülür; hepsi genel merkez GPU'sunda, gerçek veri ve gerçek modellerle:

1) fl — federe gömme eğitimi, N ∈ {2, 5, 10, 20, 50, 100} istemci (E3 ile aynı tarif: multilingual-e5-small, 3 tur, 1 yerel dönem,
   FedAvg; şube başına 2.500 sorgu–parça çifti her şubenin N/2 istemcisine bölünür — toplam veri sabit, istemci başına veri
   N büyüdükçe küçülür, yani muhafazakâr düzen). Ölçülen: MRR@10 / Hit@10 (600 kalıp + 419 yeniden ifade), tur başına
   aktarılan bayt (N × güncelleme × 2 yön), tam ince ayar ve LoRA. Ayrıca LoRA ile istemci düzeyi merkezi DP: her N için
   kırpma sınırı C, aynı N'deki gizliliksiz koşunun ortanca güncelleme normu; z ∈ {0.3, 1.0, 3.0} -> 3 turda Rényi ile
   ε ≈ 44.4 / 9.81 / 2.94 (δ = 1e-5). Ortalamadaki gürültü zC/N olduğundan aynı ε'da faydanın N ile nasıl değiştiği
   doğrudan görülür ("daha sıkı gizlilik için daha büyük federasyon" iddiasının sınanması).

2) retrieval — çapraz şube erişimi, N şube: Şube-1 parçaları N/2, Şube-2 parçaları N/2 parçaya (şubeye) bölünür, ortak 429
   parça HQ'dadır. Her şube canlı sistemdeki gibi ilk 8 adayı döndürür (MOSAIC-Embed, tam arama); HQ'da (a) yoğun skorla
   birleştirme, (b) havuzun tamamının çapraz kodlayıcıyla yeniden sıralanması (havuz = 8N + 8), (c) yoğun skora göre ilk 32
   adayla sınırlı yeniden sıralama. MRR@10 ve sorgu başına yeniden sıralama süresi (HQ GPU'su, ölçülmüş).

3) latency — dağıtım (fan-out) gecikmesi: gerçek şube servislerinin gecikme örnekleri (--probe: canlı VM'lere mTLS ile
   ölçülür; yoksa E16 özetine uydurulmuş log-normal) N bağımsız şube için örneklenir; istekler HQ'da yerel bir taklit
   servise (her biri örneklenen süre kadar bekleyip 8 parça döndürür) eşzamanlı HTTPX çağrılarıyla gönderilir, duvar saati
   süresi ölçülür (istemci tarafı ek yük dahil). Uçtan uca bilgi getirme = dağıtım + yeniden sıralama (2'den ölçülen).

Çıktı: ai/experiments/results/q1/e22_scalability.json (fazlar ayrı ayrı koşturulabilir; dosya birleştirilerek güncellenir)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics as st
import threading
import time

import numpy as np
import torch

from q1_common import OUT_DIR, ROOT, mean_ci, questions, rank_metrics, save

OUTF = "e22_scalability.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_out():
    p = OUT_DIR / OUTF
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def test_sets(qs):
    para = {r["reference"]: r["natural"] for r in json.loads((OUT_DIR / "e0_paraphrases.json").read_text(encoding="utf-8"))["kayitlar"] if r["natural"]}
    nq = [q for q in qs if q["reference"] in para]
    return {"cloze": (qs, [q["search_query"] for q in qs]), "natural": (nq, [para[q["reference"]] for q in nq])}


# ─────────────────────────────── 1) federe eğitim ───────────────────────────────
def make_clients_n(pairs, N, seed):
    """Her şubenin çiftleri N/2 istemciye bölünür (non-IID: bir istemci yalnızca tek şubenin verisini görür)."""
    rnd = random.Random(seed)
    cl = []
    for b, h in ((1, (N + 1) // 2), (2, max(1, N // 2))):  # tek N'de Şube-1 bir istemci fazla alır
        p = pairs[b][:]; rnd.shuffle(p)
        cl += [p[i::h] for i in range(h)]
    return [c for c in cl if len(c) >= 2]


def phase_fl(a, out):
    import e3_fl_embedding_dp as e3
    qs, docs, owner = questions(600)
    sets = test_sets(qs)
    pairs = e3.build_pairs_qp(docs, owner, a.per_branch, seed=0)
    R, E = 3, 1
    res = out.setdefault("fl", {"config": {"rounds": R, "local_epochs": E, "per_branch": a.per_branch, "Ns": a.ns,
                                           "lr_full": 2e-5, "lr_lora": 2e-4, "lora": "r=8, alpha=16, query/value", "delta": 1e-5,
                                           "zs": a.zs, "regime": "fixed total data, split across N/2 clients per branch"},
                                "runs": []})
    done = {(r["name"], r.get("N"), r.get("z"), r["seed"]) for r in res["runs"]}

    def nbytes(lora):
        m = e3.load_model(lora)
        n = sum(p.numel() for p in e3.tparams(m)); del m
        return int(n * 4)

    B_FULL, B_LORA = nbytes(False), nbytes(True)
    res["config"]["bytes_per_update_full"] = B_FULL; res["config"]["bytes_per_update_lora"] = B_LORA

    def record(name, N, seed, m, t0, **kw):
        ev = e3.evaluate(m, docs, sets)
        row = {"name": name, "N": N, "seed": seed, "sec": round(time.time() - t0, 1), **kw,
               **{f"{s}_{k}": v for s, d in ev.items() for k, v in d.items() if not k.startswith("mrr@10_owner")}}
        res["runs"].append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
        out["fl"] = res; save(OUTF, out)
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if ("base_e5_small", None, None, 0) not in done:
        t0 = time.time(); record("base_e5_small", None, 0, e3.load_model(False), t0)
    for N in a.ns:
        for seed in a.seeds:
            cl = make_clients_n(pairs, N, seed)
            sizes = [len(c) for c in cl]
            if ("fedavg_full", N, None, seed) not in done:
                t0 = time.time()
                record("fedavg_full", N, seed, e3.fl_run(cl, R, E, 2e-5, seed, False), t0, clients=len(cl),
                       pairs_per_client_median=int(np.median(sizes)), bytes_per_round=2 * len(cl) * B_FULL)
            norms = []
            if ("fedavg_lora", N, None, seed) not in done or any(("dp_central_lora", N, z, seed) not in done for z in a.zs):
                t0 = time.time()
                m = e3.fl_run(cl, R, E, 2e-4, seed, True, log_norms=norms)
                C = float(np.median(norms))
                res.setdefault("clip_C", {})[f"{N}_{seed}"] = round(C, 4)
                if ("fedavg_lora", N, None, seed) not in done:
                    record("fedavg_lora", N, seed, m, t0, clients=len(cl), bytes_per_round=2 * len(cl) * B_LORA,
                           update_norm_median=round(C, 4))
                else:
                    del m
            else:
                C = res["clip_C"][f"{N}_{seed}"]
            if seed in a.dp_seeds:
                for z in a.zs:
                    if ("dp_central_lora", N, z, seed) in done:
                        continue
                    t0 = time.time()
                    record("dp_central_lora", N, seed, e3.fl_run(cl, R, E, 2e-4, seed, True, dp={"z": z, "C": C, "mode": "central"}), t0,
                           z=z, C=round(C, 4), epsilon=round(e3.rdp_epsilon(z, R), 2), noise_std_on_mean=round(z * C / len(cl), 5),
                           clients=len(cl))
    # özet: her (ad, N, z) için tohum ortalaması
    summ = {}
    for r in res["runs"]:
        k = f'{r["name"]}|N={r.get("N")}|z={r.get("z")}'
        summ.setdefault(k, []).append(r)
    res["summary"] = {k: {"n_seeds": len(v), **{m: round(float(np.mean([x[m] for x in v])), 4) for m in ("cloze_mrr@10", "natural_mrr@10", "cloze_hit@10", "natural_hit@10")},
                          **({"epsilon": v[0]["epsilon"]} if "epsilon" in v[0] else {}),
                          **({"bytes_per_round": v[0]["bytes_per_round"]} if "bytes_per_round" in v[0] else {})} for k, v in summ.items()}
    out["fl"] = res; save(OUTF, out)


# ─────────────────────────────── 2) erişim ───────────────────────────────
def phase_retrieval(a, out):
    from sentence_transformers import CrossEncoder, SentenceTransformer
    qs, docs, owner = questions(600)
    sets = test_sets(qs)
    m = SentenceTransformer(str(ROOT / "ai/models/embeddings/turkembed_v3"), device=DEV); m.max_seq_length = 512
    ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device=DEV)
    with torch.no_grad():
        E = m.encode(["passage: " + d for d in docs], batch_size=128, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        Qs = {s: m.encode(["query: " + t for t in qt], batch_size=256, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
              for s, (_, qt) in sets.items()}
    rnd = random.Random(5)
    by_branch = {b: [i for i in range(len(docs)) if owner[i] == b] for b in (1, 2)}
    for b in by_branch:
        rnd.shuffle(by_branch[b])
    hq = [i for i in range(len(docs)) if owner[i] == 0]
    K, CAP = a.k, a.cap
    res = {"k_per_branch": K, "rerank_cap": CAP, "hq_chunks": len(hq), "by_N": {}}
    S = {s: (Qs[s] @ E.T).float() for s in Qs}  # tam arama skorları (n_q × n_doc)
    for N in a.ns:
        hs = {1: (N + 1) // 2, 2: max(1, N // 2)}
        shards = [by_branch[b][i::hs[b]] for b in (1, 2) for i in range(hs[b])] + [hq]
        idx = [torch.tensor(sh, device=S["cloze"].device) for sh in shards]
        row = {"branches": len(shards) - 1, "chunks_per_branch_median": int(np.median([len(x) for x in shards[:-1]]))}
        for s, (sq, qt) in sets.items():
            pools = []
            for j in range(len(sq)):
                cand = []
                for ix in idx:
                    sc = S[s][j, ix]
                    v, t = torch.topk(sc, min(K, len(ix)))
                    cand += list(zip(v.tolist(), ix[t].tolist()))
                cand.sort(key=lambda x: -x[0])
                pools.append(cand)
            dense = [rank_metrics([d for _, d in p], q["gold_idx"])["mrr@10"] for p, q in zip(pools, sq)]
            central = [rank_metrics(torch.topk(S[s][j], 100).indices.tolist(), q["gold_idx"])["mrr@10"] for j, q in enumerate(sq)]
            full, cap, t_full, t_cap = [], [], [], []
            for p, q, text in zip(pools, sq, qt):
                for kind, cand, acc, tt in (("full", p, full, t_full), ("cap", p[:CAP], cap, t_cap)):
                    if DEV == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    sc = ce.predict([[text, docs[d]] for _, d in cand], batch_size=128, show_progress_bar=False)
                    if DEV == "cuda":
                        torch.cuda.synchronize()
                    tt.append((time.perf_counter() - t0) * 1000)
                    acc.append(rank_metrics([cand[i][1] for i in np.argsort(-np.asarray(sc))], q["gold_idx"])["mrr@10"])
            row[s] = {"pool_size": len(pools[0]), "mrr@10_central_dense": mean_ci(central), "mrr@10_federated_dense": mean_ci(dense),
                      "mrr@10_rerank_full_pool": mean_ci(full), f"mrr@10_rerank_top{CAP}": mean_ci(cap),
                      "rerank_ms_full_pool": {"median": round(st.median(t_full), 1), "p95": round(float(np.percentile(t_full, 95)), 1)},
                      f"rerank_ms_top{CAP}": {"median": round(st.median(t_cap), 1), "p95": round(float(np.percentile(t_cap, 95)), 1)},
                      "dense_identical_to_central_share": round(float(np.mean([abs(x - y) < 1e-9 for x, y in zip(dense, central)])), 4),
                      # ağ trafiği: şubelerden HQ'ya taşınan aday metinlerinin UTF-8 boyutu (sorgu başına, JSON ek yükü hariç)
                      "candidate_kib_per_query_median": round(float(np.median([sum(len(docs[d].encode("utf-8")) for _, d in p) for p in pools])) / 1024, 1),
                      "requests_per_query": len(shards)}
        res["by_N"][str(N)] = row
        print("N", N, json.dumps(row, ensure_ascii=False), flush=True)
        out["retrieval"] = res; save(OUTF, out)


# ─────────────────────────────── 3) gecikme ───────────────────────────────
def probe_branches(n):
    """Canlı şube servislerine (mTLS) ve HQ ortak koleksiyonuna art arda istek; ham gecikme örnekleri (ms)."""
    import os
    import httpx
    from e2_crossbranch_generation import mtls_client, remote
    qs, _, _ = questions(600)
    b = {"b1": os.environ["BRANCH1_RAG_URL"], "b2": os.environ["BRANCH2_RAG_URL"]}
    hq = os.getenv("HQ_RAG_URL", "http://127.0.0.1:8003")
    smp = {"b1": [], "b2": [], "hq": []}
    with mtls_client() as mt, httpx.Client(timeout=60) as plain:
        for q in qs[:5]:
            remote(mt, b["b1"], q["search_query"]); remote(mt, b["b2"], q["search_query"]); remote(plain, hq, q["search_query"])
        for q in qs[5:5 + n]:
            for k, cl, url in (("b1", mt, b["b1"]), ("b2", mt, b["b2"]), ("hq", plain, hq)):
                t = time.perf_counter(); remote(cl, url, q["search_query"]); smp[k].append((time.perf_counter() - t) * 1000)
    return smp


def lognormal_from(median, p95, n, seed):
    mu = math.log(median); sigma = (math.log(p95) - mu) / 1.6449
    r = np.random.default_rng(seed)
    return list(r.lognormal(mu, sigma, n))


def _stub_main(port, payload):
    import uvicorn
    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/search")
    async def search(d: float = 40.0):
        await asyncio.sleep(d / 1000.0)
        return payload

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", loop="asyncio", backlog=4096, timeout_keep_alive=120)


class _Stub:
    def __init__(self, proc):
        self.proc = proc

    @property
    def should_exit(self):
        return False

    @should_exit.setter
    def should_exit(self, v):
        if v:
            self.proc.terminate(); self.proc.join(5)


def start_stub(port, payload):
    """Taklit şube servisi ayrı bir süreçte çalışır (ölçüm yapan istemciyle GIL paylaşmaz)."""
    import multiprocessing as mp
    import httpx
    proc = mp.get_context("spawn").Process(target=_stub_main, args=(port, payload), daemon=True)
    proc.start()
    for _ in range(150):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/search", params={"d": 0}, timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.2)
    return _Stub(proc)


def phase_latency(a, out):
    import httpx
    smp, src = None, "lognormal fit to E16 summary (median/p95 per service)"
    if out.get("latency_samples"):
        smp, src = out["latency_samples"]["samples"], out["latency_samples"]["source"]
    elif a.probe:
        try:
            smp = probe_branches(a.probe_n); src = f"live probe, {a.probe_n} sequential calls per service (mTLS to branch VMs)"
        except Exception as e:
            print("canlı ölçüm başarısız, E16 özetine dönülüyor:", repr(e)[:200], flush=True)
    if smp is None:
        smp = {"b1": lognormal_from(42.5, 55.6, 2000, 1), "b2": lognormal_from(42.8, 59.9, 2000, 2), "hq": lognormal_from(56.8, 69.1, 2000, 3)}
    branch_pool = smp["b1"] + smp["b2"]
    qs, docs, _ = questions(600)
    payload = {"chunks": [{"text": docs[i][:600], "score": 0.8, "branch": "x"} for i in range(8)]}
    srv = start_stub(a.stub_port, payload)
    rnd = random.Random(9)
    ret = out.get("retrieval", {}).get("by_N", {})
    res = {"latency_source": src, "service_ms": {k: {"median": round(st.median(v), 1), "p95": round(float(np.percentile(v, 95)), 1), "n": len(v)} for k, v in smp.items()},
           "by_N": {}}

    async def fan(cl, delays):
        for attempt in range(3):  # N = 100'de nadiren kopan bağlantı: denemeyi baştan tekrarla
            t0 = time.perf_counter()
            rs = await asyncio.gather(*[cl.get(f"http://127.0.0.1:{a.stub_port}/search", params={"d": round(d, 2)}) for d in delays],
                                      return_exceptions=True)
            if not any(isinstance(r, Exception) for r in rs):
                for r in rs:
                    r.json()
                return (time.perf_counter() - t0) * 1000
            res["failed_attempts"] = res.get("failed_attempts", 0) + 1
            await asyncio.sleep(0.5)
        raise RuntimeError("fan-out üç denemede de başarısız")

    async def run_all():
        lim = httpx.Limits(max_connections=256, max_keepalive_connections=256)
        async with httpx.AsyncClient(timeout=30, limits=lim) as cl:
            await fan(cl, [1.0] * 8)  # ısınma
            for N in a.ns:
                wall, ideal = [], []
                for _ in range(a.trials):
                    d = [rnd.choice(branch_pool) for _ in range(N)] + [rnd.choice(smp["hq"])]
                    ideal.append(max(d))
                    wall.append(await fan(cl, d))
                row = {"fanout_ms": {"median": round(st.median(wall), 1), "p95": round(float(np.percentile(wall, 95)), 1)},
                       "max_of_services_ms": {"median": round(st.median(ideal), 1), "p95": round(float(np.percentile(ideal, 95)), 1)},
                       "hq_overhead_ms_median": round(st.median([w - i for w, i in zip(wall, ideal)]), 1)}
                r = ret.get(str(N), {}).get("natural") or ret.get(str(N), {}).get("cloze")
                if r:
                    cap_key = [k for k in r if k.startswith("rerank_ms_top")][0]
                    row["retrieval_total_ms_median_full_rerank"] = round(row["fanout_ms"]["median"] + r["rerank_ms_full_pool"]["median"], 1)
                    row["retrieval_total_ms_median_capped_rerank"] = round(row["fanout_ms"]["median"] + r[cap_key]["median"], 1)
                res["by_N"][str(N)] = row
                print("N", N, json.dumps(row), flush=True)
                out["latency"] = res; save(OUTF, out)

    asyncio.run(run_all())
    srv.should_exit = True
    out["latency"] = res; save(OUTF, out)


def phase_probe(a, out):
    """Canlı şube servislerinden gecikme örnekleri (servisler açıkken, gecenin başında); latency fazı bunları kullanır."""
    smp = probe_branches(a.probe_n)
    out["latency_samples"] = {"source": f"live probe, {a.probe_n} sequential calls per service (mTLS to branch VMs)", "samples": smp}
    save(OUTF, out)
    print({k: (round(st.median(v), 1), round(float(np.percentile(v, 95)), 1)) for k, v in smp.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="retrieval,latency,fl")
    ap.add_argument("--ns", default="2,5,10,20,50,100")
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--dp-seeds", default="0,1")
    ap.add_argument("--zs", default="0.3,1.0,3.0"); ap.add_argument("--per-branch", type=int, default=2500)
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--cap", type=int, default=32)
    ap.add_argument("--probe", type=int, default=1); ap.add_argument("--probe-n", type=int, default=300)
    ap.add_argument("--trials", type=int, default=300); ap.add_argument("--stub-port", type=int, default=9310)
    a = ap.parse_args()
    a.ns = [int(x) for x in a.ns.split(",")]; a.seeds = [int(x) for x in a.seeds.split(",")]
    a.dp_seeds = [int(x) for x in a.dp_seeds.split(",")]; a.zs = [float(x) for x in a.zs.split(",")]
    out = load_out()
    for ph in a.phases.split(","):
        t0 = time.time()
        {"fl": phase_fl, "retrieval": phase_retrieval, "latency": phase_latency, "probe": phase_probe}[ph.strip()](a, out)
        print(f"faz {ph} bitti: {(time.time() - t0) / 60:.1f} dk", flush=True)


if __name__ == "__main__":
    main()
