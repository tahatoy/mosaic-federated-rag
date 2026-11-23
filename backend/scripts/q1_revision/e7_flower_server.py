"""
E7 — Gerçek Flower sunucusu (genel merkezde) ve tur sonu değerlendirmesi.

İki şube VM'i istemci olarak bağlanır; her turda LoRA ağırlıkları (yalnızca adaptör tensörleri) TLS kanalı üzerinden
gelir ve FedAvg ya da FedProx (μ=0.01) ile birleştirilir. Birleştirilmiş model her turdan sonra genel merkezde
çapraz şube soru kümelerinde (E1 ile aynı) değerlendirilir; bu değerlendirme yalnızca deney ölçümü içindir.
Kaydedilenler: tur süreleri, istemci eğitim süreleri, istemci başına aktarılan bayt, MRR@10.
Çıktı: ai/experiments/results/q1/e7_flower_<strateji>.json
"""
import argparse
import json
import time

import flwr as fl
import numpy as np
import torch

from q1_common import OUT_DIR, questions, rank_metrics, save
from e7_flower_client import add_lora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9090); ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--strategy", default="fedavg", choices=["fedavg", "fedprox"])
    ap.add_argument("--model", required=True); ap.add_argument("--tls-dir", default="")
    ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(a.model, device="cpu"); m.max_seq_length = 128
    params = add_lora(m, seed=0)
    m.to(dev)
    qs, docs, _ = questions(600)
    para = {r["reference"]: r["natural"] for r in json.loads((OUT_DIR / "e0_paraphrases.json").read_text(encoding="utf-8"))["kayitlar"] if r["natural"]}
    nq = [q for q in qs if q["reference"] in para]
    sets = {"cloze": (qs, [q["search_query"] for q in qs]), "natural": (nq, [para[q["reference"]] for q in nq])}
    log = {"strategy": a.strategy, "rounds": a.rounds, "tls": bool(a.tls_dir), "eval": [], "fit": [], "t_round_end": []}
    t_start = time.time()

    def set_params(arrs):
        with torch.no_grad():
            for p, x in zip(params, arrs):
                p.copy_(torch.from_numpy(x).to(p.device))

    def evaluate(server_round, parameters, config):
        set_params(parameters)
        m.eval(); res = {"round": server_round, "t": round(time.time() - t_start, 1)}
        with torch.no_grad():
            E = m.encode(["passage: " + d for d in docs], batch_size=256, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
            for s, (sq, qt) in sets.items():
                Q = m.encode(["query: " + q for q in qt], batch_size=256, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
                top = torch.topk((Q @ E.T).float(), 100, dim=1).indices.cpu().tolist()
                per = [rank_metrics(t, q["gold_idx"]) for t, q in zip(top, sq)]
                res[f"{s}_mrr@10"] = round(float(np.mean([p["mrr@10"] for p in per])), 4)
                res[f"{s}_hit@10"] = round(float(np.mean([p["hit@10"] for p in per])), 4)
        log["eval"].append(res); print("DEĞERLENDİRME", json.dumps(res), flush=True)
        save(f"e7_flower_{a.tag or a.strategy}.json", log)
        return 0.0, res

    def fit_cfg(rnd):
        return {"round": rnd, "lr": 2e-4, "batch": 16, "epochs": a.epochs, "proximal_mu": 0.01 if a.strategy == "fedprox" else 0.0}

    def fit_agg(metrics):
        log["fit"].append([{"n": n, **mm} for n, mm in metrics])
        log["t_round_end"].append(round(time.time() - t_start, 1))
        return {"fit_s_max": max(mm["fit_s"] for _, mm in metrics)}

    init = fl.common.ndarrays_to_parameters([p.detach().cpu().numpy() for p in params])
    kw = dict(fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=2, min_available_clients=2, min_evaluate_clients=0,
              initial_parameters=init, evaluate_fn=evaluate, on_fit_config_fn=fit_cfg, fit_metrics_aggregation_fn=fit_agg)
    strat = fl.server.strategy.FedProx(proximal_mu=0.01, **kw) if a.strategy == "fedprox" else fl.server.strategy.FedAvg(**kw)
    cert = None
    if a.tls_dir:
        from pathlib import Path
        t = Path(a.tls_dir)
        cert = ((t / "ca.crt").read_bytes(), (t / "fl_server.crt").read_bytes(), (t / "fl_server.key").read_bytes())
    log["lora_params"] = int(sum(p.numel() for p in params)); log["bytes_per_update"] = int(sum(p.numel() * 4 for p in params))
    fl.server.start_server(server_address=f"0.0.0.0:{a.port}", config=fl.server.ServerConfig(num_rounds=a.rounds),
                           strategy=strat, certificates=cert)
    log["total_s"] = round(time.time() - t_start, 1)
    save(f"e7_flower_{a.tag or a.strategy}.json", log)


if __name__ == "__main__":
    main()
