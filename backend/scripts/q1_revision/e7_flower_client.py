"""
E7 — Gerçek Flower istemcisi (şube VM'inde çalışır).

Şubenin kendi (talimat, yanıt) çiftleriyle multilingual-e5-small üzerindeki LoRA adaptörünü (r=8, alpha=16,
sorgu/değer izdüşümleri) yerel olarak eğitir ve yalnızca LoRA ağırlıklarını Flower üzerinden genel merkeze gönderir.
Metin, gömme veya vektör veritabanı içeriği şubeden çıkmaz. Kanal TLS ile korunur (--ca verilirse).
Kullanım: python e7_flower_client.py --server 10.0.0.10:9090 --branch 1 --chroma data/chromadb/branch_01 --exclude test_sha1.txt --model <yol> --ca ca.crt
"""
import argparse
import json
import math
import random
import time

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16, seed=0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        torch.manual_seed(seed)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.s = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.s


def add_lora(st_model, seed=0):
    bert = st_model[0].auto_model
    for p in bert.parameters():
        p.requires_grad = False
    i = 0
    for layer in bert.encoder.layer:
        att = layer.attention.self
        att.query = LoRALinear(att.query, seed=seed + i); i += 1
        att.value = LoRALinear(att.value, seed=seed + i); i += 1
    return [p for n, p in st_model.named_parameters() if n.endswith(".A") or n.endswith(".B")]


def pairs_from_chroma(path, exclude_file, n, seed=0):
    """Şubenin kendi ChromaDB koleksiyonundaki parçalardan (talimat, yanıt) çiftleri; test parçaları SHA-1 ile dışlanır."""
    import hashlib
    import chromadb
    excl = {l.strip() for l in open(exclude_file)} if exclude_file else set()
    cl = chromadb.PersistentClient(path=path)
    cols = sorted(cl.list_collections(), key=lambda c: -cl.get_collection(c.name if hasattr(c, "name") else c).count())
    col = cl.get_collection(cols[0].name if hasattr(cols[0], "name") else cols[0])
    docs = col.get(include=["documents"])["documents"]
    seen, out = set(), []
    for t in docs:
        if not t or hashlib.sha1(t.encode("utf-8")).hexdigest() in excl or "\n\n" not in t:
            continue
        q, r = t.split("\n\n", 1)
        q, r = q.strip(), r.strip()
        if len(q) > 10 and len(r) > 40 and q not in seen:
            seen.add(q); out.append((q, r[:600]))
    random.Random(seed).shuffle(out)
    return out[:n], len(docs)


def embed(m, texts):
    f = m.tokenize(texts)
    return F.normalize(m(f)["sentence_embedding"], dim=-1)


class Client(fl.client.NumPyClient):
    def __init__(self, model, params, pairs, branch):
        self.m, self.params, self.pairs, self.branch = model, params, pairs, branch

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.params]

    def _set(self, arrs):
        with torch.no_grad():
            for p, a in zip(self.params, arrs):
                p.copy_(torch.from_numpy(a))

    def fit(self, parameters, config):
        t0 = time.time()
        self._set(parameters)
        anchor = [p.detach().clone() for p in self.params]
        mu = float(config.get("proximal_mu", 0.0))
        lr, bs, ep = float(config.get("lr", 2e-4)), int(config.get("batch", 16)), int(config.get("epochs", 1))
        opt = torch.optim.AdamW(self.params, lr=lr)
        rnd = random.Random(int(config.get("round", 0)) * 10 + self.branch)
        self.m.train(); steps = 0; loss_sum = 0.0
        for _ in range(ep):
            p = self.pairs[:]; rnd.shuffle(p)
            for i in range(0, len(p) - 1, bs):
                b = p[i:i + bs]
                q = embed(self.m, ["query: " + x for x, _ in b]); d = embed(self.m, ["passage: " + y for _, y in b])
                loss = F.cross_entropy(20.0 * q @ d.T, torch.arange(len(b)))
                if mu > 0:
                    loss = loss + mu / 2 * sum(((w - a) ** 2).sum() for w, a in zip(self.params, anchor))
                opt.zero_grad(); loss.backward(); opt.step(); steps += 1; loss_sum += float(loss)
        self.m.eval()
        out = self.get_parameters({})
        nbytes = int(sum(a.nbytes for a in out))
        print(json.dumps({"round": config.get("round"), "fit_s": round(time.time() - t0, 1), "steps": steps,
                          "loss": round(loss_sum / max(1, steps), 4), "bytes": nbytes}), flush=True)
        return out, len(self.pairs), {"fit_s": time.time() - t0, "branch": self.branch, "loss": loss_sum / max(1, steps), "bytes": nbytes}

    def evaluate(self, parameters, config):
        return 0.0, 1, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True); ap.add_argument("--branch", type=int, required=True)
    ap.add_argument("--chroma", required=True); ap.add_argument("--exclude", default=""); ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ca", default=""); ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(0)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(a.model, device="cpu"); m.max_seq_length = 128
    params = add_lora(m, seed=0)
    pairs, ndocs = pairs_from_chroma(a.chroma, a.exclude, a.n)
    print(json.dumps({"branch": a.branch, "local_chunks": ndocs, "pairs": len(pairs), "lora_params": int(sum(p.numel() for p in params)),
                      "tls": bool(a.ca)}), flush=True)
    kw = {"root_certificates": open(a.ca, "rb").read()} if a.ca else {"insecure": True}
    fl.client.start_client(server_address=a.server, client=Client(m, params, pairs, a.branch).to_client(), **kw)


if __name__ == "__main__":
    main()
