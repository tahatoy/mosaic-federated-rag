"""
H2 deneyi (DP analizi, v2): GÜNCEL istemci mekanizmasının (backend/federated/dp.py -> privatize_update)
model ağırlıklarına etkisini ölçer.

Mekanizma (Tez Denklem 3.4):  w~ = w_g + clip(w_k - w_g, C) + N(0, (sigma*C)^2 I)
  * clip: tüm tensörler üzerinden TEK global L2 normu (dp.clip_update)
  * sigma = dp.gaussian_sigma(eps, delta, 1.0) ; C = DP_SENSITIVITY = 1.0

Kırpılmış güncellemenin global L2 normu en fazla C=1 olduğundan, gürültü normunun yanında (~sigma*C*sqrt(P))
ihmal edilebilir; bu nedenle ölçüm, yerel güncelleme sıfır alınarak (w_k = w_g) yapılır ve kırpılmış
güncellemenin etkisi için ayrıca analitik üst sınır raporlanır.
Bellek için büyük tensörler parça parça işlenir (dp.privatize_update ile matematiksel olarak eşdeğer).

Çıktı: ai/experiments/results/h2_dp_analysis_v2.json
"""
import json, math, sys
from pathlib import Path
import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend" / "federated"))
from dp import gaussian_sigma  # noqa: E402

EPS, DELTA, C = 1.0, 1e-5, 1.0
SIGMA = gaussian_sigma(EPS, DELTA, 1.0)
STD = SIGMA * C
CHUNK = 4_000_000
rng = np.random.default_rng(42)
p = ROOT / "ai" / "models" / "embeddings" / "turkembed_finetuned" / "model.safetensors"

n_t = n_p = 0
cos_t, rel_t = [], []
g_dot = g_ww = g_nn = 0.0  # global: <w, w~>, ||w||^2, ||w~||^2
with safe_open(str(p), framework="numpy") as f:
    for k in f.keys():
        w = f.get_tensor(k).ravel()
        n_t += 1; n_p += w.size
        dot = ww = nn = ee = 0.0
        for s in range(0, w.size, CHUNK):
            wc = w[s:s+CHUNK].astype(np.float64)
            noise = rng.standard_normal(wc.size) * STD
            noisy = wc + noise                      # w_g + clip(0) + gürültü
            dot += float(wc @ noisy); ww += float(wc @ wc); nn += float(noisy @ noisy); ee += float(noise @ noise)
        g_dot += dot; g_ww += ww; g_nn += nn
        denom = math.sqrt(ww) * math.sqrt(nn)
        cos_t.append(dot / denom if denom > 0 else 0.0)
        rel_t.append(math.sqrt(ee) / (math.sqrt(ww) + 1e-12))

res = {
    "mekanizma": "w~ = w_g + clip(w_k - w_g, C) + N(0,(sigma*C)^2 I); global L2 kirpma (dp.py)",
    "epsilon": EPS, "delta": DELTA, "C": C, "sigma": round(SIGMA, 4), "seed": 42,
    "tensor_sayisi": n_t, "parametre_sayisi": n_p,
    "agirlik_global_L2_normu": round(math.sqrt(g_ww), 2),
    "beklenen_gurultu_global_L2_normu": round(STD * math.sqrt(n_p), 2),
    "kirpilmis_guncelleme_normu_ust_sinir": C,
    "global_kosinus_w_vs_gurultulu": round(g_dot / math.sqrt(g_ww * g_nn), 5),
    "tensor_kosinus_medyan": round(float(np.median(cos_t)), 5),
    "tensor_kosinus_ortalama": round(float(np.mean(cos_t)), 5),
    "tensor_goreli_hata_medyan": round(float(np.median(rel_t)), 2),
    "bilesim_temel_toplam_epsilon": {f"{R}_tur": R * EPS for R in (1, 5, 10, 50)},
    "not": "Kirpilmis guncellemenin global normu <= C; gurultu normunun yaninda ihmal edilebilir.",
}
out = ROOT / "ai" / "experiments" / "results" / "h2_dp_analysis_v2.json"
out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
