"""
H2 deneyi (2/3): Federe istemcideki diferansiyel gizlilik uygulamasının analizi.

Kod tabanı (backend/client_setup/flower_rag_client.py) şunu yapar:
  * `_apply_dp_noise`: her AĞIRLIK tensörünü (güncelleme farkını değil) L2 normu <= S=1'e kırpar,
    ardından her koordinata N(0, sigma^2) ekler; sigma = sqrt(2 ln(1.25/delta)) * S / epsilon.
  * Per-örnek gradyan kırpma ve mini-batch örnekleme YOKTUR (DP-SGD değildir), tur başına gizlilik
    bütçesi bileşimi (composition) hesaplanmaz.

Bu betik (i) sigma ve sinyal/gürültü oranını sayısal olarak hesaplar, (ii) gerçek TurkEmbed ağırlıkları
üzerinde kırpma + gürültünün modeli ne kadar bozduğunu (kosinüs benzerliği) ölçer.

Çıktı: ai/experiments/results/h2_dp_analysis.json
"""
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EPS, DELTA, S = 1.0, 1e-5, 1.0
sigma = math.sqrt(2 * math.log(1.25 / DELTA)) * S / EPS
rng = np.random.default_rng(42)

res = {"epsilon": EPS, "delta": DELTA, "sensitivity": S, "sigma_per_coordinate": round(sigma, 4),
       "gaussian_mechanism_klasik_sınır_epsilon<1": "ε=1.0 sınırda; klasik formül ε∈(0,1) için kanıtlanmıştır (Dwork & Roth, Th. A.1)."}

# (i) _compute_local_update yolu: ortalama gömme vektörünün ilk 128 boyutu, ||v||<=1
d = 128
res["ortalama_vektör_yolu"] = {
    "boyut": d, "sinyal_normu_üst_sınır": S,
    "beklenen_gürültü_normu": round(sigma * math.sqrt(d), 2),
    "sinyal_gürültü_oranı_normlar": round(S / (sigma * math.sqrt(d)), 4)}

# (ii) bileşim: R tur için temel bileşim (üst sınır)
res["bileşim_temel"] = {f"{R}_tur_toplam_epsilon": R * EPS for R in (1, 5, 10, 50)}

# (iii) gerçek ağırlıklar
try:
    from safetensors import safe_open
    p = ROOT / "ai" / "models" / "embeddings" / "turkembed_finetuned" / "model.safetensors"
    n_tensors = 0; n_params = 0; norm_gt1 = 0; cos_list = []; relerr = []
    with safe_open(str(p), framework="numpy") as f:
        for k in f.keys():
            w = f.get_tensor(k).astype(np.float64).ravel()
            n_tensors += 1; n_params += w.size
            nrm = np.linalg.norm(w)
            if nrm > S:
                norm_gt1 += 1
                wc = w * (S / nrm)
            else:
                wc = w
            noisy = wc + rng.normal(0, sigma, size=w.shape)
            denom = np.linalg.norm(w) * np.linalg.norm(noisy)
            cos_list.append(float(np.dot(w, noisy) / denom) if denom > 0 else 0.0)
            relerr.append(float(np.linalg.norm(noisy - w) / (nrm + 1e-12)))
    res["gerçek_ağırlıklar"] = {
        "tensör_sayısı": n_tensors, "parametre_sayısı": n_params,
        "normu_1den_büyük_tensör": norm_gt1,
        "kırpma_sonrası_gürültülü_ile_orijinal_kosinüs_medyan": round(float(np.median(cos_list)), 5),
        "kosinüs_ortalama": round(float(np.mean(cos_list)), 5),
        "göreli_hata_medyan": round(float(np.median(relerr)), 2)}
except Exception as e:
    res["gerçek_ağırlıklar"] = {"hata": str(e)}

out = ROOT / "ai" / "experiments" / "results" / "h2_dp_analysis.json"
out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
