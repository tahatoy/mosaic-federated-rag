"""
Diferansiyel gizlilik yardımcıları (yalnızca numpy).

Önceki uygulama (flower_rag_client._apply_dp_noise) şu üç sorun nedeniyle bir (ε, δ)-DP garantisi VERMİYORDU:
  1. Kırpma, güncelleme farkına (delta) değil AĞIRLIKLARIN KENDİSİNE ve tensör tensör uygulanıyordu.
  2. diffprivlib yoksa yedek sınıf gürültüyü sigma * 0.01 ile (100 kat az) ekliyordu.
  3. Tur başına harcanan bütçe turlar arasında toplanmıyordu.
Bu modül (a) global L2 kırpma, (b) doğru kalibre edilmiş Gauss gürültüsü ve (c) basit bileşim hesabı sağlar.
Mekanizma, istemci güncellemesi üzerinde "istemci düzeyinde" (ekleme/çıkarma komşuluğu) Gauss mekanizmasıdır; DP-SGD DEĞİLDİR.
"""
from __future__ import annotations
import math
from typing import List, Sequence
import numpy as np


def gaussian_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
    """Klasik Gauss mekanizması: sigma = sqrt(2 ln(1.25/delta)) * S / epsilon  (kanıt ε<1 için; ε=1 sınırda)."""
    if epsilon <= 0 or not (0 < delta < 1) or sensitivity <= 0:
        raise ValueError("epsilon>0, 0<delta<1, sensitivity>0 olmalı")
    return math.sqrt(2.0 * math.log(1.25 / delta)) * sensitivity / epsilon


def global_l2_norm(arrays: Sequence[np.ndarray]) -> float:
    return float(math.sqrt(sum(float(np.sum(np.square(a.astype(np.float64)))) for a in arrays)))


def clip_update(delta: Sequence[np.ndarray], clip_norm: float) -> List[np.ndarray]:
    """Tüm tensörler üzerinden TEK global L2 normu clip_norm'u aşmayacak biçimde ölçekler."""
    n = global_l2_norm(delta)
    scale = min(1.0, clip_norm / (n + 1e-12))
    return [(d * scale).astype(d.dtype) for d in delta]


def privatize_update(global_params: Sequence[np.ndarray], local_params: Sequence[np.ndarray], *,
                     clip_norm: float, epsilon: float, delta: float, rng: np.random.Generator | None = None) -> List[np.ndarray]:
    """global + (kırpılmış yerel-global farkı) + N(0, (sigma*clip_norm)^2 I). Duyarlılık = clip_norm (ekleme/çıkarma)."""
    rng = rng or np.random.default_rng()
    sigma = gaussian_sigma(epsilon, delta, 1.0)  # birim duyarlılık için çarpan
    diff = [l.astype(np.float64) - g.astype(np.float64) for g, l in zip(global_params, local_params)]
    diff = clip_update(diff, clip_norm)
    out = []
    for g, d in zip(global_params, diff):
        noisy = d + rng.normal(0.0, sigma * clip_norm, size=d.shape)
        out.append((g.astype(np.float64) + noisy).astype(g.dtype))
    return out


def composed_epsilon(epsilon_per_round: float, rounds: int) -> float:
    """Temel bileşim (üst sınır): R tur => R*ε. (Daha sıkı muhasebe için RDP/moments accountant gerekir.)"""
    return epsilon_per_round * rounds
