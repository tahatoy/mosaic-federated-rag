import numpy as np
import pytest
from backend.federated.dp import gaussian_sigma, clip_update, global_l2_norm, privatize_update, composed_epsilon


def test_sigma_value():
    assert gaussian_sigma(1.0, 1e-5, 1.0) == pytest.approx(4.8448, abs=1e-3)


def test_global_clip_bounds_norm_and_preserves_small():
    a = [np.ones((3, 3)) * 5, np.ones(4) * 5]
    c = clip_update(a, 1.0)
    assert global_l2_norm(c) == pytest.approx(1.0, abs=1e-6)
    small = [np.ones(4) * 0.01]
    assert np.allclose(clip_update(small, 1.0)[0], small[0])


def test_privatize_noise_scale_matches_sigma_times_clip():
    g = [np.zeros(200_000)]
    l = [np.zeros(200_000)]
    out = privatize_update(g, l, clip_norm=0.5, epsilon=1.0, delta=1e-5, rng=np.random.default_rng(0))
    assert np.std(out[0]) == pytest.approx(4.8448 * 0.5, rel=0.02)


def test_signal_is_delta_not_weights():
    g = [np.full(10, 100.0)]
    l = [np.full(10, 100.0)]  # değişim yok -> çıktı global çevresinde gürültü olmalı, 0'a kırpılmamalı
    out = privatize_update(g, l, clip_norm=1e-9, epsilon=1.0, delta=1e-5, rng=np.random.default_rng(1))
    assert abs(out[0].mean() - 100.0) < 1e-3


def test_composition():
    assert composed_epsilon(1.0, 5) == 5.0
