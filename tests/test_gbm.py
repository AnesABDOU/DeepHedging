import math

import pytest
import torch

from deep_hedging.gbm import simulate_gbm

def _gbm(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return g

def test_shapes():
    t, S = simulate_gbm(S0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=4, n_paths=5, generator=_gbm())
    assert t.shape == (5,)
    assert S.shape == (5, 5, 1)
    assert t[0] == 0.0 and math.isclose(t[-1].item(), 1.0)
    
def test_increments_distribution():
    S0 = 100
    mu = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 10
    n_paths = 50000
    generator = _gbm()

    _, S = simulate_gbm(S0=S0, mu=mu, sigma=sigma, T=T, n_steps=n_steps, n_paths=n_paths, generator=generator)
    r = torch.diff(torch.log(S[:,:, 0]), dim=1).flatten()
    dt = T / n_steps
    m = r.numel()
    
    true_mu = (mu - 0.5 * sigma**2) * dt
    true_var = sigma**2 * dt
    assert abs(r.mean().item() - true_mu) < 4 * math.sqrt(true_var / m)
    assert abs(r.var().item() - true_var) < 4 * true_mu * math.sqrt(2 / (m - 1))
