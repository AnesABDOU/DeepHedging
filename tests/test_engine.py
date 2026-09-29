import pytest
from tests.reference_pnl import pnl_loop
import numpy as np
import torch
from deep_hedging.engine import pnl

def test_loop_matches_hand_computation():
    S = [100.0, 90.0, 95.0, 98.0]
    #strike = 100.0
    p0 = 3.0
    d = [0.5, 0.3, 0.4]
    c = 0.01
    assert pnl_loop(payoff=0.0, S=S, d=d, p0=p0, c=c) == pytest.approx(-0.467)

def test_loop_flat_path_pays_only_costs():
    S = [100.0, 100.0, 100.0, 100.0]
    p0 = 3.0
    d = [0.5, 0.3, 0.4]
    c = 0.01
    assert pnl_loop(payoff=0.0, S=S, d=d, p0=p0, c=c) == pytest.approx(1.8)
    
def test_lengths_pb_raise():
    S = [100.0, 90.0, 95.0, 98.0]
    p0 = 3.0
    d = [0.5, 0.3]
    c = 0.01
    with pytest.raises(AssertionError):
        pnl_loop(payoff=0.0, S=S, d=d, p0=p0, c=c)
        
def test_engine_matches_reference_on_random_path():
    N = 100
    n = 5
    p0=5
    gen = torch.Generator().manual_seed(0)
    S = 100.0 * torch.exp(0.05 * torch.randn(N, n + 1, 1, generator=gen, dtype=torch.float64).cumsum(dim=1))
    delta = torch.rand(N, n, 1, generator=gen, dtype=torch.float64)
    payoff = torch.rand(N, generator=gen, dtype=torch.float64)
    
    #run N times the base engine and check against the torch engine
    expected = torch.tensor(
        [pnl_loop(payoff[i].item(), S[i, :, 0].tolist(), delta[i, :, 0].tolist(), p0=p0, c=0.01) for i in range(N)],
        dtype=torch.float64
    )
    
    assert torch.allclose(pnl(payoff, S, delta, p0, c=0.01), expected)