import torch

from deep_hedging.black_scholes import black_scholes_call_delta, black_scholes_call_price
from deep_hedging.engine import pnl
from deep_hedging.payoffs import call_payoff
from deep_hedging.delta import bs_hedge_policy
from deep_hedging.gbm import simulate_gbm


def test_black_scholes_call_price():
    S0 = torch.tensor(100.0)
    mu = torch.tensor(0.0)
    sigma = torch.tensor(0.2)
    T = torch.tensor(1.0)
    K = torch.tensor(100)
    
    hand_computed_call_price = torch.tensor(7.9656)
    
    assert torch.allclose(black_scholes_call_price(S0, K, T, sigma, r=0.0), hand_computed_call_price, atol=1e-3)
    
def test_black_scholes_call_delta():
    S = torch.tensor([80.0, 100.0, 120.0], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor(0.2)
    T = torch.tensor(1.0)
    K = torch.tensor(100)
    price = black_scholes_call_price(S, K, T, sigma)
    (grad_S,) = torch.autograd.grad(price.sum(), S)
    
    delta = black_scholes_call_delta(S, K, T, sigma, r=0.0)
    
    assert torch.allclose(delta, grad_S, atol=1e-6)
    
def _gbm(seed = 0):
    g = torch.Generator().manual_seed(seed)
    return g

def test_delta_hedge_simulation():
    # should converge towards the risk free rate (0 in our case), when n_steps -> +infinity
    S0 = torch.tensor(100.0)
    mu = torch.tensor(0.0)
    sigma = torch.tensor(0.2)
    T = torch.tensor(1.0)
    K = torch.tensor(100)
    n_steps = 10_000
    n_paths = 10_000
    
    t, S = simulate_gbm(S0=S0, mu=mu, sigma=sigma, T=T, n_steps=n_steps, n_paths=n_paths, generator=_gbm())
    delta = bs_hedge_policy(S, sigma, T, t, K, r=0.0)
    payoff = call_payoff(S[:, -1, 0], K)
    p0 = black_scholes_call_price(S0, K, T, sigma, r=0.0)
    pnl_ = pnl(payoff, S, delta[:, :-1, :], p0, c=0.0)
    
    print(f"Mean accross all paths of the PnL: {pnl_.mean():.2f}")
    
    assert torch.allclose(pnl_.mean(), torch.zeros_like(pnl_.mean()), atol=1e0)