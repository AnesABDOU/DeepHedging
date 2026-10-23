import torch
from deep_hedging.nn_policy import build_nn, nn_policy
from deep_hedging.features import make_features
from deep_hedging.risk import entropic_risk
from deep_hedging.engine import pnl
from deep_hedging.payoffs import call_payoff
from deep_hedging.gbm import simulate_gbm

def test_nn_policy_shape():
    features = torch.randn(64,10,2)
    delta = nn_policy(build_nn(n_inputs=3), features)
    assert delta.shape == (64, 10, 1)
    assert torch.isfinite(delta).all()
    
def _gbm(seed = 0):
    g = torch.Generator().manual_seed(seed)
    return g
    
def test_gradient_reach():
    
    torch.manual_seed(0)
    
    S0 = 100.0
    K = 100.0
    mu = 0
    sigma = 0.2
    T = 1.0
    n_steps = 10
    n_paths = 256
    p0 = 5.0
    generator = _gbm()
    
    t, S = simulate_gbm(S0=S0, mu=mu, sigma=sigma, T=T, n_steps=n_steps, n_paths=n_paths, generator=generator)
    features = make_features(S, t, T, K)
    net = build_nn(n_inputs=3)
    delta = nn_policy(net, features)
    
    loss = entropic_risk(pnl(call_payoff(K, S[:,-1,0]), S, delta, p0, c=0.01), lam=0.3)
    loss.backward()
    
    for p in net.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0    
    
    
