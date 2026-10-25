import torch
from deep_hedging.nn_policy import build_nn, nn_policy
from deep_hedging.features import make_features
from deep_hedging.risk import entropic_risk
from deep_hedging.engine import pnl
from deep_hedging.payoffs import call_payoff
from deep_hedging.gbm import simulate_gbm

from deep_hedging.risk import cvar_oce, entropic_risk

def hedge_pnl(fnn, S, config):
    """
    calculates the final pnl of someone who sold a call
    """
    T = config["market"]["T"]
    K = config["option"]["K"]
    c = config["costs"]["c"]
    n_steps = config["market"]["n_steps"]
    
    t = torch.linspace(0.0, T, n_steps+1, dtype=S.dtype)
    
    features = make_features(S, t, T, K)
    delta = nn_policy(fnn, features)
    
    payoff = call_payoff(K, S[:, -1, 0])
    
    pl = pnl(payoff, S, delta, p0=0.0, c=c)
    
    return pl


def risk_loss(X, config, w=None):
    """
    reads the risk measure wanted
    gives us the ability to switch between risk measures directly from the config file
    """
    name = config["risk"]["name"]
    
    if name.lower()=="cvar":
        if w is None:
            raise ValueError("Calling CVaR without w")
        else:
            alpha = config["risk"]["alpha"]
            return cvar_oce(X, w, alpha)
    
    elif name.lower()=="entropic":
        lam = config["risk"]["lam"]
        return entropic_risk(X, lam)
    
    else:
        raise ValueError(f"Unknown Risk Measure Called, you called: {name}")
    