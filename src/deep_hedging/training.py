import torch
from deep_hedging.nn_policy import nn_policy
from deep_hedging.features import make_features
from deep_hedging.risk import entropic_risk, cvar_oce, cvar_empirical
from deep_hedging.engine import pnl
from deep_hedging.payoffs import call_payoff
from deep_hedging.gbm import simulate_gbm

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
    

def training(fnn, w, config):
    """
    trains the feed forward nn per batch
    we generate n_iters batch of price paths
    compute the loss
    backpropagation
    take an Adaptative Moment Estimation step
    """
    
    batch_size = config["training"]["batch_size"]
    n_iters = config["training"]["n_iters"]
    lr = config["training"]["lr"]
    
    S0 = config["market"]["S0"]
    mu = config["market"]["mu"]
    sigma = config["market"]["sigma"]
    T = config["market"]["T"]
    n_steps = config["market"]["n_steps"]
    
    params = list(fnn.parameters())
    
    if w is not None:
        params.append(w)
        
    optimizer = torch.optim.Adam(params, lr=lr)
    
    history = []
    
    for i in range(n_iters):
        
        _, S = simulate_gbm(S0, mu, sigma, T, n_steps, batch_size)
        pl = hedge_pnl(fnn, S, config)
        
        loss = risk_loss(pl, config, w)
        
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            loss_value = loss.item()
            print(f"Step: {i}/{n_iters} ({i/n_iters:.0%})      Loss: {loss_value:.4f}")

        history.append(loss_value)
        
    return history


def pnl_metrics(X, config):
    """
    calculates the pnl's metric : mean | std | entropic risk | VaR | CVaR
    """
    lam = config["risk"]["lam"]
    alpha = config["risk"]["alpha"]
    
    mean = torch.mean(X)
    std = torch.std(X)
    entropic = entropic_risk(X, lam)
    var, cvar = cvar_empirical(X, alpha)
    
    metrics = {
        "pnl_mean":mean.item(),
        "pnl_std":std.item(),
        "entropic_risk": entropic.item(),
        "var":var.item(),
        "cvar":cvar.item()
    }

    return metrics
    
    
    
    
    

    
    
    
    
    
    