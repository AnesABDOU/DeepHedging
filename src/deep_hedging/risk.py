import torch
import math

def entropic_risk(x: torch.Tensor, lam: float):
    """
    Obtained solving: U(c) = E[U(X)] with U(x) = -exp(-lambda x)
    Entropic risk: -c = rho(X) = (1/lambda) log E[exp(-lambda X)]
    x = samples of X
    """
    log_N = math.log(x.shape[0])
    rho = (1 / lam) * (torch.logsumexp(-lam * x, dim=0)- log_N)
    return rho
    

def cvar_oce(x: torch.Tensor, w: torch.Tensor, alpha: float):
    """
    OCE objective w + E[max(-X - w, 0)] / (1 - alpha)]
    Min over w = CVaR | argmin = VaR
    """
    

def cvar_empirical(x: torch.Tensor, alpha: float):
    """
    Mean of the worst ceil((1 - alpha) N) losses -X
    for test only
    """
    