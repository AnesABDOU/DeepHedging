import torch

def black_scholes_call_delta(S, K, tau, sigma, r=0.0):
    
    safeguard_tau = torch.where(tau > 0, tau, torch.ones_like(tau))
    
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * safeguard_tau) / (sigma * torch.sqrt(safeguard_tau))
    delta = torch.distributions.Normal(0, 1).cdf(d1)
    
    return torch.where(tau > 0, delta, (S > K).to(delta.dtype)) # delta at maturity is 1 if S > K, 0 otherwise

def black_scholes_call_price(S, K, tau, sigma, r=0.0):
    
    safeguard_tau = torch.where(tau > 0, tau, torch.ones_like(tau))
    
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * safeguard_tau) / (sigma * torch.sqrt(safeguard_tau))
    d2 = d1 - sigma * torch.sqrt(safeguard_tau)
    delta = torch.distributions.Normal(0, 1).cdf(d1)
    
    call_price = (S * delta) - K * torch.exp(-r * safeguard_tau) * torch.distributions.Normal(0, 1).cdf(d2)
    
    return torch.where(tau > 0, call_price, torch.clamp(S - K, min=0.0))