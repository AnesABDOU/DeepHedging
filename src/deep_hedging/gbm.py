import math
from torch import Tensor
import torch

# simulating N paths of a Geometric Brownian Motion
def simulate_gbm(S0: float, mu: float, sigma: float, T: float, n_steps: int, n_paths: int,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu"):
    
    """
    Solution of the SDE for a Geometric Brownian Motion  dS = mu * S * dt + sigma * S * dW
    S(t) = S0 * exp((mu - 0.5 * sigma^2) * dt + sigma * W(dt))
           with dt = t - t0, W(dt) = W(t) - W(t0) ~ N(0, dt)
    Args:
        S0 : Initial price
        mu : Drift
        sigma : Volatility
        T : Time horizon
        dt : Time step
        N : Number of paths

    Returns:
        t: Time points
        S: Simulated paths
    """

    if not (T > 0):
        raise ValueError("T > 0")
        
    dt = T / n_steps
    t = torch.linspace(0.0, T, n_steps + 1, dtype=dtype, device=device)

    z = torch.randn(n_paths, n_steps, generator=generator, dtype=dtype, device=device)
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z

    log_S = torch.cumsum(log_increments, dim=1)
    log_S = torch.nn.functional.pad(log_S, (1, 0))

    S = S0 * torch.exp(log_S)
    return t, S.unsqueeze(-1)