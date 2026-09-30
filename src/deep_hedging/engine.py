from torch import Tensor

def pnl(payoff: Tensor, S: Tensor, delta: Tensor, p0: float, c: float = 0.0) -> Tensor:
    """
        P&L of a option hedged with positions delta, one value per path.
        (N,) P&L = -payoff + p0 + (delta.S){T} - C{T}(delta)
        gains formula: Σ_{k=0}^{n−1} δ_k · (S_{k+1} − S_k), summed over instruments
        cost formula: Σ_{k=0}^{n} c · |δ_k − δ_{k−1}| · S_k, with δ_{−1} = δ_n = 0
    """
    N, n_dates, d = S.shape
    
    if delta.shape != (N, n_dates - 1, d):
        raise ValueError("delta must have shape (N, n_dates - 1, d)")
    
    if payoff.shape != (N,):
        raise ValueError("payoff must have shape (N,)")
    
    #substracting the prices of all instruments across all paths between every two timesteps
    #then multiply by the delta of the previous timestep to get the gains/losses for each timestep
    #sum across all timesteps and instruments givetotal gains/losses for each path
    
    #gains = (delta * (S[:, 1:, :] - S[:, :-1, :])).sum(dim=(1, 2))
    gains = (delta * S.diff(dim=1)).sum(dim=(1, 2))
    
    flat = delta.new_zeros(N, 1, d)
    trades = delta.diff(dim=1, prepend=flat, append=flat)
    costs = c * (S * trades.abs()).sum(dim=(1, 2))
    
    return p0 - payoff + gains - costs