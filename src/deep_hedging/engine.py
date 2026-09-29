from torch import Tensor

def pnl(payoff: Tensor, S: Tensor, delta: Tensor, p0: float, c: float = 0.0) -> Tensor:
    """P&L of a option hedged with positions delta, one value per path.
        (N,) P&L = -payoff + p0 + (delta.S){T} - C{T}(delta)
        with delta.S{T} = sum_{k=0}^{n-1} delta_k.(S_{k+1}-S_k)
        and C{T}(delta) = sum_{k=0}^{n-1} c * |delta_{k+1}-delta_k|.S_{k+1}
    """
    N, n_dates, d = S.shape
    
    assert delta.shape == (N, n_dates - 1, d), "delta must have shape (N, n_dates - 1, d)"
    assert payoff.shape == (N,), "payoff must be shape (N,)"
    #substracting the prices of all instruments across all paths between every two timesteps
    #then multiply by the delta of the previous timestep to get the gains/losses for each timestep
    #sum across all timesteps and instruments givetotal gains/losses for each path
    
    #gains = (delta * (S[:, 1:, :] - S[:, :-1, :])).sum(dim=(1, 2))
    gains = (delta * S.diff(dim=1)).sum(dim=(1, 2))
    
    flat = delta.new_zeros(N, 1, d)
    trades = delta.diff(dim=1, prepend=flat, append=flat)
    costs = c * (S * trades.abs()).sum(dim=(1, 2))
    
    return p0 - payoff + gains - costs