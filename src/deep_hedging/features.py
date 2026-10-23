import torch

def make_features(S:torch.Tensor, t:torch.Tensor, T:float, K:float):
    """
    S : price paths (N, t+1, d_instruments), t : time (t+1), T : maturity, K: strike.
    GBM is Markov so the future depends only on today's price:
    Meaning at each step k : log(S_{k}/K) and tau = T-t_{k} is enough information without the entire price history till k (all the information available at time k : F_k).
    n trading dates between t_0 ... t_{n-1}
    outputs features concatenated of shape (N, n, d+1)
    """
    N = S.shape[0]
    log_moneyness = torch.log(S[:,:-1,:]/K)
    tau = (T-t[:-1]).view(1, -1, 1).expand(N, -1, -1)
    
    features = torch.cat([log_moneyness, tau], dim=-1)
    
    return features