from deep_hedging.black_scholes import black_scholes_call_delta

def bs_hedge_policy(S, sigma, T, t, K, r=0.0):
        
    tau = (T - t).view(1, -1, 1)
    delta = black_scholes_call_delta(S, K, tau, sigma, r)
    
    return delta
