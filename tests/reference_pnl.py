import numpy as np
from deep_hedging.payoffs import call_payoff 

def pnl_loop(payoff, S, d, p0, c=0.0):

    assert len(S) == len(d) + 1, "Length of S must be one more than length of d"

    total_costs = 0.0
    total_gains = 0.0

    for i in range(len(S)):
        if i != 0:
            shares_held = d[i-1]
        else:
            shares_held = 0
        if i == len(S)-1:
            current_shares = 0.0
        else:
            current_shares = d[i]

        trade_size = np.abs(current_shares - shares_held)
        costs = S[i] * c * trade_size
        total_costs = costs + total_costs

        gains = current_shares * (S[i+1] - S[i]) if i != len(S)-1 else 0.0
        total_gains = gains + total_gains 


    # PnL calculation
    res = - payoff + p0 + total_gains - total_costs

    return res