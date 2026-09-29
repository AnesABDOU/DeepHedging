import numpy as np

def call_payoff(strike, S):
    return np.maximum(S - strike, 0)