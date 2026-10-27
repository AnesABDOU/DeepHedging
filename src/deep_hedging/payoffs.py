import numpy as np

def call_payoff(K, S):
    return np.maximum(S - K, 0)