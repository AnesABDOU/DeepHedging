import pytest
from tests.reference_pnl import pnl_loop
import numpy as np

def test_loop_matches_hand_computation():
    S = [100.0, 90.0, 95.0, 98.0]
    #strike = 100.0
    p0 = 3.0
    d = [0.5, 0.3, 0.4]
    c = 0.01
    assert pnl_loop(payoff=0.0, S=S, d=d, p0=p0, c=c) == pytest.approx(-0.467)

def test_loop_flat_path_pays_only_costs():
    S = [100.0, 100.0, 100.0, 100.0]
    p0 = 3.0
    d = [0.5, 0.3, 0.4]
    c = 0.01
    assert pnl_loop(payoff=0.0, S=S, d=d, p0=p0, c=c) == pytest.approx(1.8)
    
def test_lengths_pb_raise():
    S = [100.0, 90.0, 95.0, 98.0]
    p0 = 3.0
    d = [0.5, 0.3]
    c = 0.01
    with pytest.raises(AssertionError):
        pnl_loop(payoff=0.0, S=S, d=d, p0=p0, c=c)