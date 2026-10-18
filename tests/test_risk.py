import torch
from deep_hedging.risk import entropic_risk

def test_entropic_return_scalar():
    n = 10000
    x = torch.randn(n)
    assert entropic_risk(x, lam=0.5).shape == torch.Size([])
    
def test_entropic_cash_inv():
    torch.manual_seed(0)
    x = torch.randn(10000, dtype=torch.float64)
    c, lam = 2.0, 0.5
    torch.testing.assert_close(entropic_risk(x + c, lam), entropic_risk(x, lam) - c)