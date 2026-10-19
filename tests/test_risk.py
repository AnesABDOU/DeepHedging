import torch
from deep_hedging.risk import entropic_risk, cvar_oce, cvar_empirical

def test_entropic_return_scalar():
    n = 10000
    x = torch.randn(n, dtype=torch.float64)
    assert entropic_risk(x, lam=0.5).shape == torch.Size([])
    
def test_entropic_cash_inv():
    torch.manual_seed(0)
    n=10000
    x = torch.randn(n, dtype=torch.float64)
    c, lam = 2.0, 0.5
    torch.testing.assert_close(entropic_risk(x + c, lam), entropic_risk(x, lam) - c)
    
def test_cvar_oce_return_scalar():
    n = 10000
    x = torch.randn(n, dtype=torch.float64)
    w = torch.tensor(0.3, dtype=torch.float64)
    assert cvar_oce(x, w, alpha=0.95).shape == torch.Size([])
    
def test_cvar_oce_cash_inv():
    torch.manual_seed(0)
    n=10000
    x = torch.randn(n, dtype=torch.float64)
    w = torch.tensor(0.3, dtype=torch.float64)
    c, alpha = 2.0, 0.95
    torch.testing.assert_close(cvar_oce(x + c, w - c, alpha), cvar_oce(x, w, alpha) - c)
    
def test_cvar_oce_cvar_emp():
    # at w = VaR, cvar_oce = cvar_emp
    torch.manual_seed(0)
    n=10000
    x = torch.randn(n, dtype=torch.float64)
    alpha = 0.95
    
    w, emp = cvar_empirical(x, alpha)
    oce = cvar_oce(x, w, alpha)
    
    torch.testing.assert_close(oce, emp) 