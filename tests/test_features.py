import torch
from deep_hedging.features import make_features

def test_features_shape_and_values():
    N, n, T, K = 8, 4, 1.0, 100.0
    t = torch.linspace(0.0, T, n+1)
    S = 100.0 * torch.ones(N, n+1, 1)
    f = make_features(S, t, T, K)
    assert f.shape == (N, n, 2)
    assert torch.allclose(f[:,:,0], torch.zeros(N,n))
    assert torch.allclose(f[0,:,1], T - t[:-1])