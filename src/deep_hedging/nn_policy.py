import torch
import torch.nn as nn
from deep_hedging.features import make_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_nn(n_inputs, n_outputs=1, n_hidden=16, n_layers=2):
    """
    n_inputs = 3 : log moneyness, tau, and delta{k-1}, we give the time to maturity as an input to the feedforward NN deliberatly
    """
    layers = [nn.Linear(n_inputs, n_hidden), nn.ReLU()]
    
    for _ in range (n_layers-1) :
        layers.extend([nn.Linear(n_hidden, n_hidden), nn.ReLU()])
        
    # no ReLU on the last layer because it can prevent delta < 0
    layers.append(nn.Linear(n_hidden, n_outputs))
    
    ffn = nn.Sequential(*layers)
    
    return ffn
    
    
       

def nn_policy(net: torch.nn.Sequential, features: torch.Tensor):
    """
    delta_{-1} = delta_{n} = 0
    concats delta to features before sending them to the feedforward NN
    """
    paths = features.shape[0]
    timesteps = features.shape[1]
    instruments = net[-1].out_features
    
    prev_holding = torch.zeros(paths, instruments, dtype=features.dtype)
    deltas = []
    
    for k in range(timesteps):
        prev_holding = net(torch.cat([features[:,k,:], prev_holding], dim=-1))
        
        deltas.append(prev_holding)
        
    return torch.stack(deltas, dim=1)
    