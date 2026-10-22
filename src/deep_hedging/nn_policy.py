import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_ffn(n_inputs, n_outputs=1, n_hidden=16, n_layers=2):
    """
    n_inputs = 3 : log moneyness, tau, and delta{k-1}, we give the time to maturity as an input to the NN deliberatly
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
    concats delta to features
    """