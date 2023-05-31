import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FBSDE:
    def __init__(self, mu, sigma, phi, T, g):
        self.mu = mu
        self.sigma = sigma
        self.phi = phi
        self.T = T
        self.g = g
def BS_Barenblatt_sigma(volatility, t, x, y):
    return volatility * torch.diag_embed(x).to(device)
def BS_Barenblatt_mu(t, x, y, z):
    return torch.zeros(x.shape, device=device)
def BS_Barenblatt_phi(r, t, x, y, z):
    return r * (y - torch.sum(x * z, dim=-1, keepdim=True))
def BS_Barenblatt_g(x):
    return torch.sum(x ** 2, -1, keepdim=True)
    # return torch.linalg.vector_norm(x, dim=-1)**2

class BS_Barenblatt(FBSDE):
    def __init__(self, volatility, r, T):
        self.volatility = volatility
        self.r = r
        sigma = lambda t, x, y: BS_Barenblatt_sigma(volatility, t, x, y)
        mu = BS_Barenblatt_mu
        phi = lambda t, x, y, z: BS_Barenblatt_phi(r, t, x, y, z)
        g = BS_Barenblatt_g
        super().__init__(mu, sigma, phi, T, g)
    
    def exact_solution(self, t, x):
        return torch.exp(torch.tensor((self.r + self.volatility**2) * (self.T - t))) * torch.linalg.vector_norm(x, dim=-1)**2