import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NAIS_Net_Untied(nn.Module):
    # x(k+1) = x(k) + hσ(A(k)x(k) + B(k)u + b(k))
    # x(0) = 0
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size, activation, epsilon=0.01, h=1):
        super().__init__()
        self.epsilon = epsilon
        self.h = h
        self.hidden_size = hidden_size
        self.input_layer = nn.Linear(input_size, hidden_size, bias=True)
        # see forward for explanation on why we don't need A(0)
        self.activation = activation
        self.hidden_layers_state = nn.ModuleList()
        self.hidden_layers_input = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers_state.append(nn.Linear(hidden_size, hidden_size, bias=False))
            self.hidden_layers_input.append(nn.Linear(input_size, hidden_size, bias=True))
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def state_forward(self, x, layer):
        R = layer.weight
        RtR = R.T @ R
        # reprojection
        norm = torch.linalg.matrix_norm(RtR)
        delta = 1 - 2 * self.epsilon
        if norm > delta:
            RtR = delta * RtR / norm
        A = - RtR - self.epsilon * torch.eye(RtR.shape[0], device=device)
        return F.linear(x, A, None) 

    def forward(self, u):
        #x(0)=0
        #x(1)=x(0) + hσ(A(0)x(0) + B(0)u + b(0))=hσ(B(0)u + b(0))
        # so we don't need A(0) as A(0)x(0) = 0
        x = self.h * self.activation(self.input_layer(u))
        for i in range(len(self.hidden_layers_state)):
            x = x + self.h * self.activation(self.state_forward(x, self.hidden_layers_state[i]) + self.hidden_layers_input[i](u))
        return self.output_layer(x)
    
class FC_Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size, activation):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size, bias=True)
        self.activation = activation
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

class Res_Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size, activation):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size, bias=True)
        self.activation = activation
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        # residual connections between hidden layers
        for layer in self.hidden_layers:
            x = x + self.activation(layer(x))
        return self.output_layer(x)
    
class DGM_Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size, activation):
        super().__init__()
        # DGM network
        self.L = num_hidden_layers - 1
        self.activation = activation
        self.S_1 = nn.Linear(input_size, hidden_size, bias=True)
        self.Z_U = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=False) for i in range(self.L)])
        self.Z_W = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=True) for i in range(self.L)])
        self.G_U = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=False) for i in range(self.L)])
        self.G_W = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=True) for i in range(self.L)])
        self.R_U = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=False) for i in range(self.L)])
        self.R_W = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=True) for i in range(self.L)])
        self.H_U = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=False) for i in range(self.L)])
        self.H_W = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=True) for i in range(self.L)])
        self.output_layer = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        # computing S_1
        S = self.S_1(x)
        for i in range(self.L):
            # computing Z_l, G_l, R_l, H_l
            Z = self.activation(self.Z_U[i](x) + self.Z_W[i](S))
            G = self.activation(self.G_U[i](x) + self.G_W[i](S))
            R = self.activation(self.R_U[i](x) + self.R_W[i](S))
            H = self.activation(self.H_U[i](x) + self.H_W[i](S * R))
            # computing S_(l+1)
            S = (1 - G) * H + Z * S
        return self.output_layer(S)

