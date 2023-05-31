import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def loss_diff(pde, u, t, x):
  u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
  Du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
  #Hessian H[i][j] is derivative with respect to jth variable then with respect to ith variable or is it the other way around
  I_N = torch.eye(x.shape[-1], device=device)
  def get_vjp(v):
    return torch.autograd.grad(Du, x, grad_outputs=v.repeat(x.shape[0], 1), create_graph=True)
  D2u = torch.vmap(get_vjp)(I_N)[0]
  if len(x.shape) > 1:
    D2u = D2u.swapaxes(0, 1)
  A = D2u @ pde.sigma(t,x,u) @ pde.sigma(t, x, u).transpose(-2, -1)
  trace = torch.diagonal(A, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
  #trace = torch.vmap(torch.trace)(A)
  # in the code D2u[sample][i][j] is the derivative with respect to ith variable then jth variable
  f = pde.phi(t, x, u, Du) - torch.sum(Du * pde.mu(t, x, u, Du), dim=-1, keepdim=True) - 1/2 * trace
  return nn.MSELoss(reduction="sum")(u_t, f)

def loss_bc(pde, u, x):
  return nn.MSELoss(reduction="sum")(pde.g(x), u)