import torch
import torch.nn as nn

class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, in_size, out_size, hid_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hid_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hid_size, hid_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hid_size, hid_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hid_size, out_size),
        )
    def forward(self, x):
        return self.net(x)


class AffineHalfFlow(nn.Module):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the 
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    """
    def __init__(self, dim, parity, net_class=MLP, hid_size=24, device='cuda'):
        super().__init__()
        self.dim = dim
        self.parity = parity
        self.s_cond = net_class(self.dim // 2, self.dim // 2, hid_size).to(device)
        self.t_cond = net_class(self.dim // 2, self.dim // 2, hid_size).to(device)
        
    def forward(self, x):
        x0, x1 = x.chunk(2, dim=1) 
        if self.parity:
            x0, x1 = x1, x0
        s = self.s_cond(x0)
        t = self.t_cond(x0)
        z0 = x0      # untouched half
        z1 = torch.exp(s) * x1 + t     # transform this half as a function of the other
        #if torch.isnan(z1).any():
        #    raise RuntimeError('Scale factor has NaN entries')
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z):
        z0, z1 = z.chunk(2, dim=1) 
        if self.parity:
            z0, z1 = z1, z0
        s = self.s_cond(z0)
        t = self.t_cond(z0)
        x0 = z0     # this was the same
        x1 = (z1 - t) * torch.exp(-s)     # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows, device='cuda'):
        super().__init__()
        self.device=device
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m).to(self.device)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m).to(self.device)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det

  
class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """
    
    def __init__(self, prior, flows, device='cuda'):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows, device)
    
    def forward(self, x):
        zs, log_det = self.flow.forward(x)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det
    
    def sample(self, num_samples):
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.backward(z)
        return xs