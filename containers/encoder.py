import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal

class Encoder(nn.Module):
    """
    Abstract encoder class
    """
    def __init__(self, model, prior, latent_dim):
        super(Encoder, self).__init__()
        self.model = model
        self.prior = prior
        self.latent_dim = latent_dim

    def forward(self, x):
        raise NotImplementedError

    def sample(self, x, n_samples, cap_dim):
        return self.prior.sample(torch.Size([n_samples, self.latent_dim])).to(x.device)


class Gaussian_Encoder(Encoder):
    """
    Implementation of Gaussian Encoder (approximate posterior)
    """
    def __init__(self, model, latent_dim, loc=0.0, scale=1.0):
        super(Gaussian_Encoder, self).__init__(model, Normal(loc=loc, scale=scale), latent_dim)

    def forward(self, x):
        mu_logvar = self.model(x)
        mu = mu_logvar[:, :self.latent_dim]
        logvar = mu_logvar[:, self.latent_dim:]
        std = torch.exp(logvar / 2.0)

        q = Normal(mu, std)
        z = q.rsample()

        log_q_z = q.log_prob(z)
        log_p_z = self.prior.log_prob(z)

        kl = kl_divergence(q, self.prior)
        return z, kl, log_q_z, log_p_z