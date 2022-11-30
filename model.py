#############################################################################
# YOUR GENERATIVE MODEL
# ---------------------
# Should be implemented in the 'generative_model' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# You can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
#
# See below an example of a generative model
# G_\theta(Z) = np.max(0, \theta.Z)
############################################################################

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def MLP_Encoder(latent_dim, n_cin):
    model = nn.Sequential(
                nn.Linear(n_cin, latent_dim*3),
                nn.ReLU(True),
                nn.Linear(latent_dim*3, latent_dim*2),
                nn.ReLU(True),
                nn.Linear(latent_dim*2, latent_dim*2))
    return model

def MLP_Decoder(latent_dim, n_cout):
    model = nn.Sequential(
                nn.Linear(latent_dim, latent_dim*2),
                nn.ReLU(True),
                nn.Linear(latent_dim*2, latent_dim*3),
                nn.ReLU(True),
                nn.Linear(latent_dim*3, n_cout))
    return model

class Decoder(nn.Module):
    def __init__(self, model):
        super(Decoder, self).__init__()
        self.model = model

    def forward(self, x):
        raise NotImplementedError

class Bernoulli_Decoder(Decoder):
    def __init__(self, model):
        super(Bernoulli_Decoder, self).__init__(model)

    def forward(self, z, x):
        probs_x = torch.clamp(self.model(z), 0, 1)
        p = Bernoulli(probs=probs_x)
        neg_logpx_z = -1 * p.log_prob(x)

        return probs_x, neg_logpx_z

    def only_decode(self, z):
        probs_x = torch.clamp(self.model(z), 0, 1)
        return probs_x

class Gaussian_Decoder(Decoder):
    def __init__(self, model, scale=1.0, device='cuda'):
        super(Gaussian_Decoder, self).__init__(model)
        
        self.scale = torch.tensor([scale], device=device)

    def forward(self, z, x):
        mu_x = self.model(z)
        p = Normal(loc=mu_x, scale=self.scale)
        neg_logpx_z = -1 * p.log_prob(x)

        return mu_x, neg_logpx_z

    def only_decode(self, z):
        mu_x = self.model(z)
        return mu_x

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

class VAE(torch.nn.Module):
    def __init__(self, z_encoder, decoder):
        super(VAE, self).__init__()
        self.z_encoder = z_encoder
        self.decoder = decoder

    def forward(self, x):
        z, kl_z, _, _ = self.z_encoder(x)
        probs_x, neg_logpx_z = self.decoder(z, x)

        return z, probs_x, kl_z, neg_logpx_z

    def get_IS_estimate(self, x, n_samples=100):
        log_likelihoods = []

        for n in range(n_samples):
            z, kl_z, log_q_z, log_p_z = self.z_encoder(x)
            probs_x, neg_logpx_z = self.decoder(z, x)
            ll = (-1 * neg_logpx_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  + log_p_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  - log_q_z.flatten(start_dim=1).sum(-1, keepdim=True))
            log_likelihoods.append(ll)
        ll = torch.cat(log_likelihoods, dim=-1)
        is_estimate = torch.logsumexp(ll, -1)
        return is_estimate


default_config = {
    'latent_dim': 10,
    'checkpoint_path': 'parameters/vae_100e_lr1e-3_val9_d10.tar',
}


def create_model(config):
    z_encoder = Gaussian_Encoder(MLP_Encoder(latent_dim=config['latent_dim'], n_cin=6),
                                 latent_dim=config['latent_dim'],
                                 loc=0.0, scale=1.0)
    decoder = Gaussian_Decoder(MLP_Decoder(latent_dim=config['latent_dim'], n_cout=6),
                                scale=1.0, device=device)
    return VAE(z_encoder, decoder)


def load_model():
    model = create_model(default_config)
    model.to(device)

    load_checkpoint_path = default_config['checkpoint_path']
    model.load_state_dict(torch.load(load_checkpoint_path, map_location=torch.device(device)))

    return model

# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
def generative_model(noise):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim)
        input noise of the generative model
    """
    # See below an example
    # ---------------------
    latent_variable = torch.tensor(noise[:, :default_config['latent_dim']]).float().to(device)  # use the first 10 dimensions of the noise

    model = load_model()

    sample = model.decoder.only_decode(latent_variable)

    return sample.detach().cpu().numpy()



