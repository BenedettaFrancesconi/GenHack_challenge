import os
import torch
import torchvision

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