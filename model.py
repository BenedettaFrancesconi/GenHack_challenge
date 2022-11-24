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
import os
import torch
from architectures.mlp import MLP_Encoder, MLP_Decoder
from containers.encoder import Gaussian_Encoder
from containers.decoder import Gaussian_Decoder
from containers.vae import VAE

default_config = {
    'latent_dim': 10,
    'checkpoint_path': 'parameters/vae_100e_lr1e-3_val9_d10.tar',
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    model.load_state_dict(torch.load(load_checkpoint_path))

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
    latent_variable = torch.tensor(noise[:, :default_config['latent_dim']]).to(device)  # use the first 10 dimensions of the noise

    model = load_model()

    sample = model.decoder.only_decode(latent_variable)

    return sample



