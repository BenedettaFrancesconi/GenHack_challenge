from torch import nn

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