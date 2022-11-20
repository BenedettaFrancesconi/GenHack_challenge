import torch
import torch.nn as nn

def MLP_Encoder(s_dim, n_cin, n_hw):
    model = nn.Sequential(
                nn.Conv2d(n_cin, s_dim*3,
                    kernel_size=n_hw, stride=1, padding=0),
                nn.ReLU(True),
                nn.Conv2d(s_dim*3, s_dim*2,
                    kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Conv2d(s_dim*2, s_dim*2,
                    kernel_size=1, stride=1, padding=0))
    return model

def MLP_Decoder(s_dim, n_cout, n_hw):
    model = nn.Sequential(
                nn.ConvTranspose2d(s_dim, s_dim*2, 
                    kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.ConvTranspose2d(s_dim*2, s_dim*3, 
                    kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.ConvTranspose2d(s_dim*3, n_cout, 
                    kernel_size=n_hw, stride=1, padding=0),
                nn.Sigmoid())
    return model

class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to('cuda')  # sampling epsilon        
        z = mean + var*epsilon   # reparameterization trick
        return z
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var


class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var
