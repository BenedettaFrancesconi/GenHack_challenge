import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR  
from torch.utils.data import TensorDataset, DataLoader

from utils import AD_distance
from train_loops import train_epoch, eval_epoch
from architectures.mlp import MLP_Encoder, MLP_Decoder
from containers.encoder import Gaussian_Encoder
from containers.decoder import Gaussian_Decoder
from containers.vae import VAE


def load_data(data_path='data/df_train.csv', 
              n_val_years=9,
              batch_size=256):
    df = pd.read_csv(data_path)

    train = torch.tensor(df.iloc[:int(-365*n_val_years), 1:].to_numpy())
    val = torch.tensor(df.iloc[int(-365*n_val_years):, 1:].to_numpy())

    train = TensorDataset(train)
    val = TensorDataset(val)

    train_loader = DataLoader(train, batch_size = batch_size,
                                shuffle = True, drop_last = True)
    val_loader = DataLoader(val, batch_size = batch_size,
                            shuffle = False, drop_last = True)

    return train_loader, val_loader


def create_model(config, device):
    z_encoder = Gaussian_Encoder(MLP_Encoder(latent_dim=config['latent_dim'], n_cin=6),
                                 latent_dim=config['latent_dim'],
                                 loc=0.0, scale=1.0)
    decoder = Gaussian_Decoder(MLP_Decoder(latent_dim=config['latent_dim'], n_cout=6),
                                scale=1.0, device=device)
    return VAE(z_encoder, decoder)

    
def log(key, val):
    print(f"{key}: {val}")    


def main():
    config = {
        'wandb_on': False,
        'lr': 1e-4,
        'momentum': 0.9,
        'batch_size': 256,
        'max_epochs': 1000,
        'eval_epochs': 5,
        'eval_batches': 100,
        'n_val_years': 9,
        'seed': 1,
        'n_is_samples': 10,
        'latent_dim': 10,
        'checkpoint_path': 'checkpoints/vae_checkpoint.tar',
        'data_path': 'data/df_train.csv',
        }

    name = 'Basic_VAE'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader = load_data(data_path=config['data_path'],
                                         n_val_years=config['n_val_years'], 
                                         batch_size=config['batch_size'])

    model = create_model(config, device)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), 
                           lr=config['lr'],
                           momentum=config['momentum'])
    scheduler = StepLR(optimizer, step_size=1, gamma=1.0)

    for e in range(config['max_epochs']):
        log('Epoch', e)

        total_loss, total_neg_logpx_z, total_kl, num_batches = train_epoch(model, optimizer, 
                                                                     train_loader, log, e, 
                                                                     device=device,
                                                                     eval_batches=config['eval_batches'])

        log("Epoch Avg Loss", total_loss / num_batches)
        log("Epoch Avg -LogP(x|z)", total_neg_logpx_z / num_batches)
        log("Epoch Avg KL", total_kl / num_batches)
        scheduler.step()
        
        torch.save(model.state_dict(), config['checkpoint_path'])

        if e % config['eval_epochs'] == 0:
            total_loss, total_neg_logpx_z, total_kl, total_is_estimate, num_batches = eval_epoch(model, val_loader,
                                                                                                device=device,
                                                                                                n_is_samples=config['n_is_samples'])
                                                                                                                
            log("Val Avg Loss", total_loss / num_batches)
            log("Val Avg -LogP(x|z)", total_neg_logpx_z / num_batches)
            log("Val Avg KL", total_kl / num_batches)
            log("Val IS Estiamte", total_is_estimate / num_batches)

if __name__ == "__main__":
    main()