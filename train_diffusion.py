import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR  
from torch.utils.data import TensorDataset, DataLoader

from diffusion_train_loops import train_epoch_diffusion, eval_epoch_diffusion
from diffusion import GaussianDiffusion1D, Unet1D
# from architectures.mlp import Diffusion_MLP

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
    # model = Diffusion_MLP(latent_dim=config['latent_dim'], n_cin=6)

    model = Unet1D(
                    seq_length = 1,
                    dim = 32,
                    dim_mults = (1, 2, 4, 8),
                    channels = config['input_dim']
                    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 1,
        timesteps = config['timesteps'],   
        loss_type = config['loss_type']    
    )

    return diffusion

    
def log(key, val):
    print(f"{key}: {val}")    


def main():
    config = {
        'wandb_on': False,
        'lr': 1e-3,
        'adam_betas': (0.9, 0.99),
        'momentum': 0.9,
        'batch_size': 256,
        'max_epochs': 1000,
        'eval_epochs': 5,
        'eval_batches': 100,
        'n_val_years': 9,
        'seed': 1,
        'latent_dim': 10,
        'input_dim': 6,
        'timesteps': 1000,
        'loss_type': 'l1',     # L1 or L2
        'checkpoint_path': 'checkpoints/diffusion_100e_lr1e-3_val9_d10.tar',
        'data_path': 'data/df_train.csv',
        }

    name = 'Basic_Diffusion'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader = load_data(data_path=config['data_path'],
                                         n_val_years=config['n_val_years'], 
                                         batch_size=config['batch_size'])

    model = create_model(config, device)
    model.to(device)

    # load_checkpoint_path = config['checkpoint_path']
    # model.load_state_dict(torch.load(load_checkpoint_path))

    optimizer = optim.Adam(model.parameters(), 
                            lr=config['lr'],
                            betas=config['adam_betas'])

    scheduler = StepLR(optimizer, step_size=1, gamma=1.0)

    for e in range(config['max_epochs']):
        log('Epoch', e)

        total_loss,  num_batches = train_epoch_diffusion(model, optimizer, 
                                                                     train_loader, log, e, 
                                                                     device=device,
                                                                     eval_batches=config['eval_batches'])

        log("Epoch Avg Loss", total_loss / num_batches)
        scheduler.step()
        
        torch.save(model.state_dict(), config['checkpoint_path'])

        if e % config['eval_epochs'] == 0:
            total_loss, num_batches = eval_epoch_diffusion(model, val_loader, device=device)
                                                                                                                
            log("Val Avg Loss", total_loss / num_batches)

if __name__ == "__main__":
    main()