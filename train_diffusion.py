import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR  
from torch.utils.data import TensorDataset, DataLoader
from utils import AD_distance

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

# BENEDETTA KICKS IN 
li = {'s1': df['s1'], 's2': df['s2'], 's3': df['s3'], 's4': df['s4'], 's5': df['s5'], 's6': df['s6']}
trends = pd.DataFrame()
for name, value in li.items():
    
    series = df[name]
    X = [i for i in range(0, len(series))]
    X = np.reshape(X, (len(X), 1))
    y = series.values
    model = LinearRegression()
    model.fit(X, y)
    # calculate trend
    trends[name] = model.predict(X)
    # detrend
    #detrended = [y[i]-trend[i] for i in range(0, len(series))] maybe I'll use it, I don't know
    
#

def create_model(config, device):
    # model = Diffusion_MLP(latent_dim=config['latent_dim'], n_cin=6)

    model = Unet1D(
                    seq_length = 1,
                    dim = config['latent_dim'],
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
        'lr': 1e-1,
        'adam_betas': (0.9, 0.99),
        'momentum': 0.9,
        'batch_size': 256,
        'max_epochs': 100,
        'AD_epochs': 100,
        'n_AD_years': 9,
        'eval_epochs': 5,
        'eval_batches': 100,
        'n_val_years': 9,
        'seed': 1,
        'latent_dim': 8, # must be multiple of 8
        'input_dim': 6,
        'timesteps': 100,
        'loss_type': 'l1',     # L1 or L2
        'checkpoint_path': 'checkpoints/diffusion_1000e_lr1e-3_val9_d10.tar',
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
        
        #torch.save(model.state_dict(), config['checkpoint_path'])

        if e % config['eval_epochs'] == 0:
            total_loss, num_batches = eval_epoch_diffusion(model, val_loader, device=device)
                                                                                                                
            log("Val Avg Loss", total_loss / num_batches)


        if (e + 1) % config['AD_epochs'] == 0 or (e + 1) == config['max_epochs']:
            # Computing AD Metric
            val_data = val_loader.dataset.tensors[0][:config['n_AD_years']*365] # (n_test, 6)
            noise = np.random.normal(0,1, size = (val_data.shape[0], config['latent_dim']))
            noise = torch.tensor(noise).float()
            sample = model.sample(noise.shape[0])

            sample = sample.squeeze().detach().cpu().numpy().astype('float32') # (n_test, 6)

            ## TODO: Add output of ARIMA to sample here
            # How do I add the values from the dataframe trends in line 35? Its shape is (9618, 6)
            # sample = sample + (ARIMA(2006) - mean(ARIMA(1981 to 2002)))

            w = AD_distance(sample.reshape(-1, 6).transpose(), val_data.cpu().numpy().astype('float32').transpose())
    
            print("AD Distance per Station:", w)
            print("AD Distance Mean:", w.mean())



if __name__ == "__main__":
    main()