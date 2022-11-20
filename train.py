from utils import AD_distance
import pandas as pd
from normalizing_flow import norm_flow
from torch.utils.data import TensorDataset, DataLoader  
import torch

import numpy as np

def load_data(file_path='data/df_train.csv', n_val_years=9):
    df = pd.read_csv('../data/df_train.csv')

    train = df.iloc[:-365*n_val_years, 1:].to_numpy()
    val = df.iloc[-365*n_val_years:, 1:].to_numpy()

    train = TensorDataset(train)
    val = TensorDataset(val)

    return train, val


def train_model(train, val,  
                epochs = 1000,
                n_val_batches = 100,
                n_val_epochs = 10,
                batch_size = 256):
    model = norm_flow()
    train_loader = DataLoader(train, batch_size = batch_size,
                                shuffle = True, drop_last = True)
    val_loader = DataLoader(val, batch_size = batch_size,
                            shuffle = True, drop_last = True)
    for e in range(epochs):

        # Train
        for b_idx, train_batch in enumerate(train_loader):
            loss = model.update_batch(train_batch)

            if b_idx % n_val_batches == 0: 
                print("train loss: {}-{}, {}".format(e, b_idx, loss))
    
        if e % n_val_epochs == 0:
            total_val_loss = 0
            with torch.no_grad():
                for b_idx, val_batch in enumerate(val_loader):
                    loss = model.eval_batch(val_batch)
                    total_val_loss += loss

                    if b_idx % n_val_batches == 0: 
                        print("val loss: {}-{}, {}".format(e, b_idx, loss))
                print("total val loss: {}, {}".format(e, total_val_loss/len(val_loader)))
        









def save_model(model):
    pass




def main():
    train, val = load_data()
    model = train_model(train, val)
    save_model(model)


if __name__ == "__main__":
    main()
    
    
