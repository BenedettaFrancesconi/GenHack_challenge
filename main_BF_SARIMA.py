from model import generative_model
from utils import AD_distance
from model_SARIMAX_BF import SARIMA
import pandas as pd
import torch
import numpy as np
import logging

#df = pd.read_csv('C:/Users/bened/Documents/GitHub/TEAM_ROCKET/datadf_train.csv')


logging.basicConfig(filename="check_BF.log", level=logging.DEBUG, 
                    format="%(asctime)s:%(levelname)s: %(message)s", 
                    filemode='w')

def simulate(df):
    """
    simulation of your Generative Model

    Parameters
    ----------
    model : object
        generator function
    """
    #train = np.array(df.iloc[:int(-365*9), 1:])
    val = np.array(df.iloc[int(-365*9):, 1:]).transpose().astype('float32')
    #noise = torch.tensor(noise).float()

    z = np.random.normal(0,1, size = (val.shape[1],50))
    noise = torch.tensor(z).float()

    lit = {'s1': df['s1'], 's2': df['s2'], 's3': df['s3'], 's4': df['s4'], 's5': df['s5'], 's6': df['s6']}

    output = np.array(generative_model(noise)).transpose().astype('float32')
    constant = np.array(SARIMA(lit, df))
    #message = "Successful simulation" 
    #assert output.shape == (noise.shape[0], 6).transpose().astype('float32'), "Shape error, it must be (n_data, 6). Please verify the shape of the output."
        
    """
    AD_distance:
    real: np.array(6, n_test)
    fake: np.array(6, n_test)
    """

    w = AD_distance(output, val)
    
    print("AD Distance per Station:", w)
    print("AD Distance Mean:", w.mean())

    return w


    
if __name__ == "__main__":
    df = pd.read_csv("data/df_train.csv")
    simulate(df)
    
    
