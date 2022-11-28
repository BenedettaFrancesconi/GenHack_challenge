##############################################
# YOU MUST NOT MODIFY THIS FILE
##############################################
from model import generative_model
from utils import AD_distance
import pandas as pd
import torch
import numpy as np
import logging

#df = pd.read_csv('C:/Users/bened/Documents/GitHub/TEAM_ROCKET/datadf_train.csv')


logging.basicConfig(filename="check.log", level=logging.DEBUG, 
                    format="%(asctime)s:%(levelname)s: %(message)s", 
                    filemode='w')

def simulate(noise, df):
    """
    simulation of your Generative Model

    Parameters
    ----------
    model : object
        generator function
    noise : ndarray
        input of the generative model
    """
    train = np.array(df.iloc[:int(-365*9), 1:])
    val = np.array(df.iloc[int(-365*9):, 1:])

    try:
        output = generative_model(noise)
        message = "Successful simulation" 
        assert output.shape == (noise.shape[0], 6), "Shape error, it must be (n_data, 6). Please verify the shape of the output."
        
        # write the output
        #np.save("output.npy", output)
        dist = AD_distance(val, output)

    except Exception as e:
        message = e
                
    finally:
        logging.debug(message)

    return dist


    
if __name__ == "__main__":
    z = np.random.normal(0,1, size = (10,50))
    noise = z
    df = np.load("data/df_train.csv")
    simulate(noise,df)
    
    
