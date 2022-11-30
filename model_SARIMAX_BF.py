import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
#from sklearn.metrics import mean_squared_error
#from math import sqrt
#from tqdm import tqdm
import warnings
import itertools
#import matplotlib.pyplot as plt
import statsmodels.api as sm

# Defaults
#plt.rcParams['figure.figsize'] = (20.0, 10.0)
#plt.rcParams.update({'font.size': 12})
#plt.style.use('ggplot')

def SARIMA(lit,df):

  q = d = range(0, 2)
# Define the p parameters to take any value between 0 and 3
  p = range(0, 2)
  pdq = list(itertools.product(p, d, q))
  # Generate all different combinations of seasonal p, q and q triplets
  seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
  for name, value in lit.items():
    X = df[name]
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    warnings.filterwarnings("ignore") # specify to ignore warning messages

    AIC = []
    SARIMAX_model = []
    for param in pdq:
      for param_seasonal in seasonal_pdq:
        try:
          mod = sm.tsa.statespace.SARIMAX(train, order=param, 
                                          seasonal_order=param_seasonal, 
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
          results = mod.fit()
          print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
          AIC.append(results.aic)
          SARIMAX_model.append([param, param_seasonal])
        except:
            continue
    print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1]))


if __name__ == "__main__":
    df = pd.read_csv("data/df_train.csv")
    li = {'s1': df['s1'], 's2': df['s2'], 's3': df['s3'], 's4': df['s4'], 's5': df['s5'], 's6': df['s6']}
    SARIMA(li, df)
