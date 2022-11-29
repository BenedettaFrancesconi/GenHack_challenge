import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm


def ARIMA(df, ar, ma):
    i = ['s1', 's2', 's3', 's4', 's5', 's6']
    dict_autofit = {}
    for i in(li):
        for ar_lag in range(ar):
            for ma_lag in range(ma):  
                X = df[i]
                size = int(len(X) * 0.66)
                train, test = X[0:size], X[size:len(X)]
                history = [x for x in train]
                predictions = list()
                for t in tqdm(range(len(test))):
                    model = ARIMA(history, order=(ar_lag,1,ma_lag))
                    model_fit = model.fit()
                    output = model_fit.forecast()
                    yhat = output[0]
                    predictions.append(yhat)
                    obs = test.iloc[t]
                    history.append(obs)
                rmse = sqrt(mean_squared_error(test, predictions))
                list_autofit = np.append([li,ar_lag, ma_lag, rmse])
            dict_autofit[name] = pd.DataFrame(list_autofit, columns = ["li", "ar_lag", "ma_lag", "rmse"])
    return rmse