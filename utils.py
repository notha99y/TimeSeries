'''
Helper functions to do Time Series Analysis
'''

from statsmodels.tsa.stattools import acf
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def read_oil_price():
    _start_date = 'Jan 1986'
    _end_date = 'Feb 2006'

    start_date = datetime.strptime(_start_date, '%b %Y').date()
    end_date = datetime.strptime(_end_date, '%b %Y').date()

    dataset = pd.read_csv('oilprice.csv')
    dataset['Time'] = pd.date_range(start=start_date, end=end_date, freq='M')
    dataset = dataset.set_index('Time')
    return dataset


def components_decomposition(time_series):
    decomposition = seasonal_decompose(time_series)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(time_series, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    return decomposition


def test_stationary(time_series):
    print('Performing Moving average test')
    window = 12
    rolmean = time_series.rolling(window=window).mean()
    rolstd = time_series.rolling(window=window).std()

    plt.plot(time_series, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Stdev')
    plt.show(block=False)

    from statsmodels.tsa.stattools import adfuller
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(time_series['oil.price'], autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=[
                         'Test Statistics', 'p-value', '#Lags Used', '#Oberservations Used'])
    for k, v in dftest[4].items():
        dfoutput['Critical Value (%s)' % k] = v
    print(dfoutput)


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0, 1]   # corr
    mins = np.amin(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(forecast-actual)[1]                      # ACF1
    return({'mape': mape, 'me': me, 'mae': mae,
            'mpe': mpe, 'rmse': rmse, 'acf1': acf1,
            'corr': corr, 'minmax': minmax})


def time_series_regression(time_series, n_order=4):
    preds = []
    for order in range(n_order + 1)[1:]:
        print('Training order: ', order)
        poly = PolynomialFeatures(order)
        x_transform = poly.fit_transform(
            np.array(time_series.index).reshape((-1, 1)))
        lm2 = LinearRegression()
        lm2.fit(x_transform, time_series['oil.price'])
        pred = lm2.predict(x_transform)
        pred = pd.DataFrame(data=pred, index=time_series.index,
                            columns=['Predictions'])
        preds.append(pred)
    return preds


def time_series_regression_w_testing(train, test, n_order=4):
    preds = []
    for order in range(n_order + 1)[1:]:
        print('Training order: ', order)
        poly = PolynomialFeatures(order)
        x_transform = poly.fit_transform(
            np.array(train.index).reshape((-1, 1)))
        lm2 = LinearRegression()
        lm2.fit(x_transform, train['oil.price'])

        test_transform = poly.fit_transform(
            np.array(test.index).reshape((-1, 1)))
        pred = lm2.predict(test_transform)
        pred = pd.DataFrame(data=pred, index=test.index,
                            columns=['Predictions'])
        preds.append(pred)
    return preds
