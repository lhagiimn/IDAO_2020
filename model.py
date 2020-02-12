import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

from joblib import dump, load
import warnings
warnings.filterwarnings("ignore")


def smape_func(satellite_predicted_values, satellite_true_values):
    # the division, addition and subtraction are pointwise
    return np.mean(np.abs((satellite_predicted_values - satellite_true_values)
        / (np.abs(satellite_predicted_values) + np.abs(satellite_true_values))))


def prepare_timeseries(data, variables, dep_var, lag, shift):
    dataX, dataY = [], []
    length = len(data)-lag
    X=np.asarray(data[variables])
    Y=np.asarray(data[dep_var])

    for i in range(shift, length):
        dataX.append(X[i-shift:(i+lag-shift+1)])
        dataY.append(Y[i+lag])

    dataX = np.array(dataX)

    if len(dataX.shape)>2:
        dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1]*dataX.shape[2]))

    dataY = np.expand_dims(np.array(dataY), axis=1)
    dataX = pd.DataFrame(dataX)
    dataX['intercept'] = np.ones(len(dataX))

    return dataX, dataY

def regression(train, exog_var, max_lag, dep_var, max_shift):
    best_smape = np.inf
    best_lag = 0
    best_shift =0
    if len(train)<200:
        max_shift=12

    for lag in range(0, max_lag):
        for shift in range(max_shift):
            trainX, trainY = prepare_timeseries(train, exog_var, dep_var, lag, shift)
            trainX['trend'] = np.asarray(trend[(lag+shift):len(train)])
            reg = LinearRegression(fit_intercept=False, n_jobs=-1).fit(trainX, trainY)
            pred_Y = reg.predict(trainX)

            smape_reg = smape_func(pred_Y, trainY)

            if smape_reg <= 0.05:
                best_lag = lag
                best_shift = shift
                return best_lag, best_shift

            elif best_smape > smape_reg:
                best_smape = smape_reg
                best_lag = lag
                best_shift = shift

    return best_lag, best_shift


data = pd.read_csv('train.csv')
data['distance_x']= np.sqrt(data['x_sim']**2+data['y_sim']**2+data['z_sim']**2)
data['distance_v']= np.sqrt(data['Vx_sim']**2+data['Vy_sim']**2+data['Vz_sim']**2)
data['epoch'] = pd.DatetimeIndex(data['epoch'])
data['epoch']=(data['epoch'].dt.month*30 + data['epoch'].dt.day +
               data['epoch'].dt.hour/24 + data['epoch'].dt.minute/(60*24))


dep_vars=['x', 'y', 'z', 'Vx', 'Vy', 'Vz']
shift_vars = ['x_sim', 'y_sim', 'z_sim', 'Vx_sim',
             'Vy_sim', 'Vz_sim', 'distance_x', 'distance_v']

models = {}
errors = {}
lags = {}
ids = {}
trends={}
shifts={}

k=1
sat_id = data['sat_id'].unique()


for i in sat_id:
    sub = data.loc[data['sat_id'] == i, :]
    epoch = sub['epoch'].diff()
    idx=epoch[epoch==0].index

    #print(epoch.groupby(epoch).count())
    #plt.hist(np.asarray(epoch), 5)
    #plt.show()

    ids[i]=(len(idx))
    train = sub.drop(idx, axis=0)
    train[shift_vars] = np.asarray(sub[shift_vars])[:len(train)]
    trend = np.arange(1, 2*int(len(train)/24)+10)
    trend = np.repeat(trend, 24)


    # for i in range(int(len(train)/24)):
    #         plt.plot(np.asarray(train['y'])[i*24:i*24+24], label='x')
    #         plt.plot(np.asarray(train['y_sim'])[i*24:i*24+24], label='x_sim')
    #         plt.legend()
    #         plt.title('%s' %i)
    #         plt.show()

    # exit('bye')
    # plt.savefig('figures/figure_%s.png' % (i))
    # plt.close()


    for dep_var in dep_vars:

        if dep_var in ['x', 'y', 'z']:
            exog_var = ['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim', 'distance_x']
        else:
            exog_var = ['Vx_sim', 'Vy_sim', 'Vz_sim', 'x_sim', 'y_sim', 'z_sim', 'distance_v']

        best_lag, best_shift = regression(train=train, max_lag=5, dep_var=dep_var, exog_var=exog_var, max_shift=24)

        trainX, trainY = prepare_timeseries(train, exog_var, dep_var, best_lag, best_shift)
        trainX['trend'] = trend[best_lag+best_shift:len(train)]
        reg = LinearRegression(fit_intercept=False, n_jobs=-1).fit(trainX, trainY)
        pred_Y = reg.predict(trainX)

        smape = smape_func(pred_Y, trainY)

        print('%s and %s: ' % (i, dep_var), smape, 'best_lag:', best_lag, 'shift:', best_shift)

        lags['%s_%s' % (i, dep_var)] = best_lag
        shifts['%s_%s' % (i, dep_var)] = best_shift
        models['%s_%s' % (i, dep_var)] = reg.coef_

        error = pred_Y-trainY

        X1 = error[:-48]
        X2 = error[24:(len(error)-24)]
        X=np.concatenate((X1, X2), axis=1)
        Y=error[48:]

        reg2 = LinearRegression(fit_intercept=False, n_jobs=-1).fit(X, Y)
        error_Y=reg2.predict(X)
        # plt.plot(error_Y)
        # plt.plot(Y)
        # plt.show()

        X_test = np.concatenate((error[24:(len(error)-24)], error[48:]), axis=1)
        for r in range(int(len(train)/24)+5):
            error_Y = reg2.predict(X_test)
            Y=np.concatenate((Y, error_Y[-24:]), axis=0)
            X_test = np.concatenate((Y[24:len(Y) - 24], Y[48:]), axis=1)

        errors['%s_%s' % (i, dep_var)] = Y[len(X):]
        trends['%s_%s' % (i, dep_var)] = trend[len(train):]
        # plt.plot(Y)
        # plt.plot(error)
        # plt.show()


dump(models, 'models.joblib')
dump(errors, 'errors.joblib')
dump(lags, 'lags.joblib')
dump(ids, 'ids.joblib')
dump(trends, 'trends.joblib')
dump(shifts, 'shifts.joblib')

