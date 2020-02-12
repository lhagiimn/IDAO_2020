import numpy as np
import pandas as pd
from joblib import load
import warnings
warnings.filterwarnings("ignore")

def prepare_timeseries_future(data, variables, lag, shift):
    dataX = []
    length = len(data) - lag
    X=np.asarray(data[variables])

    for i in range(shift, length):
        dataX.append(X[i-shift:(i + lag -shift +1)])

    dataX = np.array(dataX)

    if len(dataX.shape)>2:
        dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1]*dataX.shape[2]))

    dataX = pd.DataFrame(dataX)
    dataX['intercept'] = np.ones(len(dataX))

    return dataX


class Prediction():
    def __init__(self, test, train, lags, models, ids, trends, errors, shifts):
        self.test = test
        self.train = train
        self.lags = lags
        self.models = models
        self.ids = ids
        self.errors = errors
        self.trends = trends
        self.shifts = shifts

    def pred(self, sat_id, dep_vars, submission):

        shift_vars = ['x_sim', 'y_sim', 'z_sim', 'Vx_sim',
                      'Vy_sim', 'Vz_sim', 'distance_x', 'distance_v']

        sub_test = self.test.loc[self.test['sat_id'] == sat_id, :]
        sub_train = self.train.loc[self.train['sat_id'] == sat_id, :]

        epoch = sub_test['epoch'].diff()


        idx = epoch[epoch == 0].index
        dropped_test = sub_test.drop(idx, axis=0)

        for dep_var in dep_vars:

            if dep_var in ['x', 'y', 'z']:
                exog_var = ['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim', 'distance_x']
            else:
                exog_var = ['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim', 'distance_v']

            best_lag = self.lags['%s_%s' % (sat_id, dep_var)]
            shift = self.shifts['%s_%s' % (sat_id, dep_var)]
            sub_total = pd.concat([sub_train.tail((best_lag+self.ids[sat_id]+shift)), sub_test])
            sub_data = pd.DataFrame(data=np.asarray(sub_total[shift_vars]), columns=shift_vars)

            testX = prepare_timeseries_future(sub_data, exog_var, best_lag, shift)

            testX['trend'] = self.trends['%s_%s' % (sat_id, dep_var)][:len(testX)]


            coef_1 = self.models['%s_%s' % (sat_id, dep_var)]
            predY=np.asarray(testX.dot(np.transpose(coef_1)))
            error = self.errors['%s_%s' % (sat_id, dep_var)]

            submission.loc[dropped_test.index, dep_var] = predY[:len(dropped_test)]-error[:len(dropped_test)]
            submission.loc[idx, dep_var] = np.asarray(submission.loc[idx-1, dep_var])


        return submission


test = pd.read_csv('test.csv')

test['distance_x']= np.sqrt(test['x_sim']**2+test['y_sim']**2+test['z_sim']**2)
test['distance_v']= np.sqrt(test['Vx_sim']**2+test['Vy_sim']**2+test['Vz_sim']**2)
test['epoch'] = pd.DatetimeIndex(test['epoch'])
test['epoch']=(test['epoch'].dt.month*30*24 + test['epoch'].dt.day*24 +
               test['epoch'].dt.hour + test['epoch'].dt.minute/(60))

train = pd.read_csv('sub_train.csv')
train['distance_x']= np.sqrt(train['x_sim']**2+train['y_sim']**2+train['z_sim']**2)
train['distance_v']= np.sqrt(train['Vx_sim']**2+train['Vy_sim']**2+train['Vz_sim']**2)
train['epoch'] = pd.DatetimeIndex(train['epoch'])
train['epoch']=(train['epoch'].dt.month*30*24 + train['epoch'].dt.day*24 +
               train['epoch'].dt.hour + train['epoch'].dt.minute/(60))

lags = load('lags.joblib')
models = load('models.joblib')
errors = load('errors.joblib')
ids = load('ids.joblib')
trends = load('trends.joblib')
shifts = load('shifts.joblib')

sat_ids = test['sat_id'].unique()
submission = pd.DataFrame()
submission['id'] = test['id']

pred_class = Prediction(test, train, lags, models, ids, trends, errors, shifts)
for i in sat_ids:
    submission = pred_class.pred(i, ['x', 'y', 'z', 'Vx', 'Vy', 'Vz'],
                                    submission)

submission =submission[["id", "x", "y", "z", "Vx", "Vy", "Vz"]]
submission.to_csv("submission.csv", index=False)

