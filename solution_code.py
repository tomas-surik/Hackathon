"""
This code is an updated version of a code I used to participate in a hackathon.
"""

# Load libraries
import pandas as pd
import numpy as np

from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier

# Load the external train dataset
data_train = pd.read_csv('train_dataset.csv')
# print(data_train.shape)

# Separate independent and dependent variables
index_test = data_train['id']
X_data = data_train.drop(['renewal', 'id'], axis=1)
y_data = data_train['renewal']

# Turn categorical variables into dummies
X_data = pd.get_dummies(X_data)

# Test some new features
X_data['relative_premium'] = X_data['Income']/X_data['premium']
X_data['total_paid'] = X_data['premium'] * X_data['no_of_premiums_paid']
X_data['no_late'] = X_data['Count_more_than_12_months_late'] + X_data['Count_6-12_months_late'] + X_data['Count_3-6_months_late']
X_data['log_income'] = np.log(X_data['Income'])
X_data['log_premium'] = np.log(X_data['premium'])
X_data['log_age_in_days'] = np.log(X_data['age_in_days'])

# Split external train data into model train and validation data 
X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.50, random_state=10)

# Originaly, Imputer was used, but it was removed from sklearn
imputer = SimpleImputer()
imputer = imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_valid = imputer.transform(X_valid)

# Consider scaling of the data, but GradientBoostingClassifier does not need it
scaler = MinMaxScaler()
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)

# Model configuration selected by trial and error to eliminate overfitting.
model = GradientBoostingClassifier(random_state=10, max_depth=6, subsample=0.8,
                                   n_estimators=75, min_samples_leaf = 800, min_samples_split = 4500)

model = model.fit(X_train, y_train)
y_train_predict = model.predict_proba(X_train)
print('Train Sample',roc_auc_score(y_train, y_train_predict[:, 1]))
y_valid_predict = model.predict_proba(X_valid)
print('Validation Sample',roc_auc_score(y_valid, y_valid_predict[:, 1]))

# Final model training on entire external train data
imputer = SimpleImputer()
imputer = imputer.fit(X_data)
X_data = imputer.transform(X_data)

# Same model configuration used
model = GradientBoostingClassifier(random_state=10, max_depth=6, subsample=0.8,
                                   n_estimators=75, min_samples_leaf = 800, min_samples_split = 4500)

model = model.fit(X_data, y_data)
train_benchmark = model.predict_proba(X_data)
print('Train All',roc_auc_score(y_data, train_benchmark[:, 1]))

# Load the external test dataset
data_test = pd.read_csv('test_dataset.csv')
# print(data_test.shape)
X_data_test = data_test.drop(['id'], axis=1)
X_data_test = pd.get_dummies(X_data_test)
test_index = data_test['id']

# Calculate new features  
X_data_test['relative_premium'] = X_data_test['Income']/X_data_test['premium']
X_data_test['total_paid'] = X_data_test['premium'] * X_data_test['no_of_premiums_paid']
X_data_test['no_late'] = X_data_test['Count_more_than_12_months_late'] + X_data_test['Count_6-12_months_late'] + X_data_test['Count_3-6_months_late']
X_data_test['log_income'] = np.log(X_data_test['Income'])
X_data_test['log_premium'] = np.log(X_data_test['premium'])
X_data_test['log_age_in_days'] = np.log(X_data_test['age_in_days'])

# Impute missing values
X_data_test = imputer.transform(X_data_test)
test_benchmark = model.predict_proba(X_data_test)
np.savetxt('benchmark_output.csv', test_benchmark, delimiter=",")

# Optimizer
benchmark = np.loadtxt('benchmark_output.csv', delimiter=',')[:, 1]
premium = data_test['premium'].values

inc_solution = []
for i in np.arange(len(benchmark)):
    bm = benchmark[i]
    prem = premium[i]
    # print(i, bm)
    scalar_fun = lambda inc: -((bm + bm * (20 * (1 - np.exp(-(10 * (1 - np.exp(-inc / 400))) / 5))) / 100) * prem - inc) # correct optimizer
    res = minimize_scalar(scalar_fun, options={'xtol': 10e-10, 'maxiter': 10000}, bounds=(0, 1000000))
    inc_solution.append(res.x)
    # print(i, res)

inc_solution = np.array(inc_solution)
np.savetxt('solution_output.csv', inc_solution, delimiter=",")

# Submission of results
renewal_prob = np.loadtxt('benchmark_output.csv', delimiter=',')[:, 1]
incentives = np.loadtxt('solution_output.csv', delimiter=',')
incentives = np.where(incentives < 0, 0, incentives)
submission = pd.DataFrame({'id': test_index, 'renewal': renewal_prob, 'incentives': incentives})
submission[['id', 'renewal', 'incentives']].to_csv('submission_output.csv', index=False)