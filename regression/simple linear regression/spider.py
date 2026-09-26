import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\sahoo\Downloads\Salary_Data.csv")

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression   # LinearRegression is a ML Algorithm
regressor = LinearRegression()  # regressor is the ML Model
regressor.fit(x_train, y_train)    # The data here we used are x_trian, y_train


print(regressor)  # regressor is ML Model which consider linear regression algorithm

print(regressor.get_params())

y_pred = regressor.predict(x_test)  # Here we've given the x_test values to check the model to check the accuracy
print(y_pred)

comparision = pd.DataFrame({'Actual': y_test, 'Prediction': y_pred})
print(comparision) # Compare the values of y_test and y_pred, to check the accuracy

plt.scatter(x_test, y_test, color = 'Red')
plt.plot(x_train, regressor.predict(x_train), color = 'Blue')
plt.title('Salary of employee based on experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

y_12  = (m_slope*12)+c_intercept
print(y_12)
 
bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)

dataset.mean()

dataset['Salary'].mean()

dataset.median()

dataset['Salary'].median()

dataset['Salary'].mode()

dataset.var()

dataset['Salary'].var()

dataset.std()

dataset['Salary'].std()

from scipy.stats import variation
variation(dataset.values)

variation(dataset['Salary'])

dataset.corr()

dataset['Salary'].corr(dataset['YearsExperience'])

dataset.skew()
dataset['Salary'].skew()

dataset.sem()
dataset['Salary'].sem()

y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

r_square = 1 - SSR/SST
print(r_square)

import pickle
filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as linear_regression_model.pkl")

import os
print(os.getcwd())