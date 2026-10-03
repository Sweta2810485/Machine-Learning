import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\sahoo\Downloads\emp_sal.csv")

X = dataset.iloc[:, 1:2].values 
y = dataset.iloc[:, 2].values 

# liner regression model 

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)  

# linear regression visualizaton 
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Linear Regression graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# we will build polynomial model
from sklearn.preprocessing import PolynomialFeatures  
poly_reg = PolynomialFeatures(degree=3) 
X_poly = poly_reg.fit_transform(X)  

poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)  

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Level or Salary(Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show() 

lin_model_pred = lin_reg.predict([[6.5]]) 
lin_model_pred 

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred

# SVR MODEL  

from sklearn.svm import SVR 
svr_reg = SVR(kernel='poly', degree = 4, gamma = 'auto') 
svr_reg.fit(X, y) 

svr_model_pred = svr_reg.predict([[6.5]]) 
svr_model_pred

#print(svr_reg.get_params())
 
#knn 
from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor()
knn_reg.fit(X,y)

knn_reg_pred = knn_reg.predict([[6.5]])
knn_reg_pred

#decission tree
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(criterion='absolute_error',splitter='random',max_depth=3)
dt_reg.fit(X,y)

dt_reg_pred = dt_reg.predict([[6.5]])
dt_reg_pred

# random forest

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=45,random_state=0,max_depth=None,max_samples=5,min_samples_leaf=2)
rf_reg.fit(X,y)

rf_reg_pred = rf_reg.predict([[6.5]])
rf_reg_pred

#xgboost
import xgboost as xg
xgb_r = xg.XGBRFRegressor(objective='reg:Linear',n_estimators = 4)
xgb_r.fit(X,y)

xgb_reg_pred=xgb_r.predict([[6.5]])
xgb_reg_pred