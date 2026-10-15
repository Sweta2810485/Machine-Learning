import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\sahoo\Downloads\Churn_Modelling.csv")

x = dataset.iloc[:,3:-1].values # These are depenent data
y = dataset.iloc[:, -1].values # this is independent Data which is  Y

print(x)
print(y)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

print(x)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[1])],remainder='passthrough')

x = np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from xgboost import XGBClassifier
Classifier = XGBClassifier(n_estimators=25) 
Classifier.fit(x_train, y_train)

print(Classifier.get_params())

y_pred = Classifier.predict(x_test)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# This is to get the Models Accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = Classifier.score(x_train, y_train)
print(bias)

# logistic 
# svm
# dt
# rf 

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = Classifier, X = x_train, y = y_train, cv = 5)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
#print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

