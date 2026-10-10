import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\sahoo\Downloads\logit classification.csv")
dataset
# split with independent data to dependent
x = dataset.iloc[:,[2,3]].values # These are depenent data
y = dataset.iloc[:, -1].values # this is independent Data which is  Y

from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

# scale the data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# svc
#from sklearn.svm import SVC
#Classifier = SVC()
#Classifier.fit(x_train, y_train)

# KNN
from sklearn.neighbors import KNeighborsClassifier
Classifier_knn = KNeighborsClassifier()
Classifier_knn.fit(x_train, y_train)

y_pred = Classifier_knn.predict(x_test)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# This is to get the Models Accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = Classifier_knn.score(x_train, y_train)
print(bias)

variance = Classifier_knn.score(x_test, y_test)
print(variance)

# This is to get the Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)
