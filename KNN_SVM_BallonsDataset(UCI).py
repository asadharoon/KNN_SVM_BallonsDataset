# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:15:40 2020

@author: Asad
"""


import numpy
import pandas as pd
from sklearn.model_selection import train_test_split

data_set_url="https://archive.ics.uci.edu/ml/machine-learning-databases/balloons/adult+stretch.data"

columns=['Color','Size','Act','Age','Inflated']

dataset=pd.read_csv(data_set_url,names=columns)


from sklearn.preprocessing import LabelEncoder

train_dataset_encoded=dataset

color_preprocessed=LabelEncoder()
size_preprocessed=LabelEncoder()
act_preprocessed=LabelEncoder()
age_preprocessed=LabelEncoder()
inflated_preprocessed=LabelEncoder()

train_dataset_encoded['Color']=color_preprocessed.fit_transform(dataset['Color'])
train_dataset_encoded['Size']=size_preprocessed.fit_transform(dataset['Size'])
train_dataset_encoded['Act']=act_preprocessed.fit_transform(dataset['Act'])
train_dataset_encoded['Age']=age_preprocessed.fit_transform(dataset['Age'])

train_dataset_encoded['Inflated']=inflated_preprocessed.fit_transform(dataset['Inflated'])

X=train_dataset_encoded.iloc[:,:-1]
y=train_dataset_encoded.iloc[:,4]
y=train_dataset_encoded['Inflated']

#Train test split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

#model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3,metric="manhattan")
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

#Metrics
from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy of KNN is ",accuracy)




#Import Library
from sklearn import svm
model=svm.SVC()
model.fit(X_train, y_train)

#Predict Output
predicted= model.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_svm=accuracy_score(y_test,predicted)
print("Accuracy of SVM is ",accuracy_svm)