import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
traindf= pd.read_csv(r'C:\Users\FAISAL\Downloads\Assignment\Training.csv')
testdf= pd.read_csv(r'C:\Users\FAISAL\Downloads\Assignment\Testing.csv')

print(len(traindf.index))
print(len(testdf.index))

df = testdf.append(traindf)
print(len(df.index))

print(df.isnull().values.any())
print(df.isnull().sum())

df.drop('Unnamed: 133', axis = 1, inplace=True)
df.drop('fluid_overload', axis = 1, inplace=True) #Unused Variable
df.isnull().sum()
print(df.isnull().sum())

x = df.drop('prognosis',axis=1)
y = df['prognosis']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 42)
print(df.describe())

print(df.info())

#Models
#Support Vector Machine
svc=SVC().fit(xtrain, ytrain)
ypred_svc=svc.predict(xtest)

#R-Squared
r_squared= svc.score(xtrain, ytrain)
print("R-Squared")
print(r_squared)
#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(ytest, ypred_svc)
print('Confusion Matrix\n')
print(confusion)
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(ytest, ypred_svc)))

print('Micro Precision: {:.2f}'.format(precision_score(ytest, ypred_svc, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(ytest, ypred_svc, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(ytest, ypred_svc, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(ytest, ypred_svc, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(ytest, ypred_svc, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(ytest, ypred_svc, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(ytest, ypred_svc, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(ytest, ypred_svc, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(ytest, ypred_svc, average='weighted')))

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(ytest, ypred_svc))


## Logistic Regression

lr=LogisticRegression().fit(xtrain, ytrain)
ypred_lr=lr.predict(xtest)
#R-Squared
r_squared= lr.score(xtrain, ytrain)
print("R-Squared")
print(r_squared)

#Confusion Matrix
confusion = confusion_matrix(ytest, ypred_lr)
print('Confusion Matrix\n')
print(confusion)
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(ytest, ypred_lr)))

print('Micro Precision: {:.2f}'.format(precision_score(ytest, ypred_lr, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(ytest, ypred_lr, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(ytest, ypred_lr, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(ytest, ypred_lr, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(ytest, ypred_lr, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(ytest, ypred_lr, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(ytest, ypred_lr, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(ytest, ypred_lr, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(ytest, ypred_lr, average='weighted')))

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(ytest, ypred_lr))



## Decision Tree Algorithm
tree=DecisionTreeClassifier().fit(xtrain, ytrain)
ypred_tree=tree.predict(xtest)

#R-Squared
r_squared= tree.score(xtrain, ytrain)
print("R-Squared")
print(r_squared)

#Confusion Matrix
confusion = confusion_matrix(ytest, ypred_tree)
print('Confusion Matrix\n')
print(confusion)
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(ytest, ypred_tree)))

print('Micro Precision: {:.2f}'.format(precision_score(ytest, ypred_tree, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(ytest, ypred_tree, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(ytest, ypred_tree, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(ytest, ypred_tree, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(ytest, ypred_tree, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(ytest, ypred_tree, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(ytest, ypred_tree, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(ytest, ypred_tree, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(ytest, ypred_tree, average='weighted')))

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(ytest, ypred_tree))





