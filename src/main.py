#Data Preprocessing


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

dataset = pd.read_csv('C:\\Users\\DELL\\Prog\\templates\\Data science projects\\breast-cancer-detection\\data\\data.csv')
dataset.head()
dataset.select_dtypes(include='object').columns
len(dataset.select_dtypes(include='object').columns)
dataset.select_dtypes(include=['float','int']).columns
len(dataset.select_dtypes(include=['float','int']).columns)


#stats 

dataset.describe()
dataset.columns

#dealing with missing values

dataset.isnull().values.any()
dataset.isnull().values.sum()
dataset.columns[dataset.isnull().any()]
len(dataset.columns[dataset.isnull().any()])
dataset = dataset.drop(columns='Unnamed: 32')
dataset.shape

#checking categorical data

dataset['diagnosis'].unique()

dataset['diagnosis'].nunique()

dataset = pd.get_dummies(data=dataset, drop_first = True) #ERRORS TRUE AND FALSE INSTEAD OF 1 and O
dataset.head()
dataset['diagnosis_M'] = dataset['diagnosis_M'].astype(int)

#countplot

print(dataset['diagnosis_M'].value_counts())
sns.countplot(x=dataset['diagnosis_M'], order=[0,1])

#sns.countplot(data = dataset['diagnosis_M'], order=[0,1], label='Count') ERROR

#sns.countplot(dataset['diagnosis_M'], label = 'Count')   #errors
plt.show()

dataset['diagnosis_M'].value_counts()


#correlation matrix and heat map 

dataset_2 = dataset.drop(columns='diagnosis_M')
dataset_2

##dataset_2.corrwith(dataset['diagnosis_M']).plot.bar(
##    figsize = (20,10), title = 'correlation with diagnosis_M', rot = 45, grid = True
#)

dataset_2.corrwith(dataset['diagnosis_M']).plot.bar(figsize = (20,10), title = 'correlation with diagnosis_M', rot = 45, grid = True)
plt.show()


#Correlation Matrix 

corr = dataset.corr()
corr

#heatmap

plt.figure(figsize=(20,10))
sns.heatmap(corr,annot=True)
plt.show()


#split of data

#matrix of features

x = dataset.iloc[:,1:-1].values
x.shape

y = dataset.iloc[:,-1].values
y.shape

#split

from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
x_test.shape
y_train.shape
y_test.shape

#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)    #DIDNT UNDERSTAND
x_test = sc.transform(x_test)


#logistic regression

from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state=0)
classifier_lr.fit(x_train,y_train)
y_pred = classifier_lr.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix,f1_score,precision_score,recall_score
acc = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)

results = pd.DataFrame([['Logistic regression',acc,f1,prec,rec]], columns = ['Model','Accuracy','F1 score','Precision','Recall'])
results

cm = confusion_matrix(y_test,y_pred)
cm


#cross validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier_lr, X = x_train, y = y_train, cv = 10)
acc1 = (accuracies.mean())*100
sd  = accuracies.std()*100
sd
acc1

#Random Forest

from sklearn.ensemble import RandomForestClassifier
classifier_rm = RandomForestClassifier(random_state=0)
classifier_rm.fit(x_train,y_train)

y_pred = classifier_rm.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix,f1_score,precision_score,recall_score
acc = accuracy_score(y_test,y_pred)
f1 = f1_score(y_pred,y_pred)
prec = precision_score(y_pred,y_pred)
rec = recall_score(y_pred,y_pred)

model_results = pd.DataFrame([['Random Forest',acc,f1,prec,rec]], columns = ['Model','Accuracy','F1 score','Precision','Recall'])
#results = results.append(model_results,index = True) - -- giving error
results = pd.concat([results, model_results], ignore_index=True)
results

cm = confusion_matrix(y_test,y_pred)
cm
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier_rm, X = x_train, y = y_train, cv = 10)
acc1 = (accuracies.mean())*100
sd  = accuracies.std()*100
sd
acc1

