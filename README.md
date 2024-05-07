# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.Import the libraries and read the data frame using pandas. 
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset. 
4.calculate Mean square error,data prediction and r2. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: LOKESH R
RegisterNumber:  212222240055
*/
```

```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])

```

## Output:

![image](https://github.com/LokeshRajamani/ml7/assets/120544804/465a1581-46a8-41e3-b4d1-a85f9e64cc35)

Date Info:

![image](https://github.com/LokeshRajamani/ml7/assets/120544804/98e17618-8b51-4263-9a11-496c27556f1a)

Optimization of null values:

![image](https://github.com/LokeshRajamani/ml7/assets/120544804/997f4afd-677d-4dd2-a969-83378f725ec0)

Converting string literals to numerical values using label encoder:

![image](https://github.com/LokeshRajamani/ml7/assets/120544804/3e17ec3c-61a0-4ec1-a094-cc05fdef5e2c)

Assigning x and y values:

![image](https://github.com/LokeshRajamani/ml7/assets/120544804/4c875e13-5e1e-4b2f-b5ab-a2628a338ad1)


Mean Squared Error:

![image](https://github.com/LokeshRajamani/ml7/assets/120544804/ce71ee57-cc8d-4e82-b365-f8d56917e30c)

R2 (variance):

![image](https://github.com/LokeshRajamani/ml7/assets/120544804/e134147f-71cb-4d28-bbc5-fe1cc2e4e000)

Prediction:

![image](https://github.com/LokeshRajamani/ml7/assets/120544804/ea92504a-e247-4759-b8eb-012a9ca96e88)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
