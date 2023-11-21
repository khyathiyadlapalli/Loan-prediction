import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import pickle

data = pd.read_csv("Loan.csv")

# Use the print function to display the result of data.head()
print(data.head(5))
print(data.columns)
print(data.info())
data['loanAmount_log']=np.log(data['LoanAmount'])
data['loanAmount_log'].hist(bins=20)
# creating a new column called TotalIncome which containes the combined data of application and coapplication....
data['TotalIncome']=data['ApplicantIncome']+data['CoapplicantIncome']
data['TotalIncome_log']=np.log(data['TotalIncome'])
data['TotalIncome_log'].hist(bins=20)
print(data['TotalIncome_log'])

#Handling the missing values...
data['Gender'].fillna(data['Gender'].mode()[0],inplace = True)
data['Married'].fillna(data['Married'].mode()[0],inplace = True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0],inplace = True)
data['Dependents'].fillna(data['Dependents'].mode()[0],inplace = True)
data.LoanAmount = data.LoanAmount.fillna(data.LoanAmount.mean())
data.loanAmount_log = data.loanAmount_log.fillna(data.loanAmount_log.mean())

data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace = True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace = True)
data.isnull().sum()


#selecting dependent and independent labels for modeel training...
x=data.iloc[:,np.r_[1:5,9:11,13:15]].values
y=data.iloc[:,12].values
print([x,y])
print(data['Credit_History'])
selected_columns = data.columns[np.r_[1:5, 9:11, 13:15]]
print("MMMMMMMmMMMM")
print(selected_columns)
print(x[0])

#Spliting the dataset into train and test....

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
Labelencoder_x=LabelEncoder()
for i in range(0,5):
    X_train[:,i]=Labelencoder_x.fit_transform(X_train[:,i])
    X_train[:,7]=Labelencoder_x.fit_transform(X_train[:,7])
X_train
print(X_train[0])
#fit transforms joins these two steps and is used for the initial fitting of the parameteron the trading set,while also returning the transformer X internally the tranfer obj just calls first fit and thentransform on the same data.  
Labelencoder_y = LabelEncoder()
y_train = Labelencoder_y.fit_transform(y_train)

y_train

for i in range(0,5):
    X_test[:,i]=Labelencoder_x.fit_transform(X_test[:,i])
    X_test[:,7] = Labelencoder_x.fit_transform(X_test[:,7])
X_test
print(X_test)
print(X_test[0])
print(X_test[0])

Labelencoder_y=LabelEncoder()
y_test=Labelencoder_y.fit_transform(y_test)
y_test
print(y_test)
print(y_test[0])

#FEATURE SCALING....

ss= StandardScaler()
X_train = ss.fit_transform(X_train)
x_test = ss.fit_transform(X_test)

#Initiating the NaiveB model...

nb_clf = GaussianNB()
#fitting the model
nb_clf.fit(X_train,y_train)

pickle.dump(nb_clf, open("loan_model.pkl", "wb"))