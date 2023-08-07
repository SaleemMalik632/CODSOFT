import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

Data   = pd.read_csv('Churn_Modelling.csv') 

Encoder  = LabelEncoder()
Data['Gender'] = Encoder.fit_transform(Data['Gender']) 
Data['Geography'] = Encoder.fit_transform(Data['Geography'])  

X_train, X_test, y_train, y_test =  train_test_split(Data[['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']] , Data['Exited']  , test_size=0.2 ) 

Model = LogisticRegression()
Model.fit(X_train , y_train) 
print(Model.predict(X_test)) 
print(Model.score(X_test , y_test)*100)  
 



  
  




