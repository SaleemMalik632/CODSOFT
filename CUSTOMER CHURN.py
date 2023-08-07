import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

Data   = pd.read_csv('Churn_Modelling.csv')   # Loading the CSV file in the program 

Encoder  = LabelEncoder()  # Object of the Encoder Bcz ML is only work on the Numrical Data 
Data['Gender'] = Encoder.fit_transform(Data['Gender'])  # Encodeing the Gender 
Data['Geography'] = Encoder.fit_transform(Data['Geography'])  #  Encodeing the Geography

# Spliting The Data in Traing  and Testing parts 

X_train, X_test, y_train, y_test =  train_test_split(Data[['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']] , Data['Exited']  , test_size=0.2 ) 


# SimpleTest   = [[792,0,0,28,4,130142.79,1,1,0,38190.78]]

Model = LogisticRegression()
Model.fit(X_train , y_train)  # Training the Model 
print(Model.predict(X_test))  #Prediction of The Model 
print(Model.score(X_test , y_test)*100)   #prediction Score of the Model 
 



  
  




