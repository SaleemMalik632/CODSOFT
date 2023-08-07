import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

Data = pd.read_csv('spam.csv') # Loading CSV File 
X = Data['v2']  
ModelNumerical = CountVectorizer() # Making the Object od the CountVectorizer bcz NB is work on the numricall 
NumericalData = ModelNumerical.fit_transform(X)  # A Matrix of Numrical data is Found 

# Spliting the Data in traing and testing pass 
X_train, X_test, y_train, y_test = train_test_split(NumericalData, Data['v1']) 

dense_X_train = X_train.toarray()
# Convert X_test to dense array 
dense_X_test = X_test.toarray() 

Model = GaussianNB()
Model.fit(dense_X_train, y_train) 
# test for the preduction of the my model 
text = "URGENT: Your bank account has been compromised. Click here to secure your account."
NumText = ModelNumerical.transform([text]) 
Test_text = NumText.toarray()   # again making the numrical matrix of the data 
predictions = Model.predict(Test_text)
print(predictions)   # prediction of my model 
print(Model.score(dense_X_test, y_test))  # Score and Accuracy of the model 
