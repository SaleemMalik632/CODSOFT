import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

Data = pd.read_csv('spam.csv')
X = Data['v2']
ModelNumerical = CountVectorizer()
NumericalData = ModelNumerical.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(NumericalData, Data['v1']) 

dense_X_train = X_train.toarray()
# Convert X_test to dense array 
dense_X_test = X_test.toarray()

Model = GaussianNB()
Model.fit(dense_X_train, y_train) 
text = "URGENT: Your bank account has been compromised. Click here to secure your account."
NumText = ModelNumerical.transform([text]) 
Test_text = NumText.toarray()  
predictions = Model.predict(Test_text)
print(predictions)  
print(Model.score(dense_X_test, y_test)) 
