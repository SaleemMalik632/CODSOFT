import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

Data = pd.read_csv('spam.csv')
print(Data)
X = Data['v2']
ModelNumerical = CountVectorizer()
NumericalData = ModelNumerical.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(NumericalData, Data['v1']) 

print(X_train) 
dense_X_train = X_train.toarray()
print('Dense training data is here') 
print(dense_X_train)

# Convert X_test to dense array
dense_X_test = X_test.toarray()
print('Dense testing data is here') 
print(dense_X_test)

Model = GaussianNB()
Model.fit(dense_X_train, y_train) 
text = 'Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free!'
NumText = ModelNumerical.transform([text]) 
Test_text = NumText.toarray()  
predictions = Model.predict(Test_text)
print(predictions)  
print(Model.score(dense_X_test, y_test)) 
