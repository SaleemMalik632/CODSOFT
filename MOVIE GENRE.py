import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 

# Define the file path
file_path = 'Genre Classification Dataset/train_data.txt'

# Read the file line by line
train_data = []
with open(file_path, 'r', encoding='unicode_escape') as file:
    for line in file:
        line = line.strip()
        if line:
            parts = line.split(':::')
            train_data.append(parts)

# Define the column names
columns = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']

train_data = pd.DataFrame(train_data, columns=columns)

title = train_data['TITLE']
description = train_data['DESCRIPTION']


# Combine title and description text into a single array
text_data = title + ' ' + description 
print(len(text_data))
NumricalData = CountVectorizer()
text_transformed = NumricalData.fit_transform(description)

print(text_transformed.getnnz())  
print(len(title))      
print(len(train_data['GENRE']))

# X_train, X_test, y_train, y_test = train_test_split( [[text_transformed]], train_data['GENRE'] , test_size=0.2)   

# Model = LogisticRegression()
# Model.fit(X_train , y_train) 















# # Reading the testing data 
# path = 'Genre Classification Dataset/test_data.txt'
# test_data = []
# with open(path , 'r'  , encoding='unicode_escape') as file:
#     for line in file:
#         line = line.strip()
#         if line:
#             parts  = line.split(':::')
#             test_data.append(parts) 

 
# columns_ = ['ID', 'TITLE', 'DESCRIPTION']
# test_data = pd.DataFrame(test_data , columns= columns_) 
# print(test_data['DESCRIPTION']) 





