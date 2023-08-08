import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
# Define the file path
file_path = 'Genre Classification Dataset/train_data.txt'
 
data = []
with open(file_path, 'r', encoding='unicode_escape') as file:
    for line in file:
        line = line.strip()
        if line:
            parts = line.split(':::')
            data.append(parts)

# Define the column names
columns = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']

df = pd.DataFrame(data, columns=columns)
# Combine 'TITLE' and 'DESCRIPTION' columns into a single text column
df['TEXT'] = df['TITLE'] + ' ' + df['DESCRIPTION'] 

print(df['TEXT']) 

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(df['TEXT']) 
 
 
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['GENRE'])  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




