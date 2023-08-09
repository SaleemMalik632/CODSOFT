import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB 

class ClassName: 
    TrainData = []
    Classifier  = LogisticRegression(max_iter=1000) 
    Classifier2 = GaussianNB()   
    TestData = [] 
    def __init__(self):
        print('Constrasvet of the class') 
    def LoadTrainData(self):
        self.path = 'Genre Classification Dataset/train_data.txt'   
        with open(self.path, 'r', encoding='unicode_escape') as file:
            for line in file:
                if line:
                    self.part = line.split(':::') 
                    self.TrainData.append(self.part) 
        train_columns = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']          
        self.TrainData = pd.DataFrame(self.TrainData , columns=train_columns) 
        self.TrainData['Text'] = self.TrainData['TITLE'] + ' ' + self.TrainData['DESCRIPTION'] 
        self.Numrical = TfidfVectorizer(stop_words='english') 
        self.NumricalData = self.Numrical.fit_transform(self.TrainData['Text'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.NumricalData , self.TrainData['GENRE'])  
        self.Classifier.fit(self.X_train , self.y_train)  
        print('Model is train') 
        self.LoadTestData() 
    def LoadTestData(self):
        self.path = 'Genre Classification Dataset/test_data.txt'
        with open(self.path , 'r' , encoding='unicode_escape') as file:
            for line in file:
                if line:
                    self.part = line.split(':::') 
                    self.TestData.append(self.part) 
        test_columns = ['ID', 'TITLE', 'DESCRIPTION']          
        self.TestData = pd.DataFrame(self.TestData , columns=test_columns) 
        self.TestData['Text'] = self.TestData['TITLE'] + ' ' + self.TestData['DESCRIPTION'] 
        self.Numrical = TfidfVectorizer(stop_words='english') 
        self.NumricalTestData = self.Numrical.fit_transform(self.TestData['Text'])      
        self.predection =  self.Classifier.predict(self.NumricalTestData)     
        print(self.predection)    


  


  
obj1 = ClassName() 
obj1.LoadTrainData() 
 