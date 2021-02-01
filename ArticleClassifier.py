#import the liberaries
import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer 
       
#import the dataset
dataset=pd.read_csv('bbcdataset.csv',encoding= 'unicode_escape') 
#ataset=pd.read_csv('bbcdataset.csv')  
#cleaning the text
corpus=[]
PS=PorterStemmer()
lemmatizer = WordNetLemmatizer() 
for i in range(0,120000):
    print(i)
    article=re.sub('[^a-zA-Z ]',' ',dataset['text'][i])
    article=article.lower()
    article=article.split()   
    article=[PS.stem(word) for word in article if not word in set(stopwords.words('english'))]
    article=[lemmatizer.lemmatize(word) for word in article]    
    article=' '.join(article)
    corpus.append(article)

# Prepared dataset
#dataset=pd.read_csv('preparedDataset.csv') 
dataset=pd.read_csv('bbcPreparedData.csv') 
#creating the bag of words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfVectorizer = TfidfVectorizer(encoding='latin-1', ngram_range=(1,1), stop_words='english')   
#tfidfVectorizer=tfidfVectorizer(max_features=1000)
tfidfVectorizer = tfidfVectorizer.fit(dataset['text'])
x=tfidfVectorizer.fit_transform(dataset['text'])
y=dataset.iloc[:,1].values

# Load libraries
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#
## Select features with highest chi-squared statistics
# for the big dataset
#chi2_selector = SelectKBest(chi2, k=15000)
# for the BBC dataset
chi2_selector = SelectKBest(chi2, k=5000)
X_kbest = chi2_selector.fit_transform(x, y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_kbest, y, test_size = 0.2, random_state = 0)

#Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)


# Fitting Random Forest Classification to the Training set
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)

# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors=7)
# classifier.fit(X_train, y_train)

# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train.toarray(), y_train)


# from sklearn import tree
# classifier = tree.DecisionTreeClassifier()
# classifier.fit(X_train, y_train)

# Fitting SVM to the Training set
# from sklearn.svm import SVC
# classifier=SVC(kernel='linear',random_state=0)
# classifier.fit(X_train,y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test.toarray())
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

