# importing the Dataset

import pandas as pd

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])

#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
#ps = PorterStemmer()
lemmer = nltk.stem.WordNetLemmatizer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [lemmer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating the TFIDF model
from sklearn.feature_extraction.text import TfidfVectorizer
TfidfVec = TfidfVectorizer(max_features=2500)
X = TfidfVec.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)


y_pred=spam_detect_model.predict(X_test)

#performance_Analysis

from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test,y_pred)

#checking_Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)



















