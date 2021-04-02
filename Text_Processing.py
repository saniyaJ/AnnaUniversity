import nltk 
#nltk.download()


#Tokenization of Sentences
     #We use the method sent_tokenize() to split a paragraph into sentence.
     
from nltk.tokenize import sent_tokenize
text = "God is Great! I won a lottery. So i am happy."
print(sent_tokenize(text))
#Output: ['God is Great!', 'I won a lottery ']


#Tokenization of words

from nltk.tokenize import word_tokenize
text = "God is Great! I won a lottery. So i am happy."
print(word_tokenize(text))
print(text.split())
#Output: ['God', 'is', 'Great', '!', 'I', 'won', 'a', 'lottery', '.']



#Stemming

import nltk
from nltk.stem.porter import PorterStemmer
porter_stemmer  = PorterStemmer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
    print("Stemming for {} is {}".format(w,porter_stemmer.stem(w)))  
    


#Lemmatization
    
import nltk
from nltk.stem import 	WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
    print("Lemma for {} is {}".format(w, wordnet_lemmatizer.lemmatize(w)))
    
    
       
#stopwords 
    
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
  
example_sent = """This is a sample sentence,
                  showing off the stop words filtration."""
  
stop_words = set(stopwords.words('english')) 
  
word_tokens = word_tokenize(example_sent) 
  
filtered_sentence = [w for w in word_tokens if not w in stop_words] 
  
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 
  
print(word_tokens) 
print(filtered_sentence) 

    
    
#Bag of Words or CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
data_corpus=["Word Embedding is a type of word representation that allows words with similar meaning to be understood by machine learning algorithms."]
vocabulary=vectorizer.fit(data_corpus)
X= vectorizer.transform(data_corpus)
print(X.toarray())
print(vocabulary.get_feature_names())




#TF-IDF Approach

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()
data_corpus=["Word Embedding is a type of word representation that allows words with similar meaning to be understood by machine learning algorithms."]
vocabulary=vectorizer.fit(data_corpus)
X= vectorizer.transform(data_corpus)
print(X.toarray())
print(vocabulary.get_feature_names())