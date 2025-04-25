import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TextProcessor:
    def __init__(self, max_features=5000):
        required_resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource)
            
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
            
       
        text = text.lower()
        
      
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
      
        try:
            tokens = word_tokenize(text)
        except LookupError:
           
            tokens = text.split()
        
        
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def fit_transform(self, texts):
        processed_texts = [self.preprocess_text(text) for text in texts]
        
      
        return self.vectorizer.fit_transform(processed_texts)
    
    def transform(self, texts):
      
        processed_texts = [self.preprocess_text(text) for text in texts]
        
      
        return self.vectorizer.transform(processed_texts)
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out() 