"""
Text preprocessing utilities for the classification system.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class TextPreprocessor:
    """Handles all text preprocessing operations."""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def clean_text(self, text):
        """Clean and normalize text data."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_remove_stopwords(self, text):
        """Tokenize text and remove stopwords."""
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens
    
    def stem_tokens(self, tokens):
        """Apply stemming to tokens."""
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess_text(self, text, use_stemming=True):
        """Complete preprocessing pipeline."""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and remove stopwords
        tokens = self.tokenize_and_remove_stopwords(cleaned_text)
        
        # Apply stemming if requested
        if use_stemming:
            tokens = self.stem_tokens(tokens)
        
        # Join tokens back to string
        return ' '.join(tokens)

class FeatureExtractor:
    """Handles feature extraction from preprocessed text."""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
    
    def extract_tfidf_features(self, texts, max_features=5000):
        """Extract TF-IDF features."""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        
        features = self.tfidf_vectorizer.fit_transform(texts)
        return features
    
    def extract_bow_features(self, texts, max_features=5000):
        """Extract Bag of Words features."""
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        features = self.count_vectorizer.fit_transform(texts)
        return features
    
    def transform_tfidf(self, texts):
        """Transform new texts using fitted TF-IDF vectorizer."""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted yet")
        return self.tfidf_vectorizer.transform(texts)
    
    def transform_bow(self, texts):
        """Transform new texts using fitted Count vectorizer."""
        if self.count_vectorizer is None:
            raise ValueError("Count vectorizer not fitted yet")
        return self.count_vectorizer.transform(texts)