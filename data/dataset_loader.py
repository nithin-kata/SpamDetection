"""
Dataset loading utilities for the text classification system.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import urllib.request
import os

def load_sms_spam_dataset():
    """
    Load SMS Spam Collection dataset.
    Downloads the dataset if not present locally.
    """
    data_path = "data/sms_spam.csv"
    
    if not os.path.exists(data_path):
        print("Downloading SMS Spam dataset...")
        url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
        
        try:
            urllib.request.urlretrieve(url, data_path)
            print("Dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None, None, None
    
    # Load the dataset
    df = pd.read_csv(data_path, encoding='latin-1')
    df = df[['v1', 'v2']].copy()  # Keep only relevant columns
    df.columns = ['label', 'text']
    
    # Convert labels to numeric
    label_mapping = {'ham': 0, 'spam': 1}
    df['label_numeric'] = df['label'].map(label_mapping)
    
    texts = df['text'].tolist()
    labels = df['label_numeric'].tolist()
    label_names = {0: 'ham', 1: 'spam'}
    
    print(f"Loaded {len(texts)} SMS messages")
    print(f"Ham messages: {labels.count(0)}")
    print(f"Spam messages: {labels.count(1)}")
    
    return texts, labels, label_names

def load_20newsgroups_dataset(categories=None):
    """
    Load 20 Newsgroups dataset.
    
    Args:
        categories: List of categories to include. If None, uses a subset.
    """
    if categories is None:
        categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    
    print(f"Loading 20 Newsgroups dataset with categories: {categories}")
    
    # Load train and test sets
    newsgroups_train = fetch_20newsgroups(
        subset='train',
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=('headers', 'footers', 'quotes')
    )
    
    newsgroups_test = fetch_20newsgroups(
        subset='test',
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=('headers', 'footers', 'quotes')
    )
    
    # Combine train and test
    texts = list(newsgroups_train.data) + list(newsgroups_test.data)
    labels = list(newsgroups_train.target) + list(newsgroups_test.target)
    label_names = {i: name for i, name in enumerate(newsgroups_train.target_names)}
    
    print(f"Loaded {len(texts)} documents across {len(categories)} categories")
    
    return texts, labels, label_names

def load_movie_reviews_dataset():
    """
    Load a sample movie reviews dataset for sentiment analysis.
    This is a simplified version - in practice, you'd use a larger dataset.
    """
    # Sample movie reviews data
    positive_reviews = [
        "This movie was absolutely fantastic! Great acting and storyline.",
        "I loved every minute of it. Highly recommended!",
        "Outstanding performance by the lead actor. A must-watch film.",
        "Brilliant cinematography and excellent direction.",
        "One of the best movies I've seen this year.",
        "Incredible story with amazing visual effects.",
        "Perfect blend of action and emotion.",
        "Superb acting and well-written script.",
        "This film exceeded all my expectations.",
        "A masterpiece of modern cinema."
    ]
    
    negative_reviews = [
        "Terrible movie with poor acting and weak plot.",
        "I wasted my time watching this boring film.",
        "Disappointing storyline and bad direction.",
        "The worst movie I've ever seen.",
        "Poor script and unconvincing performances.",
        "Completely boring and predictable plot.",
        "Bad acting ruined what could have been a good story.",
        "Not worth watching, very disappointing.",
        "Poorly made film with no redeeming qualities.",
        "Awful movie with terrible dialogue."
    ]
    
    texts = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)  # 1 = positive, 0 = negative
    label_names = {0: 'negative', 1: 'positive'}
    
    print(f"Loaded {len(texts)} movie reviews")
    print(f"Positive reviews: {labels.count(1)}")
    print(f"Negative reviews: {labels.count(0)}")
    
    return texts, labels, label_names

def get_available_datasets():
    """Return list of available datasets."""
    return {
        'sms_spam': {
            'name': 'SMS Spam Detection',
            'description': 'Binary classification of SMS messages as spam or ham',
            'loader': load_sms_spam_dataset
        },
        '20newsgroups': {
            'name': '20 Newsgroups',
            'description': 'Multi-class classification of newsgroup posts',
            'loader': load_20newsgroups_dataset
        },
        'movie_reviews': {
            'name': 'Movie Reviews Sentiment',
            'description': 'Binary sentiment analysis of movie reviews',
            'loader': load_movie_reviews_dataset
        }
    }

def load_dataset(dataset_name):
    """
    Load a dataset by name.
    
    Args:
        dataset_name: Name of the dataset to load
    
    Returns:
        texts, labels, label_names
    """
    datasets = get_available_datasets()
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}")
    
    return datasets[dataset_name]['loader']()