# utilities
from typing import List, Dict
import pandas as pd
import numpy as np
import logging

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Primary Functions

def split_data(data: pd.DataFrame, parameters: Dict) -> List:
    
    data = _handle_empty(data)
    X = data['text'].values
    y = data['sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters['test_size'], random_state=parameters['random_state'])
    return [X_train, X_test, y_train, y_test]
    
def vectorize_text(X_train: np.ndarray, X_test: np.ndarray, parameters: Dict) -> List:
    
    vectorizer = TfidfVectorizer(ngram_range=(parameters['ngram_range_min'], parameters['ngram_range_max']), max_features=parameters['max_features'])
    vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)
    return [X_train, X_test]
    
def train_model(X_train: np.ndarray, y_train: np.ndarray, parameters: Dict) -> LogisticRegression:
    
    model = LogisticRegression(max_iter=parameters['max_iter'], n_jobs=-1, C=0.5)
    model.fit(X_train, y_train)
    return model
    
def evaluate_model(X_test: np.ndarray, y_test: np.ndarray, model: LogisticRegression):
    
    y_pred = model.predict(X_test)
    logger = logging.getLogger(__name__)
    logger.info(classification_report(y_test, y_pred))

# Secondary Functions

def _handle_empty(data: pd.DataFrame) -> pd.DataFrame:
    data['text'] = data['text'].apply(str)
    return data