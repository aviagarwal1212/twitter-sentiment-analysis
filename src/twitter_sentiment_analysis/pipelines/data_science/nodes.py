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
    '''
    Splits data into training and test set.
    
     Args:
        data: Source processed data.
        parameters: Parameters defined in parameter.yml.
    
     Returns:
        A list containing split data.
        
    '''
    data = _handle_empty(data)
    X = data['text'].values
    y = data['sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters['test_size'], random_state=parameters['random_state'])
    return [X_train, X_test, y_train, y_test]
    
def vectorize_text(X_train: np.ndarray, X_test: np.ndarray, parameters: Dict) -> List:
    '''
    Vectorize text column in train and test set.
    
     Args:
        X_train: Training text data.
        X_test: Testing text data.
        parameters: Parameters defined in parameter.yml.   
       
     Returns:
        A list containing vectorized train and test feature sets. 
        
    '''
    vectorizer = TfidfVectorizer(ngram_range=(parameters['ngram_range_min'], parameters['ngram_range_max']), max_features=parameters['max_features'])
    vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)
    return [X_train, X_test]
    
def train_model(X_train: np.ndarray, y_train: np.ndarray, parameters: Dict) -> LogisticRegression:
    '''
    Train the logistic regression model.
    
     Args:
        X_train: Vectorized training text data.
        y_train: Training data for sentiment.
        parameters: Parameters defined in parameter.yml.
        
     Returns:
        Trained model.
        
    '''
    model = LogisticRegression(max_iter=parameters['max_iter'], n_jobs=-1, C=0.5)
    model.fit(X_train, y_train)
    return model
    
def evaluate_model(X_test: np.ndarray, y_test: np.ndarray, model: LogisticRegression):
    '''
    Generate and log classification report for test data.
    
     Args:
        X_test: Vectorized test text data.
        y_test: Test data for sentiment.
        model: Trained model.
        
    '''
    y_pred = model.predict(X_test)
    logger = logging.getLogger(__name__)
    logger.info(classification_report(y_test, y_pred))

# Secondary Functions

def _handle_empty(data: pd.DataFrame) -> pd.DataFrame:
    data['text'] = data['text'].apply(str)
    return data