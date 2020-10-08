# utilities
import pandas as pd
import matplotlib.pyplot as plt
from emot.emo_unicode import EMOTICONS
import re

# nltk
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')

# Primary Functions

def clean_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    
    _pertinent_columns = ['sentiment', 'text']
    data = data[_pertinent_columns]
    
    return data
    
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    
    data['text'] = data['text'].apply(_convert_links)
    data['text'] = data['text'].apply(_convert_usernames)
    data['text'] = data['text'].apply(_convert_emoticons)
    data['text'] = data['text'].apply(_lemmatize)

    return data


# Seconday Functions

def _convert_emoticons(text: str) -> str:
    
    for emot in EMOTICONS:
        text = re.sub(emot, EMOTICONS[emot], text)
    return text
    
def _convert_links(text: str) -> str:

    text = re.sub(r'(https?://\S+)|(www.\S+)', ' link ', text, flags = re.IGNORECASE)
    return text
    
def _convert_usernames(text: str) -> str:
    
    text = re.sub(r'@\S+', ' username ', text)
    return text
    
def _lemmatize(text: str) -> str:
    
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(text)
    