import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import email
import string
from bs4 import BeautifulSoup
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve 

nltk.data.path.append("/home/adminuser/venv/lib/python3.13/site-packages/nltk/data.py")

try:  # Wrap in a try-except block for better error handling
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
except LookupError:
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
class email_to_clean_text(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None): 
        return self
    def transform(self, X):
        text_list = []
        for mail in X:
            b = email.message_from_string(mail)
            body = ""

            if b.is_multipart():
                for part in b.walk():
                    ctype = part.get_content_type()
                    cdispo = str(part.get('Content-Disposition'))

                    # skip any text/plain (txt) attachments
                    if ctype == 'text/plain' and 'attachment' not in cdispo:
                        body = part.get_payload(decode=True)  # get body of email
                        break
            # not multipart - i.e. plain text, no attachments, keeping fingers crossed
            else:
                body = b.get_payload(decode=True) # get body of email
            
            soup = BeautifulSoup(body, "html.parser") #get text from body (HTML/text)
            text = soup.get_text().lower()
            text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE) #remove links
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text, flags=re.MULTILINE) #remove email addresses
            text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
            text = ''.join([i for i in text if not i.isdigit()]) # remove digits
            stop_words = stopwords.words('english')

            words_list = [w for w in text.split() if w not in stop_words] # remove stop words
            words_list = [lemmatizer.lemmatize(w) for w in words_list] #lemmatization
            words_list = [stemmer.stem(w) for w in words_list] #Stemming

            text_list.append(' '.join(words_list))
        return text_list

# Load the pre-trained pipeline:
pipeline_filename = "email_classifier_naive.joblib" 
loaded_pipeline = joblib.load(pipeline_filename)

def classify_email(email_text):
    """Classifies an email and returns probabilities."""
    prediction = loaded_pipeline.predict([email_text])
    probabilities = loaded_pipeline.predict_proba([email_text])
    
    result = "The given email is HAM" if prediction[0] == 0 else "The given email is SPAM"
    ham_prob = probabilities[0][0]
    spam_prob = probabilities[0][1]
    
    return result, ham_prob, spam_prob # Return all three values


# Streamlit app:
st.title("Spam Email Classifier")

email_input = st.text_area("Enter email text here:")

if st.button("Classify"):
    if email_input:
        result, ham_prob, spam_prob = classify_email(email_input)  # Get all returned values
        st.write(f"**Classification:** {result}")
        st.write(f"**Probability of HAM:** {ham_prob:.2f}")  # Format to 4 decimal places
        st.write(f"**Probability of SPAM:** {spam_prob:.2f}")
    else:
        st.write("Please enter some email text.")
