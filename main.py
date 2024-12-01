import nltk
nltk.download('punkt')
nltk.download('stopwords')
import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk



# Load the pre-trained model and TF-IDF vectorizer
model = joblib.load('best_model.pkl')  # Load your trained model
tfidf = joblib.load('tfidf_vectorizer.pkl')  # Load the pre-trained TF-IDF vectorizer

# Initialize NLTK tools for text processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to preprocess the text
def normalize_text(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [re.sub(r"[^\w\s]", "", word) for word in words if word.isalpha() and word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Streamlit app UI
st.title("Spam vs Ham Classifier")
st.write("Enter a message and let the model predict if it's spam or ham.")

# User input for message
message = st.text_input("Enter your message:")

if st.button('Predict'):
    if message:
        # Preprocess the input message
        processed_message = normalize_text(message)

        # Transform the message using TF-IDF vectorizer
        tfidf_message_features = tfidf.transform([processed_message])

        # Predict using the pre-trained model
        prediction = model.predict(tfidf_message_features)

        # Display the result
        if prediction == 1:
            st.write("This message is SPAM.")
        else:
            st.write("This message is HAM.")
    else:
        st.write("Please enter a message to predict.")

