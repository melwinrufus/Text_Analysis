import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# App Title
st.title("📝 Sentiment Analysis App (Single File)")
st.write("This app analyzes the sentiment (positive/negative) of your input text using a Logistic Regression model.")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
    df = pd.read_csv(url)
    df = df[['label', 'tweet']]
    df.columns = ['label', 'text']
    return df

@st.cache_resource
def train_model(data):
    X = data['text']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    return model, vectorizer

# Load and train
with st.spinner("Loading and training model..."):
    data = load_data()
    model, vectorizer = train_model(data)

# User Input
user_input = st.text_area("Enter your sentence here:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]
        label = "😊 Positive" if prediction == 1 else "☹️ Negative"
        st.success(f"Prediction: {label}")
