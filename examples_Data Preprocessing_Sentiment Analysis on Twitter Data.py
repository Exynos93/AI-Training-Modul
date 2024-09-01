# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords

# Load the data
data = pd.read_csv('data/raw/tweets.csv')

# Display first few rows
data.head()

# Data Cleaning Function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#', '', text)        # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()                  # Convert to lowercase
    return text

# Apply the cleaning function
data['cleaned_text'] = data['text'].apply(clean_text)

# Remove stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
data['cleaned_text'] = data['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Encode the target variable
le = LabelEncoder()
data['sentiment'] = le.fit_transform(data['sentiment'])

# Split the data
X = data['cleaned_text']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Save the processed data
np.save('data/processed/X_train.npy', X_train_vect.toarray())
np.save('data/processed/X_test.npy', X_test_vect.toarray())
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/y_test.npy', y_test)
