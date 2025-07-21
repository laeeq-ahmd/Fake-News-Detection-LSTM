# fast_preprocessing.py - Fast preprocessing with smaller dataset

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

print("FAST preprocessing with smaller dataset...")

# Load the combined dataset
data = pd.read_csv("../news_dataset.csv")

print(f"Original dataset size: {len(data)}")

# Take only 10,000 samples per class for faster training
fake_df = data[data['label'] == 0].head(10000)
real_df = data[data['label'] == 1].head(10000)

balanced_data = pd.concat([fake_df, real_df], ignore_index=True)
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Reduced dataset size: {len(balanced_data)}")
print(f"Class distribution:")
print(balanced_data['label'].value_counts())

# Combine title and text
balanced_data['text'] = balanced_data['title'].fillna('') + ' ' + balanced_data['text'].fillna('')

# Simplified text cleaning for speed
def clean_text_fast(text):
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remove URLs and HTML
    text = re.sub(r'https?://\S+|www\.\S+|<.*?>', '', text)
    
    # Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove stopwords (only most common ones for speed)
    common_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    words = text.split()
    words = [word for word in words if word not in common_stopwords and len(word) > 2]
    
    return ' '.join(words).strip()

print("Fast text cleaning...")
balanced_data['text'] = balanced_data['text'].apply(clean_text_fast)

# Remove empty texts
balanced_data = balanced_data[balanced_data['text'].str.len() > 10]
print(f"After cleaning: {len(balanced_data)}")

# Features and labels
X = balanced_data['text'].values
y = balanced_data['label'].values

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Save processed data
np.save("../X_train.npy", X_train)
np.save("../X_test.npy", X_test)
np.save("../y_train.npy", y_train)
np.save("../y_test.npy", y_test)

print("Fast preprocessing complete!")
print("Summary:")
print(f"   - Training samples: {len(X_train)}")
print(f"   - Test samples: {len(X_test)}")
print(f"   - Total samples: {len(X_train) + len(X_test)}") 