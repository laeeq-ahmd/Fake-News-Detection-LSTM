# model_building.py - Lightweight LSTM model

import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

print("Building lightweight model...")

# Load training and testing text
X_train = np.load("Models/X_train.npy", allow_pickle=True)
X_test = np.load("Models/X_test.npy", allow_pickle=True)

# ⚡ OPTIMIZED settings for speed
vocab_size = 5000   # Reduced from 15000
max_length = 200    # Reduced from 500

print(f"Training samples: {len(X_train)}")
print(f"Creating tokenizer with vocab_size={vocab_size}, max_length={max_length}")

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert text to padded sequences
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_length, padding='post', truncating='post')
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_length, padding='post', truncating='post')

print(f"Sequence shapes - Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")

# Save sequences
np.save("Models/X_train_seq.npy", X_train_seq)
np.save("Models/X_test_seq.npy", X_test_seq)

# Save tokenizer
with open("Models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer and sequences saved.")

# Build LIGHTWEIGHT model
print("Building lightweight model architecture...")

model = Sequential([
    # Smaller embedding
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
    
    # Single LSTM layer (not bidirectional)
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    
    # Single dense layer
    Dense(32, activation='relu'),
    Dropout(0.3),
    
    # Output layer
    Dense(1, activation='sigmoid')
])

    # Compile with optimizer
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Show model summary
model.summary()

# Save untrained model architecture
model.save("Models/lstm_model.h5")
print("Model architecture saved as Models/lstm_model.h5") 