# train.py - Training with optimized parameters

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import time

print("Starting model training...")

# Load data
X_train = np.load("Models/X_train_seq.npy")
y_train = np.load("Models/y_train.npy")

print(f"Training data shape: {X_train.shape}")
print(f"Labels shape: {y_train.shape}")

# Load untrained model
model = load_model("Models/lstm_model.h5")

# Compile model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("Model compiled successfully")

# Simple early stopping callback
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]

print("Training with early stopping (patience=3)")

# Training parameters
start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    epochs=10,           # Reduced epochs
    batch_size=128,      # Larger batch size for speed
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

end_time = time.time()
training_time = end_time - start_time

# Save trained model
model.save("Models/lstm_trained.h5")
print(f"Model saved as Models/lstm_trained.h5")

# Print training summary
final_accuracy = max(history.history['val_accuracy'])
final_loss = min(history.history['val_loss'])
epochs_trained = len(history.history['accuracy'])

print(f"\nTraining completed!")
print(f"Total training time: {training_time/60:.1f} minutes")
print(f"Best validation accuracy: {final_accuracy:.4f}")
print(f"Best validation loss: {final_loss:.4f}")
print(f"Epochs trained: {epochs_trained}")
print(f"Average time per epoch: {training_time/epochs_trained/60:.1f} minutes") 