## 🔗 Live App: https://fnd-lstm.streamlit.app/

# ⚡ Fake News Detection Model

A lightweight LSTM-based fake news detection system that can classify news articles as real or fake. This project provides both pretrained models and the ability to train your own models from scratch.

## 🎯 Features

- **Lightweight LSTM Architecture**: Efficient neural network design for fast inference
- **Text Preprocessing**: Automated text cleaning and tokenization
- **Flexible Training**: Train your own model or use provided pretrained models
- **Web Interface**: Streamlit-based web application for easy testing
- **Balanced Dataset**: Uses WELFake dataset with balanced real/fake samples

## 📁 Project Structure

```
Source Code/
├── Models/                    # Model files directory
│   ├── lstm_model.h5         # Untrained model architecture
│   ├── lstm_trained.h5       # Pretrained model weights
│   ├── tokenizer.pkl         # Text tokenizer
│   ├── X_train_seq.npy       # Training sequences
│   ├── X_test_seq.npy        # Test sequences
│   ├── X_train.npy           # Raw training text
│   ├── X_test.npy            # Raw test text
│   ├── y_train.npy           # Training labels
│   └── y_test.npy            # Test labels
├── app.py                    # Streamlit web application
├── train.py                  # Model training script
├── evaluate.py               # Model evaluation script
├── model_building.py         # Model architecture definition
├── preprocessing.py           # Data preprocessing script
├── WELFake_Dataset.csv       # Original dataset
└── README.md                 # This file
```

## 🚀 Quick Start

### Option 1: Use Pretrained Models (Recommended for Quick Testing)

If you want to test the system immediately without training:

1. **Run the web application**:
   ```bash
   cd "Source Code"
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the provided URL
3. **Enter news text** and get instant predictions

The app will automatically load the pretrained model and tokenizer from the `Models/` directory.

### Option 2: Train Your Own Model

If you want to train a custom model or retrain with different parameters:

1. **Preprocess the data**:
   ```bash
   cd "Source Code"
   python preprocessing.py
   ```

2. **Build the model architecture**:
   ```bash
   python model_building.py
   ```

3. **Train the model**:
   ```bash
   python train.py
   ```

4. **Evaluate the model**:
   ```bash
   python evaluate.py
   ```

5. **Run the web app** with your trained model:
   ```bash
   streamlit run app.py
   ```

## 🔧 Model Architecture

- **Embedding Layer**: 64-dimensional word embeddings
- **LSTM Layer**: Single LSTM layer with 64 units
- **Dense Layers**: 32-unit hidden layer with dropout
- **Output**: Binary classification (Real/Fake)
- **Vocabulary Size**: 5,000 words
- **Sequence Length**: 200 tokens

## 📊 Dataset

- **Source**: WELFake dataset
- **Size**: 20,000 balanced samples (10,000 per class)
- **Split**: 80% training, 20% testing
- **Features**: Combined title and text content
- **Labels**: 0 (Fake), 1 (Real)

## 🛠️ Requirements

Install the required packages:

```bash
pip install tensorflow pandas numpy scikit-learn nltk streamlit matplotlib seaborn
```

## 📝 Usage Examples

### Web Application
The Streamlit app provides a user-friendly interface:
- Paste any news article or text
- Get instant real/fake classification
- View confidence scores and processing details

### Programmatic Usage
```python
from tensorflow.keras.models import load_model
import pickle
import numpy as np

# Load model and tokenizer
model = load_model("Models/lstm_trained.h5")
with open("Models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Make predictions
text = "Your news text here"
# ... preprocessing and prediction logic
```

## 🔍 Customization

You can modify various parameters in the scripts:
- **Vocabulary size**: Change `vocab_size` in `model_building.py`
- **Sequence length**: Adjust `max_length` for different text lengths
- **Model architecture**: Modify layers in `model_building.py`
- **Training parameters**: Adjust epochs, batch size in `train.py`

## 📚 File Descriptions

- **`preprocessing.py`**: Data loading, cleaning, and train-test splitting
- **`model_building.py`**: Defines LSTM architecture and creates tokenizer
- **`train.py`**: Trains the model with early stopping
- **`evaluate.py`**: Evaluates model performance and generates metrics
- **`app.py`**: Streamlit web interface for real-time predictions

## 🤝 Contributing

Feel free to:
- Experiment with different model architectures
- Try different preprocessing techniques
- Test with other datasets
- Improve the web interface

## 📄 License


This project is for educational and research purposes. Please ensure you have appropriate permissions for any datasets you use. 
