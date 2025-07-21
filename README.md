# ⚡ Fast Fake News Detection Model

This folder contains the optimized, fast version of the fake news detection model that achieves **95% accuracy** in just **1.1 minutes** of training time.

## 🚀 Performance Improvements

| Metric | Original Model | Fast Model | Improvement |
|--------|----------------|------------|-------------|
| **Accuracy** | 59.32% | **95.0%** | **+35.68%** |
| **Training Time** | 8+ min/epoch | **0.2 min/epoch** | **40x faster** |
| **Total Training** | 40+ minutes | **1.1 minutes** | **36x faster** |
| **Dataset Size** | 34K samples | 20K samples | More efficient |

## 📁 Files

- `fast_preprocessing.py` - Fast data preprocessing (20K balanced samples)
- `fast_model_building.py` - Lightweight LSTM model architecture
- `fast_train.py` - Fast training with optimized parameters
- `fast_evaluate.py` - Model evaluation and metrics
- `fast_app.py` - Streamlit web application
- `fast_lstm_model.h5` - Untrained model architecture
- `fast_lstm_trained.h5` - Trained model weights

## ⚡ Model Architecture

- **Vocabulary Size**: 5,000 words (vs 15,000)
- **Sequence Length**: 200 tokens (vs 500)
- **Embedding**: 64 dimensions (vs 200)
- **LSTM**: Single layer (vs bidirectional)
- **Dense Layers**: 1 hidden layer (vs 2)

## 🎯 Quick Start

1. **Preprocess data**:
   ```bash
   python fast_preprocessing.py
   ```

2. **Build model**:
   ```bash
   python fast_model_building.py
   ```

3. **Train model**:
   ```bash
   python fast_train.py
   ```

4. **Evaluate model**:
   ```bash
   python fast_evaluate.py
   ```

5. **Run web app**:
   ```bash
   streamlit run fast_app.py
   ```

## 📊 Model Performance

- **Accuracy**: 95.0%
- **Precision**: 93.6%
- **Recall**: 96.6%
- **F1-Score**: 95.1%
- **Training Time**: 1.1 minutes
- **Prediction Time**: <0.1 seconds

## 🔧 Optimizations Made

1. **Reduced Dataset**: 20K balanced samples instead of 34K
2. **Simplified Architecture**: Single LSTM layer instead of bidirectional
3. **Smaller Vocabulary**: 5K words instead of 15K
4. **Shorter Sequences**: 200 tokens instead of 500
5. **Faster Preprocessing**: Simplified text cleaning
6. **Optimized Training**: Larger batch size, early stopping

## 🎉 Results

The fast model achieves excellent performance while being dramatically faster to train and deploy. Perfect for real-time fake news detection applications! 