# 🚀 GPT-Based SMS Spam Classifier Project

## 📋 Project Overview

A production-ready SMS spam classifier built using a modified GPT architecture, optimized for CPU training on Mac systems. Achieves **97.49% test accuracy** with efficient training in under 3 minutes.

## 🏆 Key Results

- **Test Accuracy**: 97.49%
- **Training Loss**: 0.0925
- **Validation Loss**: 0.1024
- **Test Loss**: 0.0746
- **Training Time**: < 3 minutes on Mac CPU
- **Model Size**: 4.48M parameters
- **Dataset**: UCI SMS Spam Collection (5,572 messages)

## 📁 Project Structure

```
spam_classifier/
├── fast_gpt_spam_classifier.pth     # Trained model (17MB)
├── train_gpt_spam_classifier.py     # Main training script
├── gpt_classifier.py               # GPTClassifier model class
├── test_new_dataset.py             # Basic testing script
├── test_new_dataset_improved.py    # Enhanced testing with better vocab
├── test_classifier.py              # Simple classifier testing
├── transformer_block.py            # Transformer block implementation
├── create_dataloaders.py           # Data loading utilities
├── load_sms_data.py                # SMS data loading functions
├── SMSSpamCollection               # Raw dataset (467KB)
├── sms_spam.zip                    # Compressed dataset
└── PROJECT_SUMMARY.md              # This documentation
```

## 🔧 Core Components

### 1. **GPTClassifier** (`gpt_classifier.py`)
- Modified GPT architecture for classification
- Uses transformer blocks with attention mechanism
- Classification head instead of language modeling head
- Supports pretrained weight loading with vocab mismatch handling
- Mean pooling for sequence-to-class prediction

### 2. **Training Pipeline** (`train_gpt_spam_classifier.py`)
- CPU-optimized training with stable loss computation
- Gradient clipping and NaN detection
- Early stopping and learning rate scheduling
- Comprehensive loss tracking (training, validation, test)
- Data preprocessing and vocabulary building

### 3. **Testing Infrastructure**
- **Basic Testing** (`test_new_dataset.py`): CSV file processing
- **Improved Testing** (`test_new_dataset_improved.py`): Enhanced vocabulary
- **Interactive Testing**: Real-time message classification

## ⚙️ Model Configuration

```python
# CPU-Optimized Configuration
config = {
    "vocab_size": 5002,        # Dataset-specific vocabulary
    "context_length": 64,      # Shorter for efficiency
    "emb_dim": 256,           # Reduced embedding dimension
    "n_heads": 4,             # Fewer attention heads
    "n_layers": 4,            # Fewer transformer layers
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False         # No bias in attention
}
```

## 🚀 Quick Start

### Training a New Model
```bash
cd spam_classifier
python train_gpt_spam_classifier.py
```

### Testing on New Messages
```bash
# Interactive testing
python test_new_dataset_improved.py

# CSV file testing
python test_new_dataset.py
```

### Loading and Using Trained Model
```python
import torch
from gpt_classifier import GPTClassifier

# Load model
checkpoint = torch.load('fast_gpt_spam_classifier.pth', map_location='cpu')
model = GPTClassifier(checkpoint['config'], num_classes=2)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
# ... (see test files for complete example)
```

## 📊 Training Features

### Stability Features
- **Conservative Learning Rate**: 1e-5 for stable convergence
- **Gradient Clipping**: Prevents gradient explosion
- **NaN Detection**: Automatically skips problematic batches
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive rate reduction

### CPU Optimizations
- **Reduced Model Size**: 4.48M vs 100M+ parameters
- **Smaller Batch Size**: 8 samples per batch
- **Limited Context**: 64 tokens max
- **Subset Training**: 50% of dataset for faster iteration
- **Efficient Vocabulary**: 5,000 most frequent words

## 📈 Performance Analysis

### Loss Progression
```
Epoch 1: Train: 0.4312 → Val: 0.2952
Epoch 2: Train: 0.2622 → Val: 0.1808
Epoch 3: Train: 0.1663 → Val: 0.1293
Epoch 4: Train: 0.1167 → Val: 0.1563
Epoch 5: Train: 0.0925 → Val: 0.1024
```

### Classification Report
```
              precision    recall  f1-score   support
         ham       0.98      0.99      0.99       484
        spam       0.95      0.85      0.90        74
    accuracy                           0.97       558
```

## 🔬 Architecture Details

### GPT Modifications for Classification
1. **Embedding Layer**: Token + positional embeddings
2. **Transformer Blocks**: Multi-head attention + feed-forward
3. **Mean Pooling**: Sequence-to-vector aggregation
4. **Classification Head**: 
   - Linear(256 → 128) + ReLU + Dropout
   - Linear(128 → 2) for binary classification

### Training Process
1. **Data Loading**: SMS messages → tokens → batches
2. **Forward Pass**: Text → embeddings → transformer → pooling → classification
3. **Loss Computation**: CrossEntropy loss with stability checks
4. **Optimization**: AdamW with weight decay and gradient clipping
5. **Validation**: Regular accuracy checking with early stopping

## 💾 Saved Model Contents

The `fast_gpt_spam_classifier.pth` file contains:
- `model_state_dict`: Trained model weights
- `config`: Model configuration
- `vocab_size`: Vocabulary size (5002)
- `test_accuracy`: Final test accuracy (0.9749)
- `test_loss`: Final test loss (0.0746)
- `label_encoder`: Label mapping (ham=0, spam=1)

## 🧪 Testing Examples

### Example Messages and Predictions
```
✅ HAM  (0.99) | Hi, how are you doing today?
✅ SPAM (0.95) | FREE! Win $1000 cash! Call now!
✅ HAM  (0.98) | Reminder: Your appointment is tomorrow at 3pm
✅ SPAM (0.92) | URGENT! Your account will be closed! Click here!
```

## 🔮 Future Improvements

1. **Vocabulary Enhancement**: Save complete training vocabulary
2. **Model Ensembling**: Combine multiple models for better accuracy
3. **Real-time Deployment**: API endpoint for live classification
4. **Multi-language Support**: Extend to non-English messages
5. **Advanced Preprocessing**: Better text normalization

## 🛠️ Dependencies

```
torch>=1.9.0
pandas>=1.3.0
scikit-learn>=1.0.0
numpy>=1.21.0
```

## 📝 Usage Notes

1. **Mac Compatibility**: Designed for Mac CPU training
2. **Memory Efficient**: Uses minimal RAM (<2GB)
3. **Fast Training**: Complete training in 2-3 minutes
4. **Production Ready**: Includes error handling and stability features
5. **Extensible**: Easy to modify for other text classification tasks

## 🎯 Use Cases

- **SMS Filtering**: Real-time spam detection
- **Email Classification**: Adapt for email spam detection
- **Text Moderation**: Modify for content filtering
- **Sentiment Analysis**: Retrain for sentiment classification
- **Document Classification**: Extend for longer text classification

---

**Created**: 2024  
**Status**: Production Ready  
**Accuracy**: 97.49%  
**Training Time**: <3 minutes on Mac CPU  
**Model Size**: 4.48M parameters 