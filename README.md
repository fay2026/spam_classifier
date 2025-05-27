# 🚀 GPT-Based SMS Spam Classifier

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-97.49%25-green)](https://github.com/fay2026/spam_classifier)

A production-ready SMS spam classifier built using a modified GPT architecture, optimized for CPU training on Mac systems. Achieves **97.49% test accuracy** with efficient training in under 3 minutes.

## 🏆 Key Features

- **High Accuracy**: 97.49% test accuracy
- **Fast Training**: < 3 minutes on Mac CPU
- **Memory Efficient**: Uses minimal RAM (<2GB)
- **Production Ready**: Includes error handling and stability features
- **CPU Optimized**: Designed for Mac systems without GPU

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | 97.49% |
| Training Loss | 0.0925 |
| Validation Loss | 0.1024 |
| Test Loss | 0.0746 |
| Model Size | 4.48M parameters |
| Training Time | < 3 minutes |

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/fay2026/spam_classifier.git
cd spam_classifier
pip install -r requirements.txt
```

### Test the Pre-trained Model
```bash
python test_new_dataset_improved.py
# Choose option 2 for interactive testing
```

### Train Your Own Model
```bash
python train_gpt_spam_classifier.py
# Takes ~3 minutes, outputs all losses
```

### Test on Your Data
```bash
python test_new_dataset.py
# Follow prompts to specify your CSV file and columns
```

## 💡 Usage Example

```python
import torch
from gpt_classifier import GPTClassifier

# Load the trained model
checkpoint = torch.load('fast_gpt_spam_classifier.pth', map_location='cpu')
model = GPTClassifier(checkpoint['config'], num_classes=2)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Classify a message
message = "FREE! Win $1000 cash! Call now!"
# ... (see test files for complete classification code)
```

## 📁 Project Structure

```
├── fast_gpt_spam_classifier.pth     # Trained model (17MB)
├── train_gpt_spam_classifier.py     # Main training script
├── gpt_classifier.py               # GPTClassifier model class
├── test_new_dataset_improved.py    # Enhanced testing script
├── test_new_dataset.py             # Basic testing script
├── transformer_block.py            # Transformer implementation
├── SMSSpamCollection               # Dataset
├── PROJECT_SUMMARY.md              # Detailed documentation
├── QUICK_USAGE_GUIDE.md            # Quick reference
└── requirements.txt                # Dependencies
```

## 🏗️ Architecture

- **Base Model**: Modified GPT architecture
- **Embedding**: 256-dimensional token + positional embeddings
- **Transformer**: 4 layers, 4 attention heads
- **Classification Head**: Linear layers with ReLU and dropout
- **Pooling**: Mean pooling across sequence length

## 📈 Training Features

- **Stability**: Gradient clipping, NaN detection, early stopping
- **Optimization**: AdamW optimizer with learning rate scheduling
- **CPU Efficiency**: Reduced model size, optimized batch processing
- **Monitoring**: Comprehensive loss tracking and validation

## 🎯 Use Cases

- SMS spam filtering
- Email classification
- Text content moderation
- Sentiment analysis (with retraining)
- General text classification tasks

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- pandas
- scikit-learn
- numpy

## 📖 Documentation

- [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) - Complete technical documentation
- [`QUICK_USAGE_GUIDE.md`](QUICK_USAGE_GUIDE.md) - Fast reference guide

## 🚧 Future Improvements

- [ ] Vocabulary enhancement and persistence
- [ ] Model ensembling for higher accuracy
- [ ] REST API for real-time deployment
- [ ] Multi-language support
- [ ] Advanced text preprocessing

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Created**: 2024  
**Status**: Production Ready  
**Accuracy**: 97.49%  
**Training Time**: <3 minutes on Mac CPU
