import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import string

# Load and split data (same as before)
data_file_path = "SMSSpamCollection"
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

temp_train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label'])
train_df, val_df = train_test_split(temp_train_df, test_size=0.125, random_state=42, stratify=temp_train_df['Label'])

print(f"Training set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples") 
print(f"Testing set: {len(test_df)} samples")

# Text preprocessing function
def preprocess_text(text):
    """Simple text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Apply preprocessing
train_df['Text'] = train_df['Text'].apply(preprocess_text)
val_df['Text'] = val_df['Text'].apply(preprocess_text)
test_df['Text'] = test_df['Text'].apply(preprocess_text)

# Encode labels (spam=1, ham=0)
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['Label'])
val_labels = label_encoder.transform(val_df['Label'])
test_labels = label_encoder.transform(test_df['Label'])

print(f"Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# ===== APPROACH 1: For Traditional ML Models =====
print("\n=== Traditional ML Approach (TF-IDF + Logistic Regression) ===")

# Create TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['Text'])
X_val_tfidf = tfidf_vectorizer.transform(val_df['Text'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['Text'])

print(f"TF-IDF feature shape: {X_train_tfidf.shape}")

# Train a simple logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_tfidf, train_labels)

# Evaluate
val_pred = lr_model.predict(X_val_tfidf)
val_accuracy = accuracy_score(val_labels, val_pred)
print(f"Validation accuracy: {val_accuracy:.4f}")

# ===== APPROACH 2: For PyTorch Deep Learning Models =====
print("\n=== PyTorch DataLoader Approach ===")

class SMSDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_length=100):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = self.build_vocab(texts)
        else:
            self.vocab = vocab
            
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for text in texts:
            for word in text.split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab
    
    def text_to_indices(self, text):
        """Convert text to indices using vocabulary"""
        indices = []
        for word in text.split():
            indices.append(self.vocab.get(word, self.vocab['<UNK>']))
        
        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            indices.extend([self.vocab['<PAD>']] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]
            
        return indices
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels[idx]
        
        # Convert text to tensor
        text_indices = self.text_to_indices(text)
        
        return {
            'text': torch.tensor(text_indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
train_dataset = SMSDataset(train_df['Text'], train_labels)
val_dataset = SMSDataset(val_df['Text'], val_labels, vocab=train_dataset.vocab)
test_dataset = SMSDataset(test_df['Text'], test_labels, vocab=train_dataset.vocab)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Vocabulary size: {len(train_dataset.vocab)}")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Test the data loader
print("\nSample batch from train_loader:")
for batch in train_loader:
    print(f"Text tensor shape: {batch['text'].shape}")
    print(f"Label tensor shape: {batch['label'].shape}")
    print(f"Sample text indices (first 10): {batch['text'][0][:10].tolist()}")
    print(f"Sample label: {batch['label'][0].item()}")
    break 