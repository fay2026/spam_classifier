import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import string
import sys
import os
import torch.nn as nn

# Try to import required modules
try:
    from gpt_classifier import GPTClassifier, train_gpt_classifier
except ImportError:
    print("Error: Could not import gpt_classifier. Make sure gpt_classifier.py is in the same directory.")
    sys.exit(1)

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_pretrained_config():
    """Load the pretrained model configuration and optimize for CPU training"""
    config_path = "../pretrained_models/gpt_model_config.pt"
    if os.path.exists(config_path):
        print(f"Loading pretrained config from {config_path}")
        config = torch.load(config_path, map_location='cpu')
        
        # Optimize for CPU training - reduce model complexity
        print("Optimizing config for faster CPU training...")
        config["n_layers"] = min(4, config.get("n_layers", 12))  # Reduce layers
        config["n_heads"] = min(4, config.get("n_heads", 12))    # Reduce attention heads
        config["emb_dim"] = min(256, config.get("emb_dim", 768)) # Reduce embedding dimension
        config["context_length"] = 64  # Shorter context for faster processing
        
        return config
    else:
        print("Warning: Could not find pretrained config. Using CPU-optimized configuration.")
        return {
            "vocab_size": 8000,  # Will be updated based on vocabulary
            "context_length": 64,  # Shorter context
            "emb_dim": 256,        # Smaller embedding
            "n_heads": 4,          # Fewer heads
            "n_layers": 4,         # Fewer layers
            "drop_rate": 0.1,
            "qkv_bias": False
        }

class SMSDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_length=64):  # Reduced max_length
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = self.build_vocab(texts)
        else:
            self.vocab = vocab
            
    def build_vocab(self, texts):
        """Build vocabulary from texts with size limit for faster training"""
        vocab = {'<PAD>': 0, '<UNK>': 1}
        word_counts = {}
        
        # Count word frequencies
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Only keep most frequent words (limit vocab size for faster training)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        max_vocab_size = 5000  # Limit vocabulary size
        
        for word, count in sorted_words[:max_vocab_size]:
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

def preprocess_text(text):
    """Simple text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def prepare_data(use_subset=True):
    """Load and prepare the SMS spam dataset with option to use subset for faster training"""
    print("Loading SMS spam dataset...")
    
    # Load data
    data_file_path = "SMSSpamCollection"
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    
    # Use subset for faster training on CPU
    if use_subset:
        # Use smaller subset (50% of data) for faster training
        df = df.sample(frac=0.5, random_state=42).reset_index(drop=True)
        print(f"Using subset of data for faster training: {len(df)} samples")
    
    # Preprocess text
    df['Text'] = df['Text'].apply(preprocess_text)
    
    # Split data (same as in load_data.py)
    temp_train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label'])
    train_df, val_df = train_test_split(temp_train_df, test_size=0.125, random_state=42, stratify=temp_train_df['Label'])
    
    # Encode labels (spam=1, ham=0)
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df['Label'])
    val_labels = label_encoder.transform(val_df['Label'])
    test_labels = label_encoder.transform(test_df['Label'])
    
    print(f"Training set: {len(train_df)} samples ({len(train_df)/len(df):.1%})")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df):.1%})")
    print(f"Testing set: {len(test_df)} samples ({len(test_df)/len(df):.1%})")
    print(f"Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    return train_df, val_df, test_df, train_labels, val_labels, test_labels, label_encoder

def create_data_loaders(train_df, val_df, test_df, train_labels, val_labels, test_labels, batch_size=16):  # Smaller batch size
    """Create PyTorch data loaders optimized for CPU"""
    print("Creating data loaders...")
    
    # Create datasets
    train_dataset = SMSDataset(train_df['Text'], train_labels)
    val_dataset = SMSDataset(val_df['Text'], val_labels, vocab=train_dataset.vocab)
    test_dataset = SMSDataset(test_df['Text'], test_labels, vocab=train_dataset.vocab)
    
    # Create data loaders with smaller batch size for CPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # num_workers=0 for CPU
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    vocab_size = len(train_dataset.vocab)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, vocab_size

def evaluate_model(model, test_loader, label_encoder):
    """Evaluate the trained model on test set and calculate test loss"""
    device = torch.device('cpu')  # Force CPU for Mac users
    model.eval()
    
    all_predictions = []
    all_labels = []
    test_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['text'].to(device)
            labels = batch['label'].to(device)
            attention_mask = (input_ids != 0).float()
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, 
                                 target_names=label_encoder.classes_)
    
    print(f"\nTest Results:")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return accuracy, all_predictions, all_labels, avg_test_loss

def main():
    """Main training function optimized for CPU"""
    print("=== Fine-tuning GPT-based Spam Classifier (CPU Optimized) ===\n")
    
    # Load pretrained configuration (now CPU optimized)
    pretrained_config = load_pretrained_config()
    print(f"CPU-optimized config:")
    for key, value in pretrained_config.items():
        print(f"  {key}: {value}")
    
    # Prepare data (using subset for faster training)
    train_df, val_df, test_df, train_labels, val_labels, test_labels, label_encoder = prepare_data(use_subset=True)
    
    # Create data loaders (smaller batch size)
    train_loader, val_loader, test_loader, vocab_size = create_data_loaders(
        train_df, val_df, test_df, train_labels, val_labels, test_labels, batch_size=8  # Even smaller batch size
    )
    
    # Update config for classification task
    config = pretrained_config.copy()
    config["vocab_size"] = vocab_size  # Update vocab size for our dataset
    config["context_length"] = 64      # Shorter context for faster processing
    
    print(f"\nFinal model configuration for classification:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create model
    print("\nCreating GPTClassifier...")
    model = GPTClassifier(config, num_classes=2)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load pretrained weights (optional, may skip due to architecture differences)
    pretrained_weights_path = "../pretrained_models/gpt_model_weights.pt"
    if os.path.exists(pretrained_weights_path):
        try:
            print(f"\nAttempting to load pretrained weights...")
            num_loaded = model.load_pretrained_weights(pretrained_weights_path)
            print(f"✅ Successfully loaded {num_loaded} pretrained parameters")
        except Exception as e:
            print(f"⚠️  Could not load pretrained weights: {e}")
            print("Training from scratch (this is fine and may be faster)...")
    else:
        print("❌ No pretrained weights found. Training from scratch...")
    
    # Train the model with CPU-optimized settings
    print("\nStarting stable fine-tuning...")
    print("Stable Training Features:")
    print("- Lower learning rate (1e-5) for stability")
    print("- Gradient clipping to prevent explosions")
    print("- NaN detection and skipping")
    print("- Learning rate scheduling")
    print("- Early stopping based on validation accuracy")
    print("- Conservative optimizer settings")
    
    # Initialize model weights properly
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    print("Applied proper weight initialization")
    
    trained_model, final_train_loss, final_val_loss, best_val_acc = train_gpt_classifier(
        model, train_loader, val_loader, 
        num_epochs=5,   # More epochs with early stopping
        lr=1e-5         # Very stable learning rate
    )
    
    # Display comprehensive loss information
    print(f"\n{'='*60}")
    print("TRAINING RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Final Training Loss:   {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Best Validation Acc:   {best_val_acc:.2f}%")
    print(f"{'='*60}")
    
    # Evaluate on test set
    test_accuracy, predictions, true_labels, test_loss = evaluate_model(trained_model, test_loader, label_encoder)
    
    # Save the fine-tuned model
    save_path = "fast_gpt_spam_classifier.pth"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config,
        'vocab_size': vocab_size,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'label_encoder': label_encoder
    }, save_path)
    print(f"\nFine-tuned model saved to {save_path}")
    
    print("\n=== Fast Fine-tuning Complete ===")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Training completed in minimal time on CPU!")
    
    # Final comprehensive summary
    print(f"\n{'='*70}")
    print("FINAL LOSS SUMMARY")
    print(f"{'='*70}")
    print(f"Training Loss:     {final_train_loss:.4f}")
    print(f"Validation Loss:   {final_val_loss:.4f}")
    print(f"Test Loss:         {test_loss:.4f}")
    print(f"{'='*70}")
    print(f"Test Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*70}")

if __name__ == "__main__":
    main() 