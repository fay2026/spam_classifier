import torch
import pandas as pd
import string
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from gpt_classifier import GPTClassifier

def preprocess_text(text):
    """Simple text preprocessing"""
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def load_trained_model(model_path="fast_gpt_spam_classifier.pth"):
    """Load the trained spam classifier"""
    print(f"Loading model from {model_path}")
    
    # Load checkpoint with weights_only=False for compatibility
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        checkpoint = torch.load(model_path, map_location='cpu')
    
    config = checkpoint['config']
    vocab_size = checkpoint['vocab_size']
    
    # Create model
    model = GPTClassifier(config, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Training accuracy: {checkpoint.get('test_accuracy', 'N/A'):.4f}")
    
    return model, checkpoint

def predict_single_message(model, text, vocab, max_length=64):
    """Predict if a single text message is spam or ham"""
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Convert to indices (using UNK for unknown words)
    indices = []
    for word in processed_text.split():
        indices.append(vocab.get(word, vocab.get('<UNK>', 1)))
    
    # Pad or truncate
    if len(indices) < max_length:
        indices.extend([vocab.get('<PAD>', 0)] * (max_length - len(indices)))
    else:
        indices = indices[:max_length]
    
    # Convert to tensor
    input_tensor = torch.tensor([indices], dtype=torch.long)
    attention_mask = (input_tensor != 0).float()
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(outputs, dim=1)
    
    # Return result
    spam_prob = probabilities[0][1].item()
    ham_prob = probabilities[0][0].item()
    is_spam = predicted[0].item() == 1
    
    return {
        'is_spam': is_spam,
        'spam_probability': spam_prob,
        'ham_probability': ham_prob,
        'prediction': 'SPAM' if is_spam else 'HAM'
    }

def test_on_new_dataset(model, checkpoint, messages, true_labels=None):
    """Test the model on a new dataset"""
    print(f"\nTesting on {len(messages)} new messages...")
    
    # Simple vocabulary reconstruction (limitation: we should save vocab with model)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    # Note: In production, you'd save the full vocabulary with the model
    
    predictions = []
    spam_probabilities = []
    
    for i, message in enumerate(messages):
        try:
            result = predict_single_message(model, message, vocab)
            predictions.append(1 if result['is_spam'] else 0)
            spam_probabilities.append(result['spam_probability'])
            
            if i % 100 == 0:
                print(f"Processed {i+1}/{len(messages)} messages")
                
        except Exception as e:
            print(f"Error processing message {i+1}: {e}")
            predictions.append(0)  # Default to HAM
            spam_probabilities.append(0.0)
    
    # Calculate metrics if true labels are provided
    if true_labels is not None:
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, 
                                     target_names=['HAM', 'SPAM'])
        cm = confusion_matrix(true_labels, predictions)
        
        print(f"\n=== NEW DATASET RESULTS ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(report)
        print(f"\nConfusion Matrix:")
        print("      HAM  SPAM")
        print(f"HAM   {cm[0][0]:4d} {cm[0][1]:4d}")
        print(f"SPAM  {cm[1][0]:4d} {cm[1][1]:4d}")
    
    return predictions, spam_probabilities

def load_csv_dataset(file_path, text_column, label_column=None):
    """Load a new dataset from CSV"""
    print(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)
    
    messages = df[text_column].tolist()
    
    if label_column and label_column in df.columns:
        # Assume labels are 'spam'/'ham' or 1/0
        labels = df[label_column].tolist()
        # Convert to numeric if needed
        if isinstance(labels[0], str):
            labels = [1 if label.lower() == 'spam' else 0 for label in labels]
        return messages, labels
    else:
        return messages, None

def test_example_messages():
    """Test on some example messages"""
    print("=== Testing on Example Messages ===")
    
    # Load model
    model, checkpoint = load_trained_model()
    vocab = {'<PAD>': 0, '<UNK>': 1}  # Simplified vocab
    
    # Example messages
    test_messages = [
        "Hi, how are you doing today?",
        "FREE! Win $1000 cash! Call now!",
        "Reminder: Your appointment is tomorrow at 3pm",
        "URGENT! Your account will be closed! Click here!",
        "Thanks for the great dinner last night",
        "Congratulations! You've won a free iPhone! Claim now!",
        "Can you pick up milk on your way home?",
        "Limited time offer! 50% off everything!",
        "See you at the meeting tomorrow",
        "Your bank account has been suspended. Verify now!"
    ]
    
    expected_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0=HAM, 1=SPAM
    
    print(f"\nTesting {len(test_messages)} example messages:")
    print("-" * 60)
    
    predictions = []
    for i, message in enumerate(test_messages):
        result = predict_single_message(model, message, vocab)
        predictions.append(1 if result['is_spam'] else 0)
        
        status = "✅" if predictions[i] == expected_labels[i] else "❌"
        print(f"{status} {result['prediction']:4s} ({result['spam_probability']:.2f}) | {message}")
    
    accuracy = accuracy_score(expected_labels, predictions)
    print(f"\nExample Test Accuracy: {accuracy:.2f}")

def main():
    """Main function to demonstrate usage"""
    print("🔍 Testing Fine-tuned Spam Classifier on New Data")
    print("="*50)
    
    choice = input("""
Choose testing option:
1. Test on example messages
2. Test on your own CSV file
3. Test individual messages interactively

Enter choice (1-3): """).strip()
    
    if choice == "1":
        test_example_messages()
        
    elif choice == "2":
        csv_path = input("Enter path to your CSV file: ").strip()
        text_col = input("Enter column name containing text messages: ").strip()
        label_col = input("Enter column name for labels (or press Enter if no labels): ").strip()
        label_col = label_col if label_col else None
        
        try:
            messages, labels = load_csv_dataset(csv_path, text_col, label_col)
            model, checkpoint = load_trained_model()
            predictions, probabilities = test_on_new_dataset(model, checkpoint, messages, labels)
            
            # Save results
            results_df = pd.DataFrame({
                'message': messages,
                'predicted_label': ['SPAM' if p == 1 else 'HAM' for p in predictions],
                'spam_probability': probabilities
            })
            if labels:
                results_df['true_label'] = ['SPAM' if l == 1 else 'HAM' for l in labels]
            
            results_df.to_csv('prediction_results.csv', index=False)
            print(f"\nResults saved to prediction_results.csv")
            
        except Exception as e:
            print(f"Error: {e}")
            
    elif choice == "3":
        model, checkpoint = load_trained_model()
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
        print("\nEnter messages to classify (type 'quit' to exit):")
        while True:
            message = input("\nMessage: ").strip()
            if message.lower() in ['quit', 'exit', 'q']:
                break
            if message:
                result = predict_single_message(model, message, vocab)
                icon = "🚨" if result['is_spam'] else "✅"
                print(f"{icon} {result['prediction']} (confidence: {max(result['spam_probability'], result['ham_probability']):.2f})")
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main() 