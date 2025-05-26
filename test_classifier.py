import torch
import string
from gpt_classifier import GPTClassifier

def preprocess_text(text):
    """Simple text preprocessing"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def load_trained_model(model_path="fast_gpt_spam_classifier.pth"):
    """Load the trained spam classifier"""
    print(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    vocab_size = checkpoint['vocab_size']
    
    # Create model
    model = GPTClassifier(config, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Test accuracy: {checkpoint.get('test_accuracy', 'N/A'):.4f}")
    
    return model, checkpoint.get('label_encoder')

def predict_spam(model, text, vocab, max_length=64):
    """Predict if a text message is spam or ham"""
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Convert to indices
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

def interactive_test():
    """Interactive testing function"""
    try:
        # Load model
        model, label_encoder = load_trained_model()
        
        # We need to reconstruct the vocabulary - this is a limitation
        # In production, you'd save the vocabulary with the model
        print("\n⚠️  Note: Using simplified vocabulary for demo")
        print("For production use, save the vocabulary with the model")
        
        # Simple vocabulary for testing
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
        print("\n" + "="*50)
        print("SMS SPAM CLASSIFIER - Interactive Test")
        print("="*50)
        print("Enter SMS messages to classify (type 'quit' to exit)")
        
        while True:
            text = input("\nEnter SMS message: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            try:
                result = predict_spam(model, text, vocab)
                
                print(f"\nText: '{text}'")
                print(f"Prediction: {result['prediction']}")
                print(f"Ham probability: {result['ham_probability']:.3f}")
                print(f"Spam probability: {result['spam_probability']:.3f}")
                
                if result['is_spam']:
                    print("🚨 This message is classified as SPAM")
                else:
                    print("✅ This message is classified as HAM (normal)")
                    
            except Exception as e:
                print(f"Error processing message: {e}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you've trained the model first by running train_gpt_spam_classifier.py")

if __name__ == "__main__":
    interactive_test() 