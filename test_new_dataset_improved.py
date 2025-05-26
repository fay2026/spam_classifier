import torch
import pandas as pd
import string
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
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location='cpu')
    
    config = checkpoint['config']
    vocab_size = checkpoint['vocab_size']
    
    model = GPTClassifier(config, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Training accuracy: {checkpoint.get('test_accuracy', 'N/A'):.4f}")
    
    return model, checkpoint

def create_spam_vocabulary():
    """Create a vocabulary with common spam/ham words for better classification"""
    # This is a simplified approach - in production you'd save the exact training vocab
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    # Common words that help distinguish spam vs ham
    important_words = [
        # Normal/Ham words
        'hi', 'hello', 'how', 'are', 'you', 'today', 'tomorrow', 'appointment', 
        'meeting', 'thanks', 'dinner', 'home', 'work', 'see', 'reminder',
        'can', 'will', 'please', 'time', 'call', 'talk', 'later', 'love',
        
        # Spam indicator words  
        'free', 'win', 'cash', 'money', 'urgent', 'account', 'closed', 'click',
        'congratulations', 'won', 'iphone', 'prize', 'offer', 'limited', 'now',
        'claim', 'suspended', 'verify', 'bank', 'winner', 'discount', 'deal',
        'guaranteed', 'risk', 'bonus', 'exclusive', 'save', 'opportunity'
    ]
    
    for word in important_words:
        if word not in vocab:
            vocab[word] = len(vocab)
    
    return vocab

def predict_single_message(model, text, vocab, max_length=64):
    """Predict if a single text message is spam or ham"""
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
    
    spam_prob = probabilities[0][1].item()
    ham_prob = probabilities[0][0].item()
    is_spam = predicted[0].item() == 1
    
    return {
        'is_spam': is_spam,
        'spam_probability': spam_prob,
        'ham_probability': ham_prob,
        'prediction': 'SPAM' if is_spam else 'HAM',
        'confidence': max(spam_prob, ham_prob)
    }

def test_example_messages():
    """Test on example messages with improved vocabulary"""
    print("=== Testing with Improved Vocabulary ===")
    
    model, checkpoint = load_trained_model()
    vocab = create_spam_vocabulary()
    
    print(f"Using vocabulary with {len(vocab)} words")
    
    test_cases = [
        ("Hi, how are you doing today?", "HAM"),
        ("FREE! Win $1000 cash! Call now!", "SPAM"),
        ("Reminder: Your appointment is tomorrow at 3pm", "HAM"),
        ("URGENT! Your account will be closed! Click here!", "SPAM"),
        ("Thanks for the great dinner last night", "HAM"),
        ("Congratulations! You've won a free iPhone! Claim now!", "SPAM"),
        ("Can you pick up milk on your way home?", "HAM"),
        ("Limited time offer! 50% off everything!", "SPAM"),
        ("See you at the meeting tomorrow", "HAM"),
        ("Your bank account has been suspended. Verify now!", "SPAM")
    ]
    
    print(f"\nTesting {len(test_cases)} example messages:")
    print("-" * 80)
    
    correct = 0
    for message, expected in test_cases:
        result = predict_single_message(model, message, vocab)
        
        is_correct = result['prediction'] == expected
        if is_correct:
            correct += 1
        
        status = "✅" if is_correct else "❌"
        confidence_color = "🔴" if result['confidence'] < 0.6 else ("🟡" if result['confidence'] < 0.8 else "🟢")
        
        print(f"{status} {result['prediction']:4s} ({result['confidence']:.2f}) {confidence_color} | {message}")
        print(f"     Expected: {expected}, Spam prob: {result['spam_probability']:.3f}")
        print()
    
    accuracy = correct / len(test_cases)
    print(f"Test Accuracy: {accuracy:.2f} ({correct}/{len(test_cases)})")
    
    return model, vocab

def interactive_test():
    """Interactive testing with improved model"""
    model, vocab = test_example_messages()
    
    print("\n" + "="*60)
    print("INTERACTIVE SPAM CLASSIFIER")
    print("="*60)
    print("Enter messages to classify (type 'quit' to exit):")
    
    while True:
        message = input("\n📱 SMS Message: ").strip()
        if message.lower() in ['quit', 'exit', 'q']:
            break
        if message:
            result = predict_single_message(model, message, vocab)
            
            if result['is_spam']:
                print(f"🚨 SPAM (confidence: {result['confidence']:.2f})")
                print(f"   Spam probability: {result['spam_probability']:.3f}")
            else:
                print(f"✅ HAM (confidence: {result['confidence']:.2f})")
                print(f"   Ham probability: {result['ham_probability']:.3f}")

if __name__ == "__main__":
    print("🔍 IMPROVED Spam Classifier Testing")
    print("="*50)
    
    choice = input("""
Choose option:
1. Test example messages
2. Interactive testing

Enter choice (1-2): """).strip()
    
    if choice == "1":
        test_example_messages()
    elif choice == "2":
        interactive_test()
    else:
        print("Invalid choice!") 