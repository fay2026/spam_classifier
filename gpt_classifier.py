import torch
import torch.nn as nn
from torch.nn import LayerNorm
import sys
import os

# Add parent directory to path to import GPTModel components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from transformer_block import TransformerBlock
    from config import model_config_small
except ImportError:
    print("Warning: Could not import transformer_block or config from parent directory")
    print("You may need to copy these files to the spam_classifier directory")

class GPTClassifier(nn.Module):
    """
    GPT-based model modified for text classification.
    Uses the same transformer architecture but adds a classification head.
    """
    def __init__(self, cfg, num_classes=2):
        super().__init__()
        
        # Core GPT components (same as GPTModel)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        
        # Classification head instead of language modeling head
        self.classifier = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"] // 2),
            nn.ReLU(),
            nn.Dropout(cfg["drop_rate"]),
            nn.Linear(cfg["emb_dim"] // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        tok_embeds = self.tok_emb(input_ids)
        
        # Position embeddings
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=input_ids.device)
        )
        
        # Combine embeddings
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        # Transform through blocks
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        
        # Global average pooling (or use [CLS] token approach)
        # Here we'll use mean pooling across sequence length
        if attention_mask is not None:
            # Mask out padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.size())
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # Simple mean pooling
            x = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(x)
        return logits
    
    def load_pretrained_weights(self, gpt_model_path):
        """
        Load weights from a pretrained GPTModel, excluding the output head and handling vocab size mismatch
        """
        print(f"Loading pretrained weights from {gpt_model_path}")
        checkpoint = torch.load(gpt_model_path, map_location='cpu')
        
        # Get state dict from checkpoint
        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        else:
            pretrained_dict = checkpoint
        
        # Get current model dict
        model_dict = self.state_dict()
        
        # Filter out weights that don't match or are not needed
        filtered_dict = {}
        skipped = []
        
        for k, v in pretrained_dict.items():
            if k in model_dict and 'out_head' not in k:  # Skip output head
                if model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                else:
                    # Handle embedding size mismatch
                    if 'tok_emb.weight' in k:
                        print(f"Vocabulary size mismatch: pretrained {v.shape[0]} vs current {model_dict[k].shape[0]}")
                        print("Initializing token embeddings randomly for new vocabulary")
                        # Don't load token embeddings if vocab size differs
                        skipped.append(k)
                    elif 'pos_emb.weight' in k:
                        print(f"Position embedding mismatch: pretrained {v.shape[0]} vs current {model_dict[k].shape[0]}")
                        # Truncate or pad position embeddings if needed
                        if v.shape[0] >= model_dict[k].shape[0]:
                            filtered_dict[k] = v[:model_dict[k].shape[0]]
                        else:
                            # If pretrained has fewer positions, pad with random values
                            new_pos_emb = torch.randn_like(model_dict[k])
                            new_pos_emb[:v.shape[0]] = v
                            filtered_dict[k] = new_pos_emb
                    else:
                        skipped.append(k)
            else:
                skipped.append(k)
        
        # Update current model dict
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict, strict=False)
        
        print(f"Successfully loaded {len(filtered_dict)} pretrained parameters")
        if skipped:
            print(f"Skipped {len(skipped)} parameters due to size mismatch or exclusion")
            
        return len(filtered_dict)

# Training function for the classifier
def train_gpt_classifier(model, train_loader, val_loader, num_epochs=3, lr=1e-5):
    """
    Fine-tune the GPT classifier on spam detection (Stable CPU version)
    """
    device = torch.device('cpu')  # Force CPU for Mac users
    model.to(device)
    
    print(f"Training on device: {device}")
    print(f"Using stable learning rate: {lr}")
    
    # Optimizer with more conservative settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
    criterion = nn.CrossEntropyLoss()
    
    # Add learning rate scheduler for stability
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    best_val_acc = 0
    patience_counter = 0
    max_patience = 5
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        nan_batches = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = (input_ids != 0).float()
            
            optimizer.zero_grad()
            
            try:
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected at batch {batch_idx+1}, skipping...")
                    nan_batches += 1
                    continue
                
                loss.backward()
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Check for NaN gradients
                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"Warning: NaN/Inf gradients detected at batch {batch_idx+1}, skipping...")
                    nan_batches += 1
                    continue
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
            except Exception as e:
                print(f"Error in batch {batch_idx+1}: {e}, skipping...")
                nan_batches += 1
                continue
            
            # Progress updates
            if batch_idx % 20 == 0 or batch_idx == len(train_loader) - 1:
                progress = (batch_idx + 1) / len(train_loader) * 100
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Batch {batch_idx+1}/{len(train_loader)} ({progress:.1f}%) - Loss: {loss.item():.4f} - LR: {current_lr:.2e}')
        
        # Check if too many NaN batches
        if nan_batches > len(train_loader) * 0.5:
            print(f"Too many NaN batches ({nan_batches}/{len(train_loader)}), stopping training...")
            break
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_nan_batches = 0
        
        print("Validating...")
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['text'].to(device)
                labels = batch['label'].to(device)
                attention_mask = (input_ids != 0).float()
                
                try:
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        val_nan_batches += 1
                        continue
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                except Exception:
                    val_nan_batches += 1
                    continue
        
        # Calculate metrics
        if train_total > 0:
            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / max(1, len(train_loader) - nan_batches)
        else:
            train_acc = 0
            avg_train_loss = float('inf')
        
        if val_total > 0:
            val_acc = 100 * val_correct / val_total
            avg_val_loss = val_loss / max(1, len(val_loader) - val_nan_batches)
        else:
            val_acc = 0
            avg_val_loss = float('inf')
        
        print(f'\nEpoch {epoch+1} Results:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        if nan_batches > 0:
            print(f'  Skipped {nan_batches} training batches due to NaN/Inf')
        if val_nan_batches > 0:
            print(f'  Skipped {val_nan_batches} validation batches due to NaN/Inf')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f'  🎉 New best validation accuracy: {best_val_acc:.2f}%')
        else:
            patience_counter += 1
            print(f'  Patience: {patience_counter}/{max_patience}')
        
        print('=' * 50)
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Return final losses along with the model
    final_train_loss = avg_train_loss if 'avg_train_loss' in locals() else 0.0
    final_val_loss = avg_val_loss if 'avg_val_loss' in locals() else 0.0
    
    return model, final_train_loss, final_val_loss, best_val_acc

if __name__ == "__main__":
    # Example usage
    print("GPTClassifier for spam detection created.")
    print("To use this:")
    print("1. Create the classifier with your config")
    print("2. Optionally load pretrained GPT weights")
    print("3. Train on your spam dataset using the data loaders") 