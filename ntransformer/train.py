import torch
import torch.nn as nn
import torch.optim as optim
from .model import Transformer  # Ensure this is your transformer model file
from .data_utils import create_data_loaders  # This is based on your data_utils script

class TransformerWithEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim, num_layers, heads, num_classes):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=model_dim)
        self.transformer = Transformer(num_layers=num_layers, model_dim=model_dim, heads=heads)

        # New layers
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.classifier = nn.Linear(model_dim, num_classes)  # Classification head

    def forward(self, x, mask):
        x = self.embedding(x)  # Convert token ids to embeddings
        transformer_out = self.transformer(x, mask)

        # Pooling. Since the transformer outputs [batch, sequence, features], we pool over the sequence dimension
        pooled_out = self.pooling(transformer_out.transpose(1, 2))  # Switch dimensions for pooling
        pooled_out = pooled_out.squeeze(2)  # Remove the sequence length dimension

        # Classification
        logits = self.classifier(pooled_out)

        return logits
    
def train(model, iterator, optimizer, criterion, device):
    """
    This function defines a single training epoch.
    """
    model.train()
    epoch_loss = 0
    
    for _, (labels, text, attention_mask) in enumerate(iterator):
        #print("Text shape: ", text.shape)  # Add this to check the input shape
        #print("Labels shape: ", labels.shape)  # Optional, to check label shapes
        #print('Attention mask shape: ', attention_mask.shape)  # Optional, to check attention mask shapes
        text, labels, attention_mask = text.to(device), labels.to(device), attention_mask.to(device)  # Send data to device 
        # Generate attention masks based on input sequences
        # attention_mask = (text != 0).to(device)  # Assuming that '0' is the padding token in your vocab

        optimizer.zero_grad()
        if text.min() < 0 or text.max() >= model.vocab_size:
            raise ValueError(f"Text indices must be in range [0, {model.vocab_size}), but got min={text.min()} and max={text.max()}")
        output = model(text, mask=attention_mask)  # Ensure your model can handle the input dimensions
        #print("Output shape:", output.shape)
        #print("Labels shape:", labels.shape)
        # Inside your training loop, before calculating the loss
        assert labels.ge(0).all() and labels.lt(model.num_classes).all(), f"Labels outside range: {labels.min()} to {labels.max()}"

        loss = criterion(output.squeeze(), labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Specify the parameters
    num_layers = 2  # for example
    model_dim = 32  # for example, must match with the expected input size in your model
    heads = 2
    vocab_size = 101000  # For example. This should match your dataset's vocabulary size
    num_classes = 3  # Specify the correct number of classes based on your dataset

    # Initialize your model with the specified parameters
    model = TransformerWithEmbedding(vocab_size=vocab_size, model_dim=model_dim, 
                                     num_layers=num_layers, heads=heads, num_classes=num_classes).to(device)

    
    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # or another appropriate loss function
    optimizer = optim.Adam(model.parameters())

    # Load data
    batch_size = 16
    train_loader, test_loader = create_data_loaders(batch_size)

    # Training loop
    epochs = 10  # Define the number of epochs
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f'Epoch: {epoch:02}, Train Loss: {train_loss:.3f}')  # Add more logging info as needed

if __name__ == "__main__":
    main()
