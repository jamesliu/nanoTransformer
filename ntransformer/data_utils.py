import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader

# Step 1: Prepare the data and the dataloaders

# Define the tokenizer
tokenizer = get_tokenizer('basic_english')

# Load the IMDB dataset
train_iter, test_iter = IMDB()

# Create a function to yield the tokens
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# Build the vocabulary
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Function to process the data
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]

# Padding function to ensure all sequences in a batch have the same length
def pad_sequences(sequences, max_length):
    # Assuming sequences is a list of lists (each inner list is a sequence of token IDs)
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            # Pad sequences shorter than max_length
            padded_sequences.append(seq + [vocab['<pad>']] * (max_length - len(seq)))
        else:
            # Truncate sequences longer than max_length
            padded_sequences.append(seq[:max_length])
    return padded_sequences

# Function to create data loaders
def create_data_loaders(batch_size=128):
    # Collate function to process batches
    def collate_batch(batch):
        label_list, text_list, lengths = [], [], []
        for (_label, _text) in batch:
            label_list.append(int(_label))
            processed_text = text_pipeline(_text)
            text_list.append(processed_text)
            lengths.append(len(processed_text))
        max_length = max(lengths)  # Get the length of the longest sequence
        #print('max length: ', max_length)
        padded_text_list = pad_sequences(text_list, max_length)
        attention_masks = [[float(token != vocab['<pad>'])* -1e9 for token in seq] for seq in padded_text_list]  # 1 for actual tokens, 0 for pads
        return torch.tensor(label_list, dtype=torch.int64), torch.tensor(padded_text_list, dtype=torch.int64), torch.tensor(attention_masks, dtype=torch.float32)

    # Create data loaders
    train_loader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(list(test_iter), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, test_loader

if __name__ == "__main__":
    # Usage
    batch_size = 128
    train_loader, test_loader = create_data_loaders(batch_size)
    
    # Now, train_loader and test_loader can be used in the training loop of a model.
    for label, text, attention_mask in train_loader:
        # Here, you can pass your batch of `text`, `attention_mask`, and `label` through the model for training, validation, etc.
        #print(f"Text shape: {text.shape}, Attention mask shape: {attention_mask.shape}, Label shape: {label.shape}")
        #print(label)
        #print(text)
        #print(attention_mask)
        pass
