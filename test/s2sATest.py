from torch.utils.data import Dataset, DataLoader
import time
from models.seq2seqAttention import *


class SimpleSeq2SeqDataset(Dataset):
    def __init__(self, num_samples=1000, src_len=10, tgt_len=12, vocab_size=100):
        self.num_samples = num_samples
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.vocab_size = vocab_size

        # Generate random sequence data
        self.src_data = torch.randint(1, vocab_size, (num_samples, src_len))  # exclude 0 (reserved for padding)
        self.tgt_data = torch.randint(1, vocab_size, (num_samples, tgt_len))

        # Define special tokens
        self.sos_token = 0  # start of sequence token
        self.eos_token = vocab_size - 1  # end of sequence token

        # Add start and end tokens to target sequence
        self.tgt_data = torch.cat([
            torch.full((num_samples, 1), self.sos_token),
            self.tgt_data,
            torch.full((num_samples, 1), self.eos_token)
        ], dim=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


def create_model(vocab_size, device):
    # Model hyperparameters
    embedding_size = 256
    encoder_hidden_size = 512
    decoder_hidden_size = 512
    dropout = 0.1

    # Create encoder
    encoder = Encoder(
        input_size=vocab_size,
        encoder_hidden_size=encoder_hidden_size,
        decoder_hidden_size=decoder_hidden_size,
        embedding_size=embedding_size,
        dropout=dropout
    )

    # Create attention layer
    attention = Attention(
        encoder_hidden_size=encoder_hidden_size,
        decoder_hidden_size=decoder_hidden_size
    )

    # Create decoder
    decoder = Decoder(
        output_size=vocab_size,
        decoder_hidden_size=decoder_hidden_size,
        embedding_size=embedding_size,
        attention=attention,
        dropout=dropout
    )

    # Create complete seq2seq model
    model = seq2seq(encoder, decoder, device)
    return model


def train_epoch(model, dataloader, optimizer, criterion, device, clip=1.0):
    """
    Train the model for one epoch

    Args:
        model: seq2seq model
        dataloader: DataLoader object containing training data
        optimizer: optimization algorithm
        criterion: loss function
        device: device to run the model on
        clip: gradient clipping value

    Returns:
        average loss for this epoch
    """
    model.train()
    epoch_loss = 0
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        # Transpose inputs to match model's expected shape
        output = model(src.transpose(0, 1), tgt.transpose(0, 1))

        # Calculate loss
        output = output[1:].view(-1, output.shape[-1])
        tgt = tgt.transpose(0, 1)[1:].reshape(-1)

        loss = criterion(output, tgt)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def main(device, vocab_size, batch_size, iterations, learning_rate):
    # # Set random seeds for reproducibility
    # torch.manual_seed(42)
    # random.seed(42)

    # Create training dataset and dataloader
    train_dataset = SimpleSeq2SeqDataset(vocab_size=vocab_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create validation dataset and dataloader
    val_dataset = SimpleSeq2SeqDataset(num_samples=200, vocab_size=vocab_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = create_model(vocab_size, device)
    model = model.to(device)

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding token

    # Training loop
    best_loss = float('inf')
    start_time = time.time()

    print("Starting training...")
    for epoch in range(iterations):
        # Train for one epoch
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)

        # Evaluate on validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_dataloader:
                src = src.to(device)
                tgt = tgt.to(device)

                output = model(src.transpose(0, 1), tgt.transpose(0, 1))
                output = output[1:].view(-1, vocab_size)
                tgt = tgt.transpose(0, 1)[1:].reshape(-1)

                loss = criterion(output, tgt)
                val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        # Print training statistics
        elapsed = time.time() - start_time
        print(f'Epoch: {epoch + 1}/{iterations}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Time: {elapsed:.2f}s')
        print('-' * 50)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = 100
    batch_size = 32
    iterations = 10
    learning_rate = 0.001
    main(device, vocab_size, batch_size, iterations, learning_rate)
