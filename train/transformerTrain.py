import numpy as np
from timeit import default_timer as timer
import os
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models.transformer import LearningPositionalEncoder


class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model, nhead, n_layers, output_size, dropout=0.1):
        super().__init__()
        self.fc_in = nn.Linear(input_size, d_model, bias=False)
        self.encoder = LearningPositionalEncoder(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transform = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, channels, seq_len]
        x = x.transpose(1, 2)  # x: [batch, seq_len, channels]
        x = self.fc_in(x)  # x: [batch, seq_len, d_model]
        x = self.encoder(x)
        x = self.transform(x)  # x: [batch, seq_len, d_model]
        x = self.fc_out(x)  # x: [batch, seq_len, output_size]
        
        # Apply pooling: Take only the first token (if using [CLS] token for classification)
        x = x[:, 0, :] # x: [batch, seq_len]
        return self.dropout(x)


def training(model, train_loader, optimizer, device, epoch):
    model.train()

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        # CrossEntropyLoss
        outputs = model(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # accuracy
        predictions = torch.argmax(outputs, dim=-1)
        correct = (predictions == labels)  # shape = (batch_size,)
        acc = correct.float().mean().item()

        loss.backward()
        optimizer.step()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if batch_idx % 10 == 0:
            logging.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                         f'({100. * batch_idx / len(train_loader):.0f}%)]\tAccuracy: {acc:.4f}\tLoss: {loss.item():.6f}')


def validating(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for (data, labels) in val_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            total_loss += len(data) * nn.CrossEntropyLoss()(outputs, labels)
            # accuracy
            predictions = torch.argmax(outputs, dim=-1)
            correct = (predictions == labels)  # shape = (batch_size,)
            total_acc += correct.float().sum().item()

    total_loss /= len(val_loader.dataset)  # average loss
    total_acc /= len(val_loader.dataset)

    logging.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
        total_loss, total_acc))
    return total_acc, total_loss


def save_checkpoint(model, optimizer, epoch, val_acc, val_loss, best_acc, save_dir):
    """Save the model checkpoint"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'best_acc': best_acc
    }

    # Save the latest checkpoints
    latest_path = os.path.join(save_dir, 'latest_checkpoint.pth')
    torch.save(state, latest_path)

    # If it is the best model, save an extra copy
    if val_acc >= best_acc:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_path)
        logging.info(f'Saved best model with accuracy: {val_acc:.4f}')


def main():
    parser = argparse.ArgumentParser(description='PyTorch Transformer Example')
    parser.add_argument('--train-raw', required=True)
    parser.add_argument('--validation-raw', required=True)
    parser.add_argument('--eval-raw')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--input-size', type=int, default=64, help='Number of input features (channels)')
    parser.add_argument('--d-model', type=int, default=512, help='Dimension of the model (d_model)')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=6, help='Number of transformer encoder layers')
    parser.add_argument('--output-size', type=int, default=10, help='Number of output classes')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--save-model', action='store_true', help='Flag to save the model after training')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('use_cuda is', use_cuda)

    global_timer = timer()  # global timer
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    model = TransformerClassifier(args.input_size, args.d_model, args.nhead, args.n_layers, args.output_size,
                                  args.dropout).to(device)

    logging.info('===> loading train, validation and eval dataset')
    train = torch.load(args.train_raw, weights_only=True)
    val = torch.load(args.validation_raw, weights_only=True)
    eval_set = torch.load(args.eval_raw, weights_only=True) if args.eval_raw else None

    train = TensorDataset(train['samples'].float(), train['labels'].long())
    val = TensorDataset(val['samples'].float(), val['labels'].long())
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('### Model summary below###\n {}\n'.format(str(model)))
    logging.info('===> Model total parameter: {}\n'.format(model_params))

    # Start training
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        epoch_timer = timer()
        training(model, train_loader, optimizer, device, epoch)
        val_acc, val_loss = validating(model, val_loader, device)
        if args.save_model:
            save_checkpoint(model, optimizer, epoch, val_acc, val_loss, best_acc, args.checkpoint_dir)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_loss = val_loss

        end_epoch_timer = timer()
        logging.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, args.epochs, end_epoch_timer - epoch_timer))

    # Save final model state if specified
    if args.save_model:
        final_state = {
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'best_epoch': best_epoch,
            'best_loss': best_loss
        }
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        torch.save(final_state, os.path.join(args.checkpoint_dir, 'final_transformer.pth'))
        logging.info(f'Saved final model with best accuracy: {best_acc:.4f} at epoch {best_epoch}')

    # end
    end_global_timer = timer()
    logging.info("################## Success #########################")
    logging.info("Total elapsed time: %s" % (end_global_timer - global_timer))

main()
