import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

import os
import numpy as np
from timeit import default_timer as timer

from models.cpc import CPC, SpkClassifier
from train.cpcPreTrain import ScheduledOptim


class CPCWithClassifier(nn.Module):
    def __init__(self, cpc_model, classifier):
        super().__init__()
        self.cpc_model = cpc_model
        self.classifier = classifier

    def forward(self, x):
        with torch.no_grad():
            output, c = self.cpc_model.predict(x)
        logits = self.classifier(c)
        return logits


def train_ft(model, train_loader, optimizer, device, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, labels) in enumerate(train_loader):
        # data: [batch]
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # CrossEntropyLoss
        outputs = model(data)
        loss = criterion(outputs, labels)

        # accuracy
        predictions = torch.argmax(outputs, dim=-1)
        correct = (predictions == labels)  # shape = (batch_size,)
        acc = correct.float().mean().item()

        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tlr:{lr:.5f}\tAccuracy: {acc:.4f}\tLoss: {loss.item():.6f}')


def val_ft(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_acc = 0

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for (data, labels) in val_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            total_loss += len(data) * criterion(outputs, labels)
            # accuracy
            predictions = torch.argmax(outputs, dim=-1)
            correct = (predictions == labels)  # shape = (batch_size,)
            total_acc += correct.float().sum().item()

    total_loss /= len(val_loader.dataset)  # average loss
    total_acc /= len(val_loader.dataset)

    print('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
        total_loss, total_acc))
    return total_loss


def fine_tuning(train, val, channels, spk_num, n_warmup_steps, batch_size, epochs, PreTrainModel, name):
    # path to save model
    model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trained_models')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = os.path.join(model_save_dir, f"best_model({name}).pth")

    seq_len = train['samples'].shape[-1]
    train = TensorDataset(train['samples'].float(), train['labels'].long())
    val = TensorDataset(val['samples'].float(), val['labels'].long())
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

    # Load pre-trained CPC model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpc_model = CPC(channels=channels, seq_len=seq_len)
    pre_train_path = os.path.join(model_save_dir,f"best_model({PreTrainModel}).pth")
    cpc_model.load_state_dict(torch.load(pre_train_path,weights_only=True))
    cpc_model.eval()

    spk_model = SpkClassifier(spk_num=spk_num)
    model = CPCWithClassifier(cpc_model, spk_model).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        n_warmup_steps=n_warmup_steps)

    # training
    best_loss = np.inf
    best_epoch = -1
    global_timer = timer()

    for epoch in range(1, epochs + 1):
        epoch_timer = timer()
        train_ft(model, train_loader, optimizer, device, epoch)
        val_loss = val_ft(model, val_loader, device)
        if val_loss < best_loss:
            best_acc = max(val_loss, best_loss)
            torch.save(model.state_dict(), model_save_path)
            best_epoch = epoch

        elif epoch - best_epoch > 2:
            optimizer.increase_delta()

        end_epoch_timer = timer()
        print("#### End epoch {}/{}, elapsed time: {}".format(epoch, epochs, end_epoch_timer - epoch_timer))

        ## end
    end_global_timer = timer()
    print("################## Success #########################")
    print("Total elapsed time: %s" % (end_global_timer - global_timer))

if __name__ == '__main__':
    train = torch.load('../data/Har/train.pt', weights_only=True)
    val = torch.load('../data/Har/val.pt', weights_only=True)
