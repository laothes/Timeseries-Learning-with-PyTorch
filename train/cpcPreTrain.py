import numpy as np
from timeit import default_timer as timer
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.cpc import CPC, SpkClassifier


class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


def train_one_epoch(model, train_loader, optimizer, device, epoch):
    model.train()

    for batch_idx, data in enumerate(train_loader):
        # data: [batch]
        data = data.to(device)
        optimizer.zero_grad()

        # only samples
        acc, nceloss, hidden = model(data)

        nceloss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tlr:{lr:.5f}\tAccuracy: {acc:.4f}\tLoss: {nceloss.item():.6f}')


def val_one_epoch(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)  # add channel dimension
            acc, loss, hidden = model(data)
            total_loss += len(data) * loss
            total_acc += len(data) * acc

    total_loss /= len(val_loader.dataset)  # average loss
    total_acc /= len(val_loader.dataset)  # average acc

    print('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
        total_loss, total_acc))
    return total_acc, total_loss


def pre_training(train, val, channels, n_warmup_steps, epochs, batch_size, name):
    # path to save model
    model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trained_models')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = os.path.join(model_save_dir, f"best_model({name}).pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CPCmodel = CPC(
        channels=channels,
        seq_len=train.shape[-1]
    ).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, CPCmodel.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        n_warmup_steps=n_warmup_steps)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

    model_params = sum(p.numel() for p in CPCmodel.parameters() if p.requires_grad)
    print('### Model summary below###\n {}\n'.format(str(CPCmodel)))
    print('===> Model total parameter: {}\n'.format(model_params))

    # training
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1
    global_timer = timer()

    for epoch in range(1, epochs + 1):
        epoch_timer = timer()
        train_one_epoch(CPCmodel, train_loader, optimizer, device, epoch)
        val_acc, val_loss = val_one_epoch(CPCmodel, val_loader, device)

        if val_acc > best_acc:
            best_acc = max(val_acc, best_acc)
            torch.save(CPCmodel.state_dict(), model_save_path)
            best_epoch = epoch

        elif epoch - best_epoch > 2:
            optimizer.increase_delta()

        end_epoch_timer = timer()
        print("#### End epoch {}/{}, elapsed time: {}".format(epoch, epochs, end_epoch_timer - epoch_timer))

    ## end
    end_global_timer = timer()
    print("################## Success #########################")
    print("Total elapsed time: %s" % (end_global_timer - global_timer))


if __name__ == "__main__":
    train = torch.load('../data/Epilepsy/train.pt', weights_only=True)['samples'].float()
    val = torch.load('../data/Epilepsy/val.pt', weights_only=True)['samples'].float()
    pre_training(train, val,
                 channels=1,
                 n_warmup_steps=1000,
                 epochs=100,
                 batch_size=256,
                 name='test'
                 )
