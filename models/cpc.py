import torch
import torch.nn as nn


class CPC(nn.Module):
    def __init__(self, channels, seq_len):
        '''
        :param timestep: number of future time steps for prediction
        :param channels: input size
        :param seq_len: time series length
        '''
        super().__init__()
        self.timestep = seq_len // 2
        self.channels = channels
        self.seq_len = seq_len

        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(16, 8, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(8, 16, bias=False) for i in range(self.timestep)])
        self.softmax = nn.Softmax(dim=-1)
        self.lsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # x: [batch_size, channels, seq_len]
        batch_size = x.shape[0]
        z = self.encoder(x)
        z = z.transpose(1, 2)
        # z: [batch_size, seq_len, enc_channels]
        nce = 0  # average over timestep and batch

        encode_samples = z[:, self.timestep:, :].transpose(0, 1)  # [pred_len, batch_size, enc_channels]
        forward_seq = z[:, :self.timestep, :]
        output, hidden = self.gru(forward_seq)
        ct = hidden.squeeze(0)  # [1, batch_size, gru_hidden] -> [batch_size, gru_hidden]
        pred = torch.empty((self.timestep, batch_size, 16)).float()  # [pred_len, batch_size, enc_channels]
        for i in range(self.timestep):
            # Predk+ct = Wk*ct
            pred[i] = self.Wk[i](ct)

        for i in range(self.timestep):
            total = encode_samples[i] @ pred[i].transpose(0, 1)  # [batch_size,batch_size]
            correct = torch.sum(
                torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch_size)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * batch_size * self.timestep
        accuracy = 1. * correct.item() / batch_size

        return accuracy, nce, hidden

    def predict(self, x):
        z = self.encoder(x)
        z = z.transpose(1, 2)
        output, hidden = self.gru(z)
        return output, hidden.squeeze(0)


class SpkClassifier(nn.Module):
    ''' linear classifier '''

    def __init__(self, spk_num):
        super(SpkClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, spk_num)
        )

    def forward(self, x):
        x = self.classifier(x)
        return torch.log_softmax(x, dim=-1)


if __name__ == '__main__':
    # Parameters
    batch_size = 8  # Number of samples in a batch
    channels = 1  # Number of input channels
    seq_len = 178  # Length of the time series
    spk_num = 1  # Number of speaker classes (for SpkClassifier)

    # Create synthetic data
    x = torch.randn(batch_size, channels, seq_len)  # Random input data

    # Initialize CPC model
    cpc_model = CPC(channels=channels, seq_len=seq_len)

    # Test CPC forward method
    accuracy, nce, hidden = cpc_model(x)
    print(f"CPC Forward Pass:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - NCE Loss: {nce:.4f}")
    print(f"  - Hidden State Shape: {hidden.shape}\n")

    # Test CPC predict method
    output, hidden = cpc_model.predict(x)
    print(f"CPC Prediction:")
    print(f"  - Output Shape: {output.shape}")
    print(f"  - Hidden State Shape: {hidden.shape}\n")
