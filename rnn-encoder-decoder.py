import torch
import torch.nn as nn
import random

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1, dropout=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, n_layers=n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x -> [batch size, sequence len, input_size]
        embedded = self.dropout(torch.relu(self.linear(x)))  # [batch size, sequence len, embedding size]
        output, (ht, ct) = self.lstm(embedded)
        # output -> [batch size, sequence len, hidden size]
        # hidden = [1, batch size, hidden size]
        # cell = [1, batch size, hidden size]
        # 1 -> num_layers
        return ht, ct


class LSTMDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, embedding_size, n_layers=1, dropout=0):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(output_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, n_layers=n_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, ht, ct):
        x = x.unsqueeze(1)  # [batch size, output size] -> [batch size, 1, output size]
        embedded = self.dropout(torch.relu(self.embedding(x)))  # [batch size, 1, embedding_size]
        output, _ = self.lstm(embedded, (ht, ct))  # [batch size, 1, hidden_size]
        prediction = self.linear(output.squeeze(1))
        return prediction


class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_size, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device  # 'cuda:0' / 'cpu'

        assert encoder.hidden_size == decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, x, y, teacher_forcing_ratio = 1):
        # x = [batch size, observed sequence len, input size]
        # y = [batch size, target sequence len, output size]
        batch_size = x.shape[0]
        target_len = y.shape[1]

        # tensor to store decoder outputs of each time step
        outputs = torch.zeros(y.shape).to(self.device)

        ht, ct = self.encoder(x)

        decoder_input = x[:, -1, :]  # [batch size, 1, input size]

        for i in range(target_len):
            output, ht, ct = self.decoder(decoder_input, ht, ct)

            outputs[i] = output

            teacher_forcing = random.random() < teacher_forcing_ratio # 1 use true label, 0 use predicted label

        return outputs


###############################Parameters#################################
input_size, embedding_size_en, hidden_size_en = 2, 128, 256
output_size, embedding_size_de, hidden_size_de = 2, 128, 256
# en_layers, de_layers, en_dropout, de_dropout

enc = LSTMEncoder(input_size, embedding_size_en, hidden_size_en)
dec = LSTMDecoder(output_size, embedding_size_de, hidden_size_de)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LSTMEncoderDecoder(enc, dec, dev).to(dev)
