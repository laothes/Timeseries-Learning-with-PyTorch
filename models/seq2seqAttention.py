import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    def __init__(self, input_size, encoder_hidden_size, decoder_hidden_size, embedding_size, num_layers=2, dropout=0):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, encoder_hidden_size, batch_first=True, num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=True)
        self.linear = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)  # bidirectional -> 2 * hidden_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_x):
        # x -> [batch size, sequence len]
        embedded = self.dropout(self.embedding(enc_x))  # [batch size, sequence len, embedding size]
        output, (ht, ct) = self.lstm(embedded)
        # output -> [batch_size, seq_len, encoder_hidden_size * 2]
        # ht -> [n_layers * num_directions, batch_size, encoder_hidden_size]
        # ct -> [n_layers * num_directions, batch_size, encoder_hidden_size]
        forward_ht = ht[-2, :, :]  # [batch_size, encoder_hidden_size]
        backward_ht = ht[-1, :, :]  # [batch_size, encoder_hidden_size]
        ht_ = torch.cat((forward_ht, backward_ht), dim=1)  # [batch_size, encoder_hidden_size * 2]
        s0 = torch.tanh(self.linear(ht_))  # [1, batch_size, decoder_hidden_size]
        return output, s0


class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super().__init__()
        self.linear = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)

    def forward(self, s, enc_output):
        # s: [batch_size, decoder_hidden_size] -> [batch_size, 1, decoder_hidden_size]
        s = s.unsqueeze(1)
        # enc_output: [batch size, seq_len, encoder_hidden_size * 2] -> [batch size, seq_len, decoder_hidden_size]
        enc_output_ = self.linear(enc_output)
        e = s @ enc_output_.transpose(1, 2)  # [batch_size, 1, seq_len]
        alpha = torch.softmax(e.squeeze(1), dim=1)  # [batch_size, seq_len]
        alpha = alpha.unsqueeze(1)  # [batch_size, 1, seq_len]

        # [batch_size, 1, seq_len] @ [batch_size, seq_len, decoder_hidden_size]
        c = (alpha @ enc_output_).squeeze(1)  # [batch_size, decoder_hidden_size]

        return c


class Decoder(nn.Module):
    def __init__(self, output_size, decoder_hidden_size, embedding_size, attention, num_layers=1,
                 dropout=0):
        super().__init__()
        self.output_size = output_size
        self.attention = attention
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, decoder_hidden_size, batch_first=True, num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=False)
        self.linear = nn.Linear(2 * decoder_hidden_size + embedding_size, output_size)  # yt, st, c
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_x, st, enc_output):
        # dec_x: [batch size]
        # st: [batch size, decoder_hidden_size]
        # enc_output: [batch size, seq_len, encoder_hidden_size * 2]
        dec_x = dec_x.unsqueeze(1)  # [batch size, 1]
        embedded = self.dropout(self.embedding(dec_x))  # [batch size, 1, embedding size]
        c = self.attention(st, enc_output)  # [batch size, decoder_hidden_size]
        output, (st_, _) = self.lstm(embedded, (st.unsqueeze(0), c.unsqueeze(0)))
        # output -> [batch size, 1, decoder_hidden_size]
        # st_ -> [n_layers * num_directions, batch_size, encoder_hidden_size]
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        pred = self.linear(torch.cat((embedded, output, c), dim=1))  # yt, output, c
        # pred -> [batch size, output_size]
        return pred, st_[-1, :, :]


class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device  # 'cuda:0' / 'cpu'

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_size

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        enc_output, s = self.encoder(src.transpose(0, 1))
        dec_input = trg[0, :]
        for i in range(1, trg_len):
            dec_output, s = self.decoder(dec_input, s, enc_output)
            # dec_output -> [batch size, output_size]
            outputs[i] = dec_output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = dec_output.argmax(1)
            dec_input = trg[i] if teacher_force else top1
        return outputs


if __name__ == '__main__':
    pass
