import math
import torch
import torch.nn as nn
import torch.optim as optim


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(0) is PAD token
    # .detach() creates a new tensor that shares the same data but does not track gradients
    pad_attn_mask = seq_k.detach().eq(0).unsqueeze(1)
    # [batch_size, 1, len_k] -> [batch_size, len_q, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_seq_mask(seq):  # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # seq: [batch_size, tgt_len, tgt_len]
    # EXAMPLE
    # 0 1 1
    # 0 0 1
    # 0 0 0
    return subsequence_mask.bool().to(seq.device)


class InputEmbeddings(nn.Module):

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class LearningPositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model * 2, d_model)

        pe = torch.zeros(max_seq_len, d_model)  # [max_seq_len, d_model]
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # [max_seq_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).unsqueeze(
            0)  # [1, d_model/2]
        pe[:, 0::2] = torch.sin(pos @ div_term)  # even position
        pe[:, 1::2] = torch.cos(pos @ div_term)  # odd position
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_len, d_model]

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.size()
        pe = self.pe[:, :seq_len, :].expand(batch_size, -1,
                                            -1)  # [1, max_seq_len, d_model] -> [batch_size, seq_len, d_model]
        x = self.linear(torch.cat((x, pe), dim=-1))
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale_factor, dropout=0):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q: [batch_size, n_head, len_q, d_k]
        # k: [batch_size, n_head, len_k, d_k]
        # v: [batch_size, n_head, len_k, d_v]
        # softmax(q@kT/√d_k)v
        # Default: len_q,len_k = seq_len; d_k, d_v = 64
        # attn: [batch_size, n_head, len_q, len_k]
        attn = (q @ k.transpose(2, 3)) / self.scale_factor

        if mask is not None:
            # True in mask will be masked
            # 1) set padding score after softmax as 0
            # 2) Cover up unpredictable words in decoder
            # mask: [batch_size, 1, src_len, src_len]
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(torch.softmax(attn, dim=-1))
        heads = attn @ v  # [batch_size, n_head, len_q, d_v]
        return heads, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Parameter(torch.Tensor(d_model, n_head * d_k))
        self.w_k = nn.Parameter(torch.Tensor(d_model, n_head * d_k))
        self.w_v = nn.Parameter(torch.Tensor(d_model, n_head * d_v))
        self.w_o = nn.Parameter(torch.Tensor(d_model, n_head * d_v))
        # Initialize parameters
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)
        nn.init.xavier_uniform_(self.w_o)

        self.attention = ScaledDotProductAttention(scale_factor=math.sqrt(d_k), dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        # initial q k v: [batch_size， seq_len， d_model]
        batch_size, len_q, len_k = q.size(0), q.size(1), k.size(1)

        residual = q

        q = (q @ self.w_q).view(batch_size, len_q, self.n_head, self.d_k)
        k = (k @ self.w_k).view(batch_size, len_k, self.n_head, self.d_k)
        v = (v @ self.w_v).view(batch_size, len_k, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # q: [batch_size， seq_len， d_model] -> [batch_size， len_q， n_head, d_k] -> [batch_size, n_head, len_q, d_k]
        # k: [batch_size， seq_len， d_model] -> [batch_size， len_k， n_head, d_k] -> [batch_size, n_head, len_k, d_k]
        # v: [batch_size， seq_len， d_model] -> [batch_size， len_k， n_head, d_v] -> [batch_size, n_head, len_k, d_v]

        if mask is not None:
            # mask: [batch_size, src_len, src_len] -> [batch_size, 1, src_len, src_len]
            mask = mask.unsqueeze(1)

        heads, attn = self.attention(q, k, v, mask=mask)
        heads = heads.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        # [batch_size, n_head, len_q, d_v] -> [batch_size, len_q, n_head * d_v] = [batch_size, len_q, d_model]

        output = self.dropout(heads @ self.w_o)
        output = self.layer_norm(output + residual)
        return output, attn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.FFN = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        residual = input
        output = self.FFN(input)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, d_ff):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_inputs: [batch_size, src_len, d_model]
        # enc_self_attn_mask: [batch_size, src_len, src_len]

        enc_outputs, attn = self.multi_head_attention(q=enc_inputs, k=enc_inputs, v=enc_inputs, mask=enc_self_attn_mask)
        enc_outputs = self.feed_forward(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_ff, enc_n):
        super().__init__()
        self.enc_layers = nn.ModuleList([EncoderLayer(n_head, d_model, d_k, d_v, d_ff) for _ in range(enc_n)])

    def forward(self, enc_inputs, enc_outputs):
        # enc_inputs: [batch_size, src_len]
        # enc_outputs: [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_attns = []

        for layer in self.enc_layers:
            enc_outputs, attn = layer(enc_outputs, enc_self_attn_mask)
            # attn : [batch_size, n_heads, src_len, src_len]
            enc_attns.append(attn)
        # enc_attns : [enc_n, batch_size,n_heads, src_len, src_len]
        return enc_outputs, enc_attns


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, d_ff):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.dec_crs_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_crs_attn_mask):
        # dec_inputs: [batch_size, tgt_len, d_model]
        # enc_outputs: [batch_size, src_len, d_model]
        # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        # dec_crs_attn_mask: [batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs,
                                                        dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_crs_attn = self.dec_crs_attn(dec_outputs, enc_outputs, enc_outputs,
                                                      dec_crs_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_crs_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs = self.ffn(dec_outputs)  # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_crs_attn


class Decoder(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_ff, dec_n):
        super().__init__()
        self.dec_layers = nn.ModuleList([DecoderLayer(n_head, d_model, d_k, d_v, d_ff) for _ in range(dec_n)])

    def forward(self, dec_inputs, dec_outputs, enc_inputs, enc_outputs):
        # dec_inputs: [batch_size, tgt_len]
        # dec_outputs: [batch_size, tgt_len, d_model]
        # enc_inputs: [batch_size, src_len]
        # enc_outputs: [batch_size, src_len, d_model]

        # Mask for subsequent decoding steps
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_seq_mask(dec_inputs)
        # Combine masks using logical OR
        dec_self_attn_mask = dec_self_attn_pad_mask | dec_self_attn_subsequent_mask

        dec_crs_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_crs_attns = [], []
        for layer in self.dec_layers:
            dec_outputs, dec_self_attn, dec_crs_attn = layer(dec_outputs, enc_outputs,
                                                             dec_self_attn_mask, dec_crs_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_crs_attns.append(dec_crs_attn)
        return dec_outputs, dec_self_attns, dec_crs_attns


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_head, d_k, d_v, d_ff, enc_n, dec_n):
        super().__init__()
        self.src_emb = InputEmbeddings(src_vocab_size, d_model)
        self.tgt_emb = InputEmbeddings(tgt_vocab_size, d_model)
        self.pos_emb = LearningPositionalEncoder(
            d_model)  # I use the same LearningPositionalEncoder for bose encode and decode
        self.encoder = Encoder(d_model, n_head, d_k, d_v, d_ff, enc_n)
        self.decoder = Decoder(d_model, n_head, d_k, d_v, d_ff, dec_n)
        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, enc_inputs, dec_inputs):
        # enc_inputs: [batch_size, src_len]
        # dec_inputs: [batch_size, tgt_len]
        src_len = enc_inputs.size(1)
        tgt_len = dec_inputs.size(1)

        # Convert indexes to embeddings
        enc_outputs = self.src_emb(enc_inputs)
        dec_outputs = self.tgt_emb(dec_inputs)

        # Add positional encodings
        enc_outputs = self.pos_emb(enc_outputs)
        dec_outputs = self.pos_emb(dec_outputs)

        # Encode the source sequence
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, enc_outputs)

        # Decode the target sequence
        dec_outputs, dec_self_attns, dec_crs_attns = self.decoder(dec_inputs, dec_outputs, enc_inputs, enc_outputs)

        # Linearly project the decoder outputs to the target vocabulary size
        dec_outputs = self.linear(
            dec_outputs)  # dec_outputs: [batch_size, src_len, d_model] -> [batch_size, src_len, tgt_vocab_size]

        return torch.softmax(dec_outputs, dim=-1)


if __name__ == '__main__':
    params = {
        "src_vocab_size": 500,
        "tgt_vocab_size": 500,
        "d_model": 512,
        "n_head": 8,
        "d_k": 64,
        "d_v": 64,
        "d_ff": 2048,
        "enc_n": 6,
        "dec_n": 6,
    }

    learning_rate = 0.00001
    epochs = 100
    batch_size = 4
    src_len = 10
    tgt_len = 12

    # Initialize model, optimizer and loss
    model = Transformer(**params)

    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


    # Add weight initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training loop
    for epoch in range(epochs):
        model.train()

        # Generate random training data (replace with your actual data)
        enc_inputs = torch.randint(0, params["src_vocab_size"], (batch_size, src_len))
        dec_inputs = torch.randint(0, params["tgt_vocab_size"], (batch_size, tgt_len))
        # Shift decoder targets by one position
        target_outputs = torch.randint(0, params["tgt_vocab_size"], (batch_size, tgt_len))

        # Forward pass
        optimizer.zero_grad()
        outputs = model(enc_inputs, dec_inputs)

        # Calculate loss
        loss = criterion(outputs.view(-1, params["tgt_vocab_size"]), target_outputs.view(-1))

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print training progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

            # Evaluate max probabilities and vocab indices
            with torch.no_grad():
                model.eval()
                eval_outputs = model(enc_inputs, dec_inputs)
                max_prob, max_vocab = eval_outputs.max(dim=-1)
                print("\nMax probabilities:")
                print(max_prob)
                print("\nCorresponding vocab indices:")
                print(max_vocab)
