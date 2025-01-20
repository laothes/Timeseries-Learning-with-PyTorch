import torch
import torch.nn as nn


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    Gather tensors from all GPUs in a distributed training setup
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class MoCo(nn.Module):
    '''
    We use Transformer to replace ResNet 50, so we don't need to shuttle the batch
    as transformer using layer normalization instead of batch normalization
    '''

    def __init__(self, input_channels, num_mlp=3,
                 d_model=256, nhead=8, num_layers=3, dim_feedforward=2048, dropout=0.1,
                 K=65536, m=0.999, T=0.07):
        super().__init__()
        self.K = K  # dictionary size
        self.m = m  # momentum
        self.T = T  # temperature
        self.context_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.cov_q = nn.Conv1d(input_channels, d_model, kernel_size=1)
        self.cov_k = nn.Conv1d(input_channels, d_model, kernel_size=1)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_q = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_k = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projection heads - SimCLR v2
        def make_projection_head(input_dim, hidden_dim, output_dim, num_layers):
            layers = []
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                input_dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, output_dim))
            return nn.Sequential(*layers)

        self.projection_q = make_projection_head(d_model, d_model, d_model, num_mlp)
        self.projection_k = make_projection_head(d_model, d_model, d_model, num_mlp)

        # Encode
        self.encoder_q = nn.Sequential(self.cov_q, self.transformer_q, self.projection_q)
        self.encoder_k = nn.Sequential(self.cov_q, self.transformer_k, self.projection_k)

        # Initialize the momentum encoder's parameters
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # Momentum encoder is not updated by gradients

        # Queue for negative samples
        self.register_buffer("queue", torch.randn(d_model, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key(self, layer_q, layer_k):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
                layer_q.parameters(), layer_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue, but my computer doesn't support multiprocess
        # keys = concat_all_gather(keys)
        # key: [batch, d_model]
        # queue: [d_model, K]

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, q, k):
        batch_size = q.shape[0]
        q = self.cov_q(q)  # [batch, d_model, seq_len]

        with torch.no_grad():
            self._momentum_update_key(self.cov_q, self.cov_k)
            k = self.cov_k(k)  # [batch, d_model, seq_len]

        q = q.transpose(1, 2)  # [batch, seq_len, d_model]
        k = k.transpose(1, 2)  # [batch, seq_len, d_model]

        q = torch.cat([self.context_token.expand(batch_size, -1, -1), q], dim=1)  # [batch, seq_len+1, d_model]
        k = torch.cat([self.context_token.expand(batch_size, -1, -1), k], dim=1)  # [batch, seq_len+1, d_model]

        q = self.transformer_q(q)
        q_c = self.projection_q(q[:, 0, :])  # output of context_token [batch, d_model]
        # q_c = nn.functional.normalize(q_c, dim=-1)
        # when using nn.functional.normalize(), the model doesn't fit

        with torch.no_grad():
            self._momentum_update_key(self.encoder_q, self.encoder_k)
            k = self.transformer_k(k)
            k_c = self.projection_k(k[:, 0, :])
            # k_c = nn.functional.normalize(k_c, dim=-1)

        l_pos = torch.einsum("nc,nc->n", [q_c, k_c]).unsqueeze(-1)
        # l_pos = q_c.unsqueeze(1) @ k_c.unsqueeze(2) -> [batch, 1, d_model] @ [batch, d_model, 1] -> [batch, 1]

        l_neg = torch.einsum("nc,ck->nk", [q_c, self.queue.clone().detach()])  # [batch, K]

        logits = torch.cat([l_pos, l_neg], dim=1)  # [batch, 1+K]
        logits /= self.T  # apply temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_c)

        return logits, labels

    def predict(self, x):
        # for fine-tuning
        # [batch, channels, seq_len]
        batch_size = x.shape[0]
        x = self.cov_q(x).transpose(1, 2)
        x = torch.cat([self.context_token.expand(batch_size, -1, -1), x], dim=1)
        h = self.transformer_q(x)[:, 0, :]
        first_layer = self.projection_q[0]
        h = first_layer(h)
        return h


if __name__ == '__main__':
    pass
