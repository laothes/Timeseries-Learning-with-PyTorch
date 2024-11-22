import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
from enum import IntEnum
import datetime
import torch.optim as optim
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class GRU_batchfirst(nn.Module):
    # hidden layer ->  [batch_size, seq_len, hidden_size]
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # reset gate rt
        self.W_r = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_r = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_r = Parameter(torch.Tensor(hidden_size))
        # update gate zt
        self.W_z = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_z = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_z = Parameter(torch.Tensor(hidden_size))
        # proposed unit ht
        self.W_h = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_h = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = Parameter(torch.Tensor(hidden_size))

        # initialize
        self._initialize_weights()

    def _initialize_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                # w,u
                nn.init.xavier_uniform_(p.data)
            else:
                # b
                nn.init.zeros_(p.data)

    def _init_states(self, x):
        h_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=x.dtype).to(x.device)
        return h_t

    def forward(self, x, init_states=None):
        batch_size, seq_size, _ = x.size()
        hidden_seq = []

        # initialize
        if init_states is None:
            h_t = self._init_states(x)
        else:
            h_t = init_states

        for t in range(seq_size):
            x_t = x[:, t, :]  # [batch_size, input_size]

            r_t = torch.sigmoid(x_t @ self.W_r + h_t @ self.U_r + self.b_r)  # [batch_size, hidden_size]
            z_t = torch.sigmoid(x_t @ self.W_z + h_t @ self.U_z + self.b_z)  # [batch_size, hidden_size]
            h_tt = torch.tanh(x_t @ self.W_h + h_t @ self.U_h + self.b_h)  # [batch_size, hidden_size]

            h_t = (1 - z_t) * h_t + z_t * h_tt  # [batch_size, hidden_size]
            hidden_seq.append(h_t)

        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)  # [seq_size, batch_size, hidden_size]
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()  # [batch_size, seq_size, hidden_size]
        return hidden_seq, h_t


class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = GRU_batchfirst(input_size=input_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        batch_size, _, _ = x.size()
        h, h_t = self.gru(x)
        y = self.activation(h)

        y = y.view(-1, self.hidden_size)  # [batch_size * seq_size, hidden_size]
        y = self.linear(y)
        y = y.view(batch_size, -1, output_size)  # [batch_size * seq_size, hidden_size]
        return y


###########################Global Var###################################

num_time_steps = 16
input_size = 3
hidden_size = 16
output_size = 3
lr = 0.01


def getdata():
    x1 = np.linspace(1, 10, 30).reshape(30, 1)
    y1 = (np.zeros_like(x1) + 2) + np.random.rand(30, 1) * 0.1
    z1 = (np.zeros_like(x1) + 2).reshape(30, 1)
    tr1 = np.concatenate((x1, y1, z1), axis=1)
    return tr1


#####################train#################################
def train_GRU(data):
    model = Net(input_size, hidden_size)
    print('model:\n', model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)
    l = []
    # 3000 times
    for iter in range(3000):
        # loss = 0
        start = np.random.randint(10, size=1)[0]
        end = start + 15
        x = torch.tensor(data[start:end]).float().view(1, num_time_steps - 1, 3)
        # Randomly select 15 points in the data as input to predict the 16th
        y = torch.tensor(data[start + 5:end + 5]).float().view(1, num_time_steps - 1, 3)

        output = model(x)

        loss = criterion(output, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            print("Iteration: {} loss {}".format(iter, loss.item()))
            l.append(loss.item())

    ##############################Loss#################################
    plt.plot(l, 'r')
    plt.xlabel('Times')
    plt.ylabel('loss')
    plt.title('GRU - Loss function decline curve')
    plt.show()

    return model


#############################Prediction#########################################

def GRU_pre(model, data):
    data_test = data[19:29]
    data_test = torch.tensor(np.expand_dims(data_test, axis=0), dtype=torch.float32)

    pred1 = model(data_test)
    print('pred1.shape:', pred1.shape)
    pred2 = model(pred1)
    print('pred2.shape:', pred2.shape)
    pred1 = pred1.detach().numpy().reshape(10, 3)
    pred2 = pred2.detach().numpy().reshape(10, 3)
    predictions = np.concatenate((pred1, pred2), axis=0)
    print('predictions.shape:', predictions.shape)

    #############################Visualize########################################

    fig = plt.figure(figsize=(9, 6))
    ax = Axes3D(fig)
    fig.add_axes(ax)
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c='red')
    ax.scatter3D(predictions[:, 0], predictions[:, 1], predictions[:, 2], c='blue')
    ax.set_xlabel('X')
    ax.set_xlim(0, 8.5)
    ax.set_ylabel('Y')
    ax.set_ylim(0, 10)
    ax.set_zlabel('Z')
    ax.set_zlim(0, 4)
    plt.title("GRU track prediction")
    plt.show()


if __name__ == '__main__':
    data = getdata()
    start = datetime.datetime.now()
    model = train_GRU(data)
    end = datetime.datetime.now()
    print('The training time: %s' % str(end - start))
    GRU_pre(model, data)
