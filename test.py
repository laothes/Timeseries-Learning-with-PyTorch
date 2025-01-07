import numpy as np
from timeit import default_timer as timer

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

train = torch.load('data/Epilepsy/train.pt', weights_only=True)
print(train['samples'].shape[-1])

train = torch.utils.data.TensorDataset(train['samples'], train['labels'])
trainload = DataLoader(train, batch_size=256, shuffle=True)
for batch_idx, (sample, target) in enumerate(trainload):
    print(batch_idx)
    print(sample)
    print(target)
