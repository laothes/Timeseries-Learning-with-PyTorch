import torch

# 加载数据
test = torch.load('./data/Preprocessed Epilepsy dataset/test.pt')

# 查看数据内容
for i,j in test.items():
    print(i)
    print(j.shape)
    print('--------------------------------')