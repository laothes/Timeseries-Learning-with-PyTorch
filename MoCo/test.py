import torch

class SimpleMessagePassingGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleMessagePassingGNN, self).__init__()
        # 用于消息计算的线性变换
        self.message_transform = torch.nn.Linear(in_channels, out_channels)
        # 用于节点更新的线性变换
        self.node_update = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        """
        x: 节点特征矩阵，形状为 [num_nodes, in_channels]
        edge_index: 边索引，形状为 [2, num_edges]，每列表示一条边 (source, target)
        """
        num_nodes = x.size(0)

        # Step 1: 消息计算
        # 对每条边的起点节点的特征进行变换
        edge_sources = edge_index[0]  # 源节点索引
        messages = self.message_transform(x[edge_sources])  # 对源节点特征进行线性变换

        # Step 2: 消息聚合
        # 将消息聚合到每个目标节点
        edge_targets = edge_index[1]  # 目标节点索引
        aggregated_messages = torch.zeros(num_nodes, messages.size(1))
        aggregated_messages = aggregated_messages.index_add(0, edge_targets, messages)  # 按目标节点索引聚合消息

        # Step 3: 节点状态更新
        updated_nodes = self.node_update(aggregated_messages)  # 使用聚合的消息更新节点状态

        return updated_nodes


# 示例输入
# 节点特征矩阵 (4 个节点，每个有 3 个特征)
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0],
                  [10.0, 11.0, 12.0]], dtype=torch.float)

# 边索引 (4 条边)
edge_index = torch.tensor([[0, 1, 2, 0],  # 源节点
                           [1, 2, 0, 3]],  # 目标节点
                          dtype=torch.long)

# 定义模型
model = SimpleMessagePassingGNN(in_channels=3, out_channels=2)

# 前向传播
output = model(x, edge_index)
print("Output:", output)
