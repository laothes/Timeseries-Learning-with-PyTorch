import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from src.moco import MoCo


def generate_synthetic_data(batch_size=256, seq_len=200, channels=16):
    """改进的数据生成函数"""
    # 生成基础序列
    base = torch.randn(batch_size, channels, seq_len)

    # 创建更复杂的增强视图
    def augment(x):
        # 随机时间偏移
        shift = torch.randint(-10, 11, (1,)).item()
        if shift > 0:
            x = torch.cat([x[:, :, shift:], x[:, :, :shift]], dim=2)
        elif shift < 0:
            x = torch.cat([x[:, :, shift:], x[:, :, :shift]], dim=2)

        # 随机振幅缩放
        scale = 0.8 + 0.4 * torch.rand(1).item()  # 0.8-1.2之间的随机缩放
        x = scale * x

        # 添加高斯噪声
        noise_scale = 0.05 * torch.rand(1).item()
        x = x + noise_scale * torch.randn_like(x)

        return x

    view1 = augment(base)
    view2 = augment(base)

    return view1, view2


def compute_accuracy(logits, labels):
    """
    计算对比学习的准确率
    参数:
        logits: shape [batch_size, 1+K] 的张量，包含一个正样本和K个负样本的相似度分数
        labels: shape [batch_size] 的张量，值全为0，表示正样本在第一个位置
    返回:
        accuracy: 预测正确的样本比例
    """
    # 将logits转换为预测的概率分布
    predictions = torch.softmax(logits, dim=1)
    # 获取每个样本预测概率最大的位置
    pred_labels = torch.argmax(predictions, dim=1)
    # 计算预测正确的样本数（应该都预测为0，因为正样本在第一个位置）
    correct = (pred_labels == labels).sum().item()
    # 计算准确率
    accuracy = correct / labels.size(0)
    return accuracy


def test_moco():
    # Initialize model parameters
    input_channels = 16
    d_model = 128
    batch_size = 256
    seq_len = 200

    # Create MoCo model
    model = MoCo(
        input_channels=input_channels,
        d_model=d_model,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        K=256*10,  # Smaller queue size for testing
        T=0.07,
        m=0.99,
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training loop setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 100
    losses = []
    accuracies = []  # 记录每个epoch的平均准确率

    print("Starting training loop...")

    for epoch in range(n_epochs):
        epoch_losses = []
        epoch_accuracies = []  # 记录当前epoch中每个批次的准确率

        # Generate synthetic data for this epoch
        query, key = generate_synthetic_data(batch_size, seq_len, input_channels)
        query, key = query.to(device), key.to(device)

        # Forward pass
        logits, labels = model(query, key)

        # Calculate loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)

        # Calculate accuracy
        accuracy = compute_accuracy(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record metrics
        epoch_losses.append(loss.item())
        epoch_accuracies.append(accuracy)

        # Calculate average metrics for this epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)

        losses.append(avg_loss)
        accuracies.append(avg_accuracy)

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2%}")

    # Plot training metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(losses, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('MoCo Training Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(accuracies, label='Training Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('MoCo Training Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Test the encoder's representations
    print("\nTesting learned representations...")

    # Generate test samples
    test_samples, _ = generate_synthetic_data(batch_size=5, seq_len=seq_len, channels=input_channels)
    test_samples = test_samples.to(device)

    # Get representations
    with torch.no_grad():
        representations = model.predict(test_samples)

    # Print representation statistics
    print(f"\nRepresentation shape: {representations.shape}")
    print(f"Mean activation: {representations.mean().item():.4f}")
    print(f"Std deviation: {representations.std().item():.4f}")
    print(f"Final training accuracy: {accuracies[-1]:.2%}")


if __name__ == '__main__':
    test_moco()