# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
import numpy as np


# 1. 数据加载函数
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    class_names = [str(i) for i in range(10)]

    return train_loader, test_loader, class_names


# 2. CNN模型定义
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 3. 训练函数
def train(model, train_loader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return model


# 4. 测试函数
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    results = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            for i in range(len(images)):
                results.append((images[i].cpu(), preds[i].item(), labels[i].item()))

    accuracy = 100 * correct / total
    print(f"测试准确率: {accuracy:.2f}%")
    return results


# 5. 可视化测试结果函数
def visualize_test_results(results, class_names, num=6):
    plt.figure(figsize=(12, 6))
    for i in range(num):
        idx = np.random.randint(len(results))
        img, pred, true = results[idx]
        plt.subplot(2, 3, i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        color = 'green' if pred == true else 'red'
        plt.title(f"预测: {class_names[pred]}\n真实: {class_names[true]}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# 6. 图片预处理函数
def preprocess_image(image_path):
    """预处理单张本地图片，使其符合模型输入要求"""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转为灰度图
        transforms.Resize((28, 28)),  # 调整为28x28大小
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 增加批次维度
    return image


# 7. 批量识别10张图片函数（核心修改）
def predict_multiple_images(model, image_paths, device, class_names):
    """
    批量识别多张图片（适配10张）
    :param model: 训练好的模型
    :param image_paths: 图片路径列表（建议10个）
    :param device: 运行设备
    :param class_names: 类别名称
    :return: 预测结果列表 [(图片路径, 预测数字), ...]
    """
    predictions = []
    valid_paths = []

    # 筛选有效图片路径
    for idx, path in enumerate(image_paths):
        if not os.path.exists(path):
            print(f"警告：第{idx + 1}张图片 {path} 不存在，跳过")
            continue
        valid_paths.append(path)

    # 若有效图片不足10张，提示
    if len(valid_paths) == 0:
        print("错误：没有有效图片路径")
        return []
    elif len(valid_paths) < 10:
        print(f"提示：仅找到 {len(valid_paths)} 张有效图片（目标10张），将识别这些图片")

    # 批量预测
    model.eval()
    with torch.no_grad():
        plt.figure(figsize=(15, 8))  # 调整画布大小适配10张图
        for idx, path in enumerate(valid_paths):
            # 预处理图片
            image = preprocess_image(path).to(device)
            # 模型预测
            output = model(image)
            _, pred = torch.max(output, 1)
            predicted_class = class_names[pred.item()]
            predictions.append((path, predicted_class))

            # 显示单张图片和预测结果（布局：4行3列，适配10张图）
            plt.subplot(4, 3, idx + 1)  # 4行3列可容纳12个位置，足够放10张
            original_image = Image.open(path).convert('L')
            plt.imshow(original_image, cmap='gray')
            plt.title(f"第{idx + 1}张\n预测: {predicted_class}", fontsize=10)
            plt.axis('off')

            # 达到10张则停止（防止超过10张）
            if idx + 1 == 10:
                break

    plt.tight_layout()
    plt.show()

    # 打印汇总结果
    print("\n===== 批量识别结果汇总 =====")
    for idx, (path, pred) in enumerate(predictions, 1):
        print(f"第{idx}张：{os.path.basename(path)} → 预测数字：{pred}")

    return predictions


# 8. 保存和加载模型函数
def save_model(model, path='mnist_cnn_model.pth'):
    """保存模型权重"""
    torch.save(model.state_dict(), path)
    print(f"模型已保存至 {path}")


def load_model(path='mnist_cnn_model.pth', device=None):
    """加载模型权重"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"已从 {path} 加载模型")
    return model


# 9. 自动获取指定目录下的10张图片路径（便捷功能）
def get_image_paths_from_dir(img_dir, max_num=10):
    """
    从指定目录自动读取图片路径（支持png/jpg/jpeg格式）
    :param img_dir: 图片目录
    :param max_num: 最多读取10张
    :return: 图片路径列表
    """
    if not os.path.isdir(img_dir):
        print(f"错误：目录 {img_dir} 不存在")
        return []

    # 支持的图片格式
    img_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    img_paths = []

    for file in os.listdir(img_dir):
        if any(file.lower().endswith(ext) for ext in img_extensions):
            img_paths.append(os.path.join(img_dir, file))
            if len(img_paths) >= max_num:
                break

    return img_paths


# 主函数
def main():
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    train_loader, test_loader, class_names = load_data()

    # 加载/训练模型
    model_path = 'mnist_cnn_model.pth'
    if os.path.exists(model_path):
        model = load_model(model_path, device)
    else:
        model = SimpleCNN().to(device)
        model = train(model, train_loader, device, epochs=5)
        save_model(model, model_path)

    # 测试模型（可选）
    test_results = test(model, test_loader, device)
    visualize_test_results(test_results, class_names)

    # ========== 批量识别10张图片核心配置 ==========
    # 方式1：手动指定10张图片路径（推荐，精准控制）
    # 请将下面的路径替换为你的图片实际路径
    image_paths = [
        "num1.png",  # 第1张
        "num2.png",  # 第2张
        "num3.png",  # 第3张
        "num4.png",  # 第4张
        "num5.png",  # 第5张
        "num6.png",  # 第6张
        "num7.png",  # 第7张
        "num8.png",  # 第8张
        "num9.png",  # 第9张
        "num10.png"  # 第10张
    ]

    # 方式2：自动读取指定目录下的10张图片（便捷）
    # img_dir = "./num_images"  # 图片存放目录
    # image_paths = get_image_paths_from_dir(img_dir, max_num=10)

    # 执行批量识别
    predict_multiple_images(model, image_paths, device, class_names)


if __name__ == "__main__":
    main()
