# train_model1.py
import os

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MyDataset, MyDataset2
from model import Unet2
from save_results import save_middle_result


def save_image(output_directory, original_image, index):
    os.makedirs(output_directory, exist_ok=True)

    # 保存原始预测图像
    # original_image = denormalize(original_image.squeeze(), mean, std)
    original_image = original_image[0].cpu()
    # print(original_image.size())

    original_image = original_image.permute(1, 2, 0).numpy()
    original_image = np.clip(original_image, 0, 1)

    original_image = (original_image * 255).astype(np.uint8)
    original_image_pil = Image.fromarray(original_image)

    original_image_pil.save(os.path.join(output_directory, f'epoch_{index}.png'))


# 参数
image_dir = 'training dataset/dataset/chessboard_images without bg'
groundtruth_dir = 'training dataset/dataset/ground truth without bg'


batch_size = 10
num_epochs = 5
learning_rate = 0.0001

# 数据集和数据加载器
dataset = MyDataset2(image_dir=image_dir, groundtruth_dir=groundtruth_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Unet2(3, 3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练
# model.train()
for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0
    for images, groundtruths in dataloader:

        images, groundtruths = images.to(device), groundtruths.to(device)
        noise = torch.rand(images.size()).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        # outputs = model(noise)

        loss = criterion(outputs, groundtruths)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')

    if epoch % 5 == 0:
        with torch.no_grad():
            save_middle_result('middle results2/results', outputs, epoch)
            save_middle_result('middle results2/inputs', images, epoch)
            save_middle_result('middle results2/groundtruths', groundtruths, epoch)

# 保存生成器模型参数
model_dir = 'model_data/model2_data'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
torch.save(model.state_dict(), os.path.join(model_dir, 'model2 good 5.pth'))




