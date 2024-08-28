import os

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MyDataset
from model import Unet1
from save_results import save_masks, save_middle_result, save_middle_result2


# 参数
image_dir = 'training dataset/dataset/chessboard_images'
groundtruth_dir = 'training dataset/dataset/mask'


batch_size = 10
num_epochs = 5
learning_rate = 0.0001

# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
# ])

# 数据集和数据加载器
dataset = MyDataset(image_dir=image_dir, groundtruth_dir=groundtruth_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet1(3, 1).to(device)
criterion = nn.MSELoss()
# criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练
# model.train()
for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0
    for image, groundtruth in dataloader:
        image, groundtruth = image.to(device), groundtruth.to(device)
        # noise = torch.rand(image.size()).to(device)

        optimizer.zero_grad()
        output = model(image)
        # print(output.shape)

        loss = criterion(output, groundtruth)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')

    if epoch % 5 == 0:
        with torch.no_grad():
            # save_image('predict/middle results/results', output, epoch)

            save_middle_result('middle results/inputs', image, epoch)
            save_middle_result2('middle results/results', output[0], epoch)
            save_middle_result2('middle results/groundtruths', groundtruth[0], epoch)

# 保存生成器模型参数
model_dir = 'model_data/model1_data'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
torch.save(model.state_dict(), os.path.join(model_dir, 'model1 good1 5.pth'))



