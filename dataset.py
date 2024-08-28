import glob
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from natsort import natsorted


class MyDataset(Dataset):
    def __init__(self, image_dir, groundtruth_dir=None, transform=None):
        self.image_dir = image_dir
        self.groundtruth_dir = groundtruth_dir

        self.image_paths = natsorted(
            [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')])
        if self.groundtruth_dir:
            self.groundtruth_paths = natsorted(
                [os.path.join(groundtruth_dir, fname) for fname in os.listdir(groundtruth_dir) if
                 fname.endswith('.png')])

        # 如果没有指定 transform，则默认使用 Resize 和 ToTensor
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            # print(f'1: {image.shape}')  # 打印变换后的图像尺寸

        if self.groundtruth_dir:
            groundtruth_path = self.groundtruth_paths[idx]
            # groundtruth = Image.open(groundtruth_path).convert('RGB')
            groundtruth = Image.open(groundtruth_path).convert('L')

            if self.transform:
                groundtruth = self.transform(groundtruth)
                # print(f'2: {groundtruth.shape}')
            return image, groundtruth
        else:
            return image


class MyDataset2(Dataset):
    def __init__(self, image_dir, groundtruth_dir=None, transform=None):
        self.image_dir = image_dir
        self.groundtruth_dir = groundtruth_dir

        self.image_paths = natsorted(
            [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')])
        if self.groundtruth_dir:
            self.groundtruth_paths = natsorted(
                [os.path.join(groundtruth_dir, fname) for fname in os.listdir(groundtruth_dir) if
                 fname.endswith('.png')])

        # 如果没有指定 transform，则默认使用 Resize 和 ToTensor
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            # print(f'1: {image.shape}')  # 打印变换后的图像尺寸

        if self.groundtruth_dir:
            groundtruth_path = self.groundtruth_paths[idx]
            groundtruth = Image.open(groundtruth_path).convert('RGB')
            # groundtruth = Image.open(groundtruth_path).convert('L')

            if self.transform:
                groundtruth = self.transform(groundtruth)
                # print(f'2: {groundtruth.shape}')
            return image, groundtruth
        else:
            return image
