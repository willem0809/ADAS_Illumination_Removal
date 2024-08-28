import os

import numpy as np
from PIL import Image
from PIL.Image import Resampling


def save_images(output_directory, original_image, index):
    os.makedirs(output_directory, exist_ok=True)

    original_image = original_image[0].cpu()
    original_image = original_image.permute(1, 2, 0).numpy()
    original_image = np.clip(original_image, 0, 1)
    original_image = (original_image * 255).astype(np.uint8)
    original_image_pil = Image.fromarray(original_image).resize((512, 512), Resampling.LANCZOS)

    original_image_pil.save(os.path.join(output_directory, f'image{index}.png'))


def save_masks(output_directory, original_image, index):
    os.makedirs(output_directory, exist_ok=True)

    # 打印调试信息
    # print(f"Original image shape: {original_image.shape}, dtype: {original_image.dtype}")

    original_image = original_image.squeeze().cpu().numpy()

    # 再次打印调试信息
    # print(f"Squeezed image shape: {original_image.shape}, dtype: {original_image.dtype}")

    # 扩展灰度值到[0, 255]
    original_image = (original_image * 255).astype(np.uint8)

    # 将灰度值低于128的像素设为0，其他设为255
    binary_mask = np.where(original_image < 128, 0, 255).astype(np.uint8)

    # 调整图像大小到 512x512
    binary_mask_pil = Image.fromarray(binary_mask, mode='L').resize((512, 512), Resampling.LANCZOS)

    # 保存图像
    binary_mask_pil.save(os.path.join(output_directory, f'image{index}.png'))


def save_middle_result(output_directory, original_image, index):
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


def save_middle_result2(output_directory, original_image, index):
    os.makedirs(output_directory, exist_ok=True)

    original_image = original_image.squeeze().cpu().numpy()

    # 扩展灰度值到[0, 255]
    original_image = (original_image * 255).astype(np.uint8)

    # 将灰度值低于128的像素设为0，其他设为255
    binary_mask = np.where(original_image < 128, 0, 255).astype(np.uint8)

    binary_mask_pil = Image.fromarray(binary_mask, mode='L')

    # 保存图像
    binary_mask_pil.save(os.path.join(output_directory, f'epoch_{index}.png'))