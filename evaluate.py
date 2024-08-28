import os
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def evaluate_images(processed_dir, groundtruth_dir):
    # 获取处理后图像和 ground truth 图像的文件名
    processed_images = sorted(os.listdir(processed_dir))
    groundtruth_images = sorted(os.listdir(groundtruth_dir))

    if len(processed_images) != len(groundtruth_images):
        raise ValueError("The number of processed images and ground truth images must be the same.")

    ssim_scores = []
    psnr_scores = []

    for processed_image_name, groundtruth_image_name in zip(processed_images, groundtruth_images):
        processed_image_path = os.path.join(processed_dir, processed_image_name)
        groundtruth_image_path = os.path.join(groundtruth_dir, groundtruth_image_name)

        # 读取图像
        processed_image = cv2.imread(processed_image_path, cv2.IMREAD_GRAYSCALE)
        groundtruth_image = cv2.imread(groundtruth_image_path, cv2.IMREAD_GRAYSCALE)

        if processed_image is None or groundtruth_image is None:
            print(f"Error reading {processed_image_name} or {groundtruth_image_name}. Skipping.")
            continue

        # 计算 SSIM 和 PSNR
        ssim_value = ssim(groundtruth_image, processed_image)
        psnr_value = psnr(groundtruth_image, processed_image)

        ssim_scores.append(ssim_value)
        psnr_scores.append(psnr_value)

        print(f"{processed_image_name} - SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.2f} dB")

    # 返回平均 SSIM 和 PSNR 分数
    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    avg_psnr = sum(psnr_scores) / len(psnr_scores)

    print(f"\nAverage SSIM: {avg_ssim:.4f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")

    return avg_ssim, avg_psnr


# 示例使用
# processed_dir = 'predict/results'
# groundtruth_dir = 'training dataset/dataset/ground truth'

# processed_dir = 'predict/masks'
# groundtruth_dir = 'training dataset/dataset/mask'

processed_dir = 'p1'
groundtruth_dir = 'training dataset/dataset/ground truth'


evaluate_images(processed_dir, groundtruth_dir)
