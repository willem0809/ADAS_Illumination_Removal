import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


# 设定图片尺寸及棋盘格子数来生成对于棋盘标定板
def generate_chessboard_image(img_size, grid_size):
    image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    square_size = img_size // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            if (i + j) % 2 == 0:
                top_left = (i * square_size, j * square_size)
                bottom_right = ((i + 1) * square_size, (j + 1) * square_size)
                cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), -1)
    return image


# 对图片增加高光
def add_specular_highlight(img, center, radius, intensity):
    height, width, _ = img.shape

    # 创建一个与图像相同尺寸的空白图像用于高光
    highlight = np.zeros_like(img)

    # 定义高光的中心位置和半径
    x, y = center
    x *= width
    x = int(x)
    y *= height
    y = int(y)

    rad = int(radius * (x + y) / 2)
    ksize = max(1, 2 * (rad // 2) + 1)

    # 创建高光的颜色 (白色)
    color = (255, 255, 255)  # BGR格式

    # 使用高斯模糊创建光晕效果
    cv2.circle(highlight, (x, y), rad, color, -1, cv2.LINE_AA)
    blurred_highlight = cv2.GaussianBlur(highlight, (ksize, ksize), sigmaX=rad / 2, sigmaY=rad / 2)

    # 根据intensity调整高光的强度
    alpha = intensity / 100.0

    # 叠加高光图像并保持原始背景亮度
    highlighted_img = cv2.addWeighted(img, 1, blurred_highlight, alpha, 0)

    return highlighted_img


# 生成随机的目标角点坐标
def generate_random_dst_points(width, height, margin=0.2):
    """
    参数:
    width: int, 图像宽度
    height: int, 图像高度
    margin: float, 随机角点的边缘距离比例，默认为0.2

    返回:
    dst_points: np.array, 目标图像的四个角点
    """
    margin_x = int(width * margin)
    margin_y = int(height * margin)

    dst_points = np.float32([
        [np.random.randint(0, margin_x), np.random.randint(0, margin_y)],
        [np.random.randint(width - margin_x, width), np.random.randint(0, margin_y)],
        [np.random.randint(width - margin_x, width), np.random.randint(height - margin_y, height)],
        [np.random.randint(0, margin_x), np.random.randint(height - margin_y, height)]
    ])

    return dst_points


# 将正面的图片转换为斜面的图片，并生成相应的掩码
def transform_to_oblique_view(image, src_points=None, dst_points=None):
    """
    参数:
    image: np.array, 输入图像
    src_points: np.array, 输入图像的四个角点，可选
    dst_points: np.array, 目标图像的四个角点，可选

    返回:
    transformed_image: np.array, 斜面视角的图像
    mask: np.array, 生成的掩码
    """
    height, width = image.shape[:2]

    if src_points is None:
        src_points = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ])

    if dst_points is None:
        dst_points = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ])

    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 应用透视变换
    transformed_image = cv2.warpPerspective(image, matrix, (width, height))

    # 创建掩码，表示斜面棋盘格的部分
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(dst_points)], (255, 255, 255))

    return transformed_image, mask


# 将前景图片叠加到背景图片上，并使用掩码来控制叠加部分
def add_background(background, foreground, position, mask):
    """
    参数:
    background: np.array, 背景图像
    foreground: np.array, 前景图像（需要叠加的图像）
    position: tuple, 前景图像放置在背景图像上的位置 (x, y)
    mask: np.array, 前景图像的掩码

    返回:
    result: np.array, 叠加后的图像
    """
    fg_height, fg_width = foreground.shape[:2]
    x, y = position

    # 检查前景图像是否超出背景图像边界
    if x + fg_width > background.shape[1] or y + fg_height > background.shape[0]:
        print("Foreground image exceeds background boundaries!")
        return background

    # 将前景图像和背景图像转换为相同的数据类型
    background = background.astype(np.uint8)
    # without_background = np.zeros_like(background).astype(np.uint8)
    foreground = foreground.astype(np.uint8)
    mask = mask.astype(np.float32) / 255.0

    # 获取前景图像的 ROI
    foreground_roi = foreground * mask[:, :, np.newaxis]
    background_roi = background[y:y + fg_height, x:x + fg_width]

    # 使用掩码将前景图像与背景图像合并
    masked_background = background_roi * (1 - mask[:, :, np.newaxis])
    combined = cv2.add(foreground_roi, masked_background)

    # 将融合后的 ROI 放回背景图像
    background[y:y + fg_height, x:x + fg_width] = np.clip(combined, 0, 255).astype(np.uint8)

    without_background = np.zeros_like(background).astype(np.uint8)
    without_background[y:y + fg_height, x:x + fg_width] = foreground_roi

    black_background = np.zeros_like(background)
    black_background[y:y + fg_height, x:x + fg_width] = (mask * 255).astype(np.uint8)[:, :, np.newaxis]
    return background, black_background, without_background


# 示例使用

# 生成一组数据（带有背景的图片及gt，不带背景的图片与gt，mask），p为产生视角变换标定板图片的概率
def generate_imgs(background_image, p=0.5):
    chessboard_size = 256
    grids = [4, 8, 16]
    rand = np.random.randint(0, 3)
    base_chessboard = generate_chessboard_image(chessboard_size, grids[rand])

    width, height = np.random.rand(2)
    r = np.random.rand(1) * 0.5 + 0.5
    intensity = 95
    highlight_chessboard = add_specular_highlight(base_chessboard, center=(width, height), radius=r,
                                                  intensity=intensity)

    # 生成斜面视角的标定板图片
    n = np.random.rand(1)
    if n > p:
        dst_points = generate_random_dst_points(chessboard_size, chessboard_size)
    else:
        dst_points = None

    highlight_chessboard, mask = transform_to_oblique_view(highlight_chessboard, dst_points=dst_points)
    base_chessboard, mask1 = transform_to_oblique_view(base_chessboard, dst_points=dst_points)

    # 将生成的斜面标定板图片叠加到背景图片上，并显示掩码
    x_offset = int(np.random.rand() * (512 - chessboard_size))
    y_offset = int(np.random.rand() * (512 - chessboard_size))
    highlight_image, image_mask, highlight_image_without_bg = add_background(background_image, highlight_chessboard,
                                                                             (x_offset, y_offset), mask)
    ground_truth_with_bg, m2, ground_truth_without_bg = add_background(background_image, base_chessboard,
                                                                       (x_offset, y_offset), mask)

    return ground_truth_with_bg, highlight_image, image_mask, ground_truth_without_bg, highlight_image_without_bg


def generate_board(background_image):
    chessboard_size = 256
    grids = [4, 8, 16]
    rand = np.random.randint(0, 3)
    base_chessboard = generate_chessboard_image(chessboard_size, grids[rand])

    # 生成斜面视角的标定板图片
    n = np.random.rand(1)

    dst_points = generate_random_dst_points(chessboard_size, chessboard_size)

    # highlight_chessboard, mask = transform_to_oblique_view(highlight_chessboard, dst_points=dst_points)
    base_chessboard, mask = transform_to_oblique_view(base_chessboard, dst_points=None)
    base_chessboard1, mask1 = transform_to_oblique_view(base_chessboard, dst_points=dst_points)

    # 将生成的斜面标定板图片叠加到背景图片上，并显示掩码
    x_offset = int(np.random.rand() * (512 - chessboard_size))
    y_offset = int(np.random.rand() * (512 - chessboard_size))

    ground_truth_with_bg, m2, ground_truth_without_bg = add_background(background_image, base_chessboard,
                                                                       (x_offset, y_offset), mask)

    ground_truth_with_bg1, m2, ground_truth_without_bg1 = add_background(background_image, base_chessboard1,
                                                                       (x_offset, y_offset), mask1)

    return ground_truth_with_bg, ground_truth_with_bg1


