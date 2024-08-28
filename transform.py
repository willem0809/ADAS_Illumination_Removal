import os
import numpy as np
import cv2
from PIL import Image
import csv


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# 获取四个角点
def find_quadrilateral_corners(binary_mask):
    binary_mask_cv = binary_mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        corners = approx.reshape((4, 2))
        corners = order_points(corners)
        return corners
    else:
        raise ValueError("The detected contour does not have exactly four corners.")


# 稍微外移角点以覆盖全部标定板区域
def offset_corners(corners, offset):
    center = np.mean(corners, axis=0)  # 计算角点的中心
    new_corners = []

    for corner in corners:
        direction = corner - center
        norm = np.linalg.norm(direction)
        if norm != 0:
            direction = direction / norm
        new_corner = corner + direction * offset
        new_corners.append(new_corner)

    return np.array(new_corners, dtype=np.int32)


# 提取角点内区域并进行透视变换
def extract_and_warp_region(original_image_cv, corners, output_marked_path=None, size=(512, 512)):
    # 在原始图像上标记角点
    if output_marked_path is not None:
        marked_image_cv = original_image_cv.copy()
        for corner in corners:
            cv2.circle(marked_image_cv, tuple(corner), 5, (0, 0, 255), -1)  # 红色，填充圆
        cv2.imwrite(output_marked_path, marked_image_cv)

    # 计算目标的四个角点（正方形）
    dst_corners = np.array([[0, 0], [size[0] - 1, 0], [size[0] - 1, size[1] - 1], [0, size[1] - 1]], dtype=np.float32)

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)

    # 应用透视变换
    warped_region = cv2.warpPerspective(original_image_cv, M, size)

    # 检测是否需要进行水平或垂直翻转
    if (corners[0][0] > corners[1][0]) != (dst_corners[0][0] > dst_corners[1][0]):
        warped_region = cv2.flip(warped_region, 1)  # 水平翻转
    if (corners[0][1] > corners[3][1]) != (dst_corners[0][1] > dst_corners[3][1]):
        warped_region = cv2.flip(warped_region, 0)  # 垂直翻转

    return warped_region


def process_mask(mask_path, original_image_path, img_with_corners_path, output_path, offset=5):
    original_image = Image.open(original_image_path)
    original_image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

    mask_image = Image.open(mask_path)
    mask_array = np.array(mask_image.convert('L'))
    binary_mask = np.where(mask_array < 128, 0, 255).astype(np.uint8)

    corners = find_quadrilateral_corners(binary_mask)

    # 将角点外移
    new_corners = offset_corners(corners, offset)

    # 提取并变换区域
    warped_region = extract_and_warp_region(original_image_cv, new_corners, img_with_corners_path, size=(512, 512))

    # 保存调整后的图像
    cv2.imwrite(output_path, warped_region)

    return new_corners


def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  # 仅处理图片文件
                file_paths.append(os.path.join(root, file))
    return file_paths


def save_corners_to_csv(corners_info, csv_path):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Name', 'Corner 1', 'Corner 2', 'Corner 3', 'Corner 4'])  # CSV Header
        for info in corners_info:
            image_name, corners = info
            row = [image_name] + corners.flatten().tolist()
            writer.writerow(row)


def transform_imgs(mask_dir,
                   image_dir,
                   img_with_corners_dir,
                   output_dir,
                   corners_csv_path,
                   offset=5):
    os.makedirs(img_with_corners_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    mask_paths = get_all_file_paths(mask_dir)

    corners_info = []

    for mask_path in mask_paths:
        image_name = os.path.basename(mask_path)
        image_path = os.path.join(image_dir, image_name)

        if os.path.exists(image_path):
            output_path = os.path.join(output_dir, image_name)
            try:
                img_with_corners_path = os.path.join(img_with_corners_dir, image_name)
                corners = process_mask(mask_path, image_path, img_with_corners_path
                                       , output_path, offset)
                corners_info.append((image_name, corners))
                # print(f"Processed and saved: {output_path}")
            except ValueError as e:
                print(f"Skipping {mask_path} due to error: {e}")
        else:
            print(f"Corresponding image not found for mask: {mask_path}")

    # Save corners information to CSV
    save_corners_to_csv(corners_info, corners_csv_path)


def draw_corners_on_image(image_path, corners, output_path):
    # 读取图像
    image = Image.open(image_path)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 在图像上标记角点
    for corner in corners:
        cv2.circle(image_cv, tuple(corner), 5, (0, 0, 255), -1)  # 红色，填充圆

    # 保存标记了角点的图像
    cv2.imwrite(output_path, image_cv)


def mark_corners_from_csv(image_directory, csv_path, output_directory):
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(csv_path, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # 跳过头行
        for row in csv_reader:
            image_name = row[0]
            corners = np.array(row[1:], dtype=np.int32).reshape(4, 2)

            image_path = os.path.join(image_directory, image_name)
            output_path = os.path.join(output_directory, image_name)

            if os.path.exists(image_path):
                draw_corners_on_image(image_path, corners, output_path)
                print(f"Processed and saved: {output_path}")
            else:
                print(f"Image not found: {image_path}")
