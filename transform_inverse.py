import os
import numpy as np
import cv2
import csv


def load_corners_from_csv(csv_path):
    corners_dict = {}
    with open(csv_path, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            image_name = row[0]
            corners = np.array(row[1:], dtype=np.float32).reshape(4, 2)
            corners_dict[image_name] = corners
    return corners_dict


def visualize_corners(image, corners):
    for i, corner in enumerate(corners):
        cv2.circle(image, tuple(corner.astype(int)), 5, (0, 255, 0), -1)
        cv2.putText(image, f'P{i+1}', tuple(corner.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow("Corners", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def apply_inverse_transform_and_merge(original_image, processed_image, corners, M_inv, output_path):
    h, w = original_image.shape[:2]
    warped_back = cv2.warpPerspective(processed_image, M_inv, (w, h))

    print("M_inv:", M_inv)
    print("Corners:", corners)

    mask = np.zeros_like(original_image)
    cv2.fillConvexPoly(mask, corners.astype(int), (255, 255, 255))
    inverse_mask = cv2.bitwise_not(mask)
    original_image_masked = cv2.bitwise_and(original_image, inverse_mask)
    result = cv2.bitwise_or(original_image_masked, warped_back)

    cv2.imwrite(output_path, result)


def inverse_transform_imgs(original_dir, processed_dir, corners_csv_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    corners_dict = load_corners_from_csv(corners_csv_path)

    for image_name in os.listdir(original_dir):
        original_image_path = os.path.join(original_dir, image_name)
        processed_image_path = os.path.join(processed_dir, image_name)

        if os.path.exists(original_image_path) and os.path.exists(processed_image_path):
            original_image = cv2.imread(original_image_path)
            processed_image = cv2.imread(processed_image_path)

            corners = corners_dict.get(image_name)

            if corners is not None:

                # 可视化角点以确认顺序
                # visualize_corners(original_image.copy(), corners)

                M_inv = cv2.getPerspectiveTransform(
                    np.array([[0, 0], [511, 0], [511, 511], [0, 511]], dtype=np.float32),
                    corners.astype(np.float32)
                )

                output_path = os.path.join(output_dir, image_name)
                apply_inverse_transform_and_merge(original_image, processed_image, corners, M_inv, output_path)
                print(f"Processed and saved: {output_path}")
            else:
                print(f"Missing data for image: {image_name}")
        else:
            print(f"Image not found: {image_name}")

