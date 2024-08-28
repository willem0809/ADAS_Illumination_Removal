import glob
import os
import time

import cv2

from imgs_generate_methods import generate_imgs
from transform import transform_imgs


# gt_dir = 'dataset/ground truth'
# img_dir = 'dataset/chessboard_images'
# mask_dir = 'dataset/mask'
# gt_nobg_dir = 'dataset/ground truth without bg'
# img_nobg_dir = 'dataset/chessboard_images without bg'
#
# backgrounds_paths = glob.glob(os.path.join('backgrounds', '*.jpg'))
# backgrounds = []
#
# starttime = time.time()
#
# for path in backgrounds_paths:
#     background = cv2.resize(cv2.imread(path), (512, 512))
#     backgrounds.append(background)
# # print(len(backgrounds))
#
# for i in range(1000):
#     j = i % len(backgrounds)
#     ground_truth, result_image, mask, ground_truth_without_bg, highlight_image_without_bg = generate_imgs(
#         backgrounds[j], 0.8)
#
#     image_name = 'image' + str(i) + '.png'
#
#     filename = os.path.join(gt_dir, image_name)
#     cv2.imwrite(filename, ground_truth)
#     print(f"Saved: {filename}")
#
#     filename = os.path.join(img_dir, image_name)
#     cv2.imwrite(filename, result_image)
#     print(f"Saved: {filename}")
#
#     filename = os.path.join(mask_dir, image_name)
#     cv2.imwrite(filename, mask)
#     print(f"Saved: {filename}")
#
#     filename = os.path.join(gt_nobg_dir, image_name)
#     cv2.imwrite(filename, ground_truth_without_bg)
#     print(f"Saved: {filename}")
#
#     filename = os.path.join(img_nobg_dir, image_name)
#     cv2.imwrite(filename, highlight_image_without_bg)
#     print(f"Saved: {filename}")
#
# transform_imgs(mask_dir='dataset/mask',
#                image_dir='dataset/chessboard_images',
#                img_with_corners_dir='dataset/corners',
#                output_dir='dataset/masked',
#                corners_csv_path='dataset/corners.csv',
#                offset=0)
#
# transform_imgs(mask_dir='dataset/mask',
#                image_dir='dataset/ground truth',
#                img_with_corners_dir='dataset/corners gt',
#                output_dir='dataset/masked gt',
#                corners_csv_path='dataset/corners.csv',
#                offset=0)
#
# endtime = time.time()
# print(f'Total run time: {endtime - starttime}')


class DatasetGenerator:
    def __init__(self, directory, background_dir, dataset_size):
        self.size = dataset_size
        self.background_dir = background_dir
        self.gt_dir = os.path.join(directory, 'ground truth')
        self.img_dir = os.path.join(directory, 'chessboard_images')
        self.mask_dir = os.path.join(directory, 'mask')
        self.gt_nobg_dir = os.path.join(directory, 'ground truth without bg')
        self.img_nobg_dir = os.path.join(directory, 'chessboard_images without bg')
        self.dir = directory

        os.makedirs(self.gt_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        os.makedirs(self.gt_nobg_dir, exist_ok=True)
        os.makedirs(self.img_nobg_dir, exist_ok=True)



        self.backgrounds = []
        backgrounds_paths = glob.glob(os.path.join(background_dir, '*.jpg'))
        for path in backgrounds_paths:
            background = cv2.resize(cv2.imread(path), (512, 512))
            self.backgrounds.append(background)

    def go(self):
        print(self.gt_dir)
        os.makedirs(self.gt_dir, exist_ok=True)

    def generate(self):
        for i in range(self.size):
            j = i % len(self.backgrounds)
            ground_truth, result_image, mask, ground_truth_without_bg, highlight_image_without_bg = generate_imgs(
                self.backgrounds[j], 0.8)

            image_name = 'image' + str(i) + '.png'

            filename = os.path.join(self.gt_dir, image_name)
            cv2.imwrite(filename, ground_truth)
            print(f"Saved: {filename}")

            filename = os.path.join(self.img_dir, image_name)
            cv2.imwrite(filename, result_image)
            print(f"Saved: {filename}")

            filename = os.path.join(self.mask_dir, image_name)
            cv2.imwrite(filename, mask)
            print(f"Saved: {filename}")

            filename = os.path.join(self.gt_nobg_dir, image_name)
            cv2.imwrite(filename, ground_truth_without_bg)
            print(f"Saved: {filename}")

            filename = os.path.join(self.img_nobg_dir, image_name)
            cv2.imwrite(filename, highlight_image_without_bg)
            print(f"Saved: {filename}")

        transform_imgs(mask_dir=self.mask_dir,
                       image_dir=self.img_dir,
                       img_with_corners_dir=os.path.join(self.dir, 'corners'),
                       output_dir=os.path.join(self.dir, 'masked'),
                       corners_csv_path=os.path.join(self.dir, 'corners.csv'),
                       offset=0)

        transform_imgs(mask_dir=self.mask_dir,
                       image_dir=self.gt_dir,
                       img_with_corners_dir=os.path.join(self.dir, 'corners gt'),
                       output_dir=os.path.join(self.dir, 'masked gt'),
                       corners_csv_path=os.path.join(self.dir, 'corners.csv'),
                       offset=0)


g = DatasetGenerator('t', 'backgrounds', 20)
g.generate()
