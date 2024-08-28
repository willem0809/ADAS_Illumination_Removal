import glob
import os
import torch
from torch.utils.data import DataLoader

from dataset import MyDataset, MyDataset2
from model import Unet1, Unet2
from save_results import save_images, save_masks
from transform import transform_imgs
from transform_inverse import inverse_transform_imgs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, data_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for image in data_loader:
            image = image.to(device)
            outputs = model(image)
            predictions.append(outputs.cpu())

    return predictions


def clear_directory(directory_path):
    # 确保目录路径存在
    if not os.path.exists(directory_path):
        print(f"目录 {directory_path} 不存在。")
        return

    # 获取目录内所有文件的路径
    file_paths = glob.glob(os.path.join(directory_path, '*'))

    # 删除每个文件
    for file_path in file_paths:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"已删除: {file_path}")
            elif os.path.isdir(file_path):
                # 如果是目录，则不删除，保留目录结构
                print(f"保留目录: {file_path}")
        except Exception as e:
            print(f"删除文件 {file_path} 时发生错误: {e}")


class Predict:
    def __init__(self, imgs_dir='training dataset/dataset/chessboard_images',
                 model_data=None, mean=None, std=None):

        self.model1 = Unet1(3, 1).to(device)
        self.model1.load_state_dict(torch.load(model_data[0]))
        self.model1.eval()

        self.model2 = Unet2(3, 3).to(device)
        self.model2.load_state_dict(torch.load(model_data[1]))
        self.model2.eval()

        self.imgs_dir = imgs_dir

        self.dir = 'predict'
        self.mask_dir = os.path.join(self.dir, 'masks')
        self.img_with_corners_dir = os.path.join(self.dir, 'img_with_corners')
        self.masked_dir = os.path.join(self.dir, 'masked_imgs')
        self.masked_gt_dir = os.path.join(self.dir, 'masked_gt')
        self.corners_path = os.path.join(self.dir, 'corners.csv')
        self.results_dir = os.path.join(self.dir, 'results')
        clear_directory('predict')
        os.makedirs(self.results_dir, exist_ok=True)

        if mean and std:
            self.mean, self.std = mean, std
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

    def start(self):

        dataset = MyDataset(image_dir=self.imgs_dir)
        data_loader = DataLoader(dataset, shuffle=False)

        masks = predict(self.model1, data_loader)

        for i, mask in enumerate(masks):

            save_masks(self.mask_dir, mask, i)

        print("Model 1 Finish")

        # transform_imgs(offset=2, corners_csv_path='predict/corners.csv')
        transform_imgs(mask_dir=self.mask_dir,
                       image_dir=self.imgs_dir,
                       img_with_corners_dir=self.img_with_corners_dir,
                       output_dir=self.masked_dir,
                       corners_csv_path=self.corners_path,
                       offset=2)
        print('image transform completed')

        dataset2 = MyDataset(image_dir=self.masked_dir)
        data_loader2 = DataLoader(dataset2, shuffle=False)

        predictions = predict(self.model2, data_loader2)

        for i, (pred, mask) in enumerate(zip(predictions, masks)):
            save_images(self.masked_gt_dir, pred, i)

        print("Model 2 Finish")

        inverse_transform_imgs(self.imgs_dir,
                               self.masked_gt_dir,
                               self.corners_path,
                               self.results_dir)
        print('complete')


model_datas = ['model_data/model1_data/model1 good1.pth',
               'model_data/model2_data/model2 good.pth']

p = Predict(model_data=model_datas)
p.start()

