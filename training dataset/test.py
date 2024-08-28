from transform import transform_imgs, mark_corners_from_csv

transform_imgs(mask_dir='dataset/mask',
               image_dir='dataset/chessboard_images',
               img_with_corners_dir='dataset/corners',
               output_dir='dataset/masked1',
               corners_csv_path='dataset/corners.csv',
               offset=2)

# mark_corners_from_csv('training dataset/dataset/ground truth',
#                       'training dataset/dataset/corners.csv',
#                       'training dataset/dataset/corners gt')

# transform_imgs(mask_dir='dataset/mask',
#                image_dir='dataset/ground truth',
#                img_with_corners_dir='dataset/corners gt',
#                output_dir='dataset/masked gt',
#                corners_csv_path='dataset/corners.csv',
#                offset=2)
