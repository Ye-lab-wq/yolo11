import cv2
import os
import random
from pathlib import Path

def create_image_grid(image_paths, grid_size=(4,4), output_path='grid.jpg', scale=1.0):
    """
    将多张图片拼接成网格
    :param image_paths: 图片路径列表（长度需 >= grid_size[0]*grid_size[1]）
    :param grid_size: 网格的行列数 (rows, cols)
    :param output_path: 输出路径
    :param scale: 缩放因子（可统一缩小图片）
    """
    rows, cols = grid_size
    assert len(image_paths) >= rows * cols, "图片数量不足"

    # 读取所有图片并缩放至统一尺寸（以第一张图尺寸为基准）
    imgs = []
    for i in range(rows * cols):
        img = cv2.imread(image_paths[i])
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {image_paths[i]}")
        # 可选缩放
        if scale != 1.0:
            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        imgs.append(img)

    # 获取单张图片的尺寸（假设所有图片尺寸相同，否则需要统一处理）
    h, w = imgs[0].shape[:2]

    # 创建空白画布
    grid_h = h * rows
    grid_w = w * cols
    grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    # 填充图片
    for idx, img in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        grid_img[r*h:(r+1)*h, c*w:(c+1)*w] = img

    # 保存
    cv2.imwrite(output_path, grid_img)
    print(f"网格图片已保存至: {output_path}")

if __name__ == '__main__':
    import numpy as np

    # 设置数据集图片文件夹路径（请修改为你的实际路径）
    images_dir = Path('/home/b520/Downloads/yuheping/DetectDataset/train/images')
    all_images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    random.shuffle(all_images)

    # 选取前16张图片（可根据需要调整网格大小）
    selected = all_images[:16]

    # 拼接并保存
    create_image_grid(selected, grid_size=(4,4), output_path='dataset_grid.jpg', scale=0.5)