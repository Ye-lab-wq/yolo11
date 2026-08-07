import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
import numpy as np
import os

# --- 1. 自定义数据集类，用于加载YOLO格式的标注 ---
class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        self.imgs = list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png'))

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        image = Image.open(img_path).convert('RGB')
        w, h = image.size

        # 读取对应的 .txt 文件
        label_path = self.label_dir / (img_path.stem + '.txt')
        boxes = []
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    class_id = int(parts[0]) + 1  # Faster R-CNN的类别ID必须从1开始
                    x_c, y_c, bw, bh = parts[1:5]
                    # 将YOLO格式(x_center, y_center, w, h)转换为(x1, y1, x2, y2)
                    x1 = (x_c - bw/2) * w
                    y1 = (y_c - bh/2) * h
                    x2 = (x_c + bw/2) * w
                    y2 = (y_c + bh/2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)

        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # labels = torch.as_tensor(labels, dtype=torch.int64)
                # 处理背景图片（没有目标的空标签情况）
                if len(boxes) == 0:
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.zeros((0,), dtype=torch.int64)
                else:
                    boxes = torch.as_tensor(boxes, dtype=torch.float32)
                    labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'area': area, 'iscrowd': iscrowd}

        if self.transforms:
            image = self.transforms(image)

        return F.to_tensor(image), target

    def __len__(self):
        return len(self.imgs)

# --- 2. 初始化数据集和 DataLoader ---
train_dataset = YOLODataset('DetectDataset/train/images', 'DetectDataset/train/labels')
#train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
# 修改第 57 行的 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=8,         # 开启 8 个子进程在后台为你提前搬运数据
    pin_memory=True,       # 加速 CPU 到 GPU 的数据拷贝
    collate_fn=lambda x: tuple(zip(*x))
)

# --- 3. 加载预训练模型并修改分类头 (关键步骤) ---
# 加载在COCO上预训练的Faster R-CNN模型
print("正在加载 Faster R-CNN 预训练模型，请稍候...")
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights='DEFAULT',
    min_size=640,  # 限制最小边
    max_size=640   # 限制最大边
)
# 获取输入特征数（分类头前一层的维度）
in_features = model.roi_heads.box_predictor.cls_score.in_features
# 替换分类头，类别数 = 你的类别数 + 背景类(1)
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2) # 1 (drone) + 1 (background)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("正在将庞大的模型搬运至 RTX 3090 (这通常需要 3-5 秒建立 CUDA 环境)...")
model.to(device)
print("模型已就绪，开始训练！")
model.train()

# --- 4. 设置优化器 ---
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# --- 5. 开始训练循环 ---
num_epochs = 30
for epoch in range(num_epochs):
    total_loss = 0
    for imgs, targets in train_loader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}')
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), f'faster_rcnn_epoch_{epoch+1}.pth')

print("训练完成！")