import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
from pathlib import Path
import numpy as np
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json


# --- 1. 数据集类 ---
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
        label_path = self.label_dir / (img_path.stem + '.txt')
        boxes, labels = [], []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    class_id = int(parts[0]) + 1
                    x_c, y_c, bw, bh = parts[1:5]
                    x1, y1 = (x_c - bw / 2) * w, (y_c - bh / 2) * h
                    x2, y2 = (x_c + bw / 2) * w, (y_c + bh / 2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx]),
                  'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                  'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)}
        return F.to_tensor(image), target

    def __len__(self):
        return len(self.imgs)


# --- 2. 核心评估函数 ---
def evaluate():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"正在使用设备: {device}")

    # 加载模型
    model = fasterrcnn_resnet50_fpn(min_size=640, max_size=640)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    checkpoint_path = 'faster_rcnn_epoch_30.pth'
    if not os.path.exists(checkpoint_path):
        print(f"错误：找不到权重文件 {checkpoint_path}")
        return
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()

    # 准备验证集
    val_dataset = YOLODataset('DetectDataset/valid/images', 'DetectDataset/valid/labels')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # 构建 COCO 格式的真值数据
    print("正在构建真值 (Ground Truth) 索引...")
    results = []
    coco_gt_json = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "drone"}]}
    ann_id = 1

    for idx in range(len(val_dataset)):
        _, target = val_dataset[idx]
        img_id = target["image_id"].item()
        coco_gt_json["images"].append({"id": img_id})
        for i, box in enumerate(target["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            coco_gt_json["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": target["labels"][i].item(),
                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0
            })
            ann_id += 1

    with open('tmp_gt.json', 'w') as f:
        json.dump(coco_gt_json, f)
    coco_gt = COCO('tmp_gt.json')

    # 开始推理
    print("开始推理验证集...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                img_id = target["image_id"].item()
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for b, s, l in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = b
                    results.append({
                        "image_id": img_id, "category_id": int(l),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(s)
                    })
            if (i + 1) % 100 == 0: print(f"已处理 {i + 1} / {len(val_dataset)}")

    # 计算指标
    if results:
        with open('tmp_res.json', 'w') as f:
            json.dump(results, f)
        coco_dt = coco_gt.loadRes('tmp_res.json')
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # 清理临时文件
        os.remove('tmp_gt.json')
        os.remove('tmp_res.json')
    else:
        print("未检测到任何目标。")


if __name__ == "__main__":
    evaluate()