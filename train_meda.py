import os
import sys
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 依然加载你最好的权重
    model = YOLO('yolo11n-MEDA-Pro.yaml')

    model.train(
        data='/home/b520/Downloads/yuheping/DetectDataset/data.yaml',
        epochs=200,
        batch=16,
        imgsz=640,

        # 🚀 关键修改：取消 resume=True，改为手动指定开始轮次
        # 这样系统会把它当成一个“新任务”，从而允许我们修改超参数
        resume=False,

        # 🚀 强制覆盖超参数
        box=7.5,  # 强制拉高回归权重
        lr0=0.001,  # 使用较小的学习率防止震荡

        amp=False,
        warmup_epochs=0,
        workers=4
    )