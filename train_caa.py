import sys
import os
import torch

# 将当前目录加入 Python 路径，确保能导入 CAA 模块
sys.path.append(os.path.dirname(__file__))
from CAA import CAA

# 关键修复：将 CAA 注入到 parse_model 函数的全局变量中
import ultralytics.nn.tasks
ultralytics.nn.tasks.parse_model.__globals__['CAA'] = CAA

# 现在导入 YOLO，解析时就能找到 CAA 了
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载改进后的模型配置
    model = YOLO('yolo11-CAA.yaml')
   # model = YOLO('yolo11n.pt')

    # 开始训练（请根据实际情况修改 data 路径）
    results = model.train(
        data='/home/b520/Downloads/yuheping/DetectDataset/data.yaml',  # 你的数据集配置文件
        epochs=150,                 # 训练轮数
        imgsz=640,                   # 输入图片大小
        batch=16,                     # 批次大小（显存不足时减小）
        device=0,                      # GPU 设备号
        workers=4,                     # 数据加载线程数
        project='runs/train',        # 结果保存目录
        #name='yolov11_caa',          # 实验名称
        exist_ok=False,                 # 允许覆盖
        pretrained=False,              # 从零开始训练
        optimizer='AdamW',             # 优化器
        lr0=0.001,                      # 初始学习率
        amp=False                       # 混合精度（False 更稳定）
    )

    print("✅ 训练完成！模型保存在 runs/train/yolov11_caa/weights/")