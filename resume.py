from ultralytics import YOLO

# 1. 指向你刚才跑了一半的那个权重文件
# 注意：一定要确认这个路径是正确的
path_to_last = '/home/b520/Downloads/yuheping/runs/detect/train9/weights/last.pt'

# 2. 加载这个模型
model = YOLO(path_to_last)

# 3. 开启断点续训 (resume=True)
# 注意：resume 模式下不需要再传 data, epochs, imgsz, batch 等参数
# 它会自动接着上次中断的那一刻（包括学习率、进度等）继续跑
model.train(resume=True)