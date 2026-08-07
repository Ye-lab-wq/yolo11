import cv2
from ultralytics import YOLO
import numpy as np

# 加载模型
model = YOLO('runs/train/train_meda2_datafiou_150/weights/best.pt')

# 打开视频文件
video_path = 'original_vedio.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 获取视频参数
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 定义输出视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video3.mp4', fourcc, fps, (width, height))

# 置信度阈值
conf_threshold = 0.7

print(f"开始处理视频: {video_path} (共 {total_frames} 帧)")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 推理
    results = model(frame, imgsz=640, conf=conf_threshold, iou=0.5, verbose=False)
    boxes_data = results[0].boxes

    # 如果检测到物体
    if boxes_data is not None and len(boxes_data) > 0:
        # 获取边界框坐标 (xyxy格式)，置信度，类别ID
        xyxy = boxes_data.xyxy.cpu().numpy().astype(int)
        confs = boxes_data.conf.cpu().numpy()
        # 类别ID（如果有多类，可用于过滤特定类别，这里只显示无人机一类）
        # cls_ids = boxes_data.cls.cpu().numpy()

        # 根据置信度过滤
        for i, conf in enumerate(confs):
            if conf >= conf_threshold:
                x1, y1, x2, y2 = xyxy[i]
                # 绘制矩形框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
                # 显示置信度文本
                label = f'{conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 写入输出视频
    out.write(frame)

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"已处理: {frame_count}/{total_frames} 帧...")

cap.release()
out.release()
print("\n处理完成！结果已保存至 output_video3.mp4")