import cv2
from ultralytics import YOLO
from pathlib import Path

model = YOLO('/home/b520/Downloads/yuheping/runs/train/train_yolo_100/weights/best.pt')
val_img_dir = Path('/home/b520/Downloads/yuheping/v5811_pic')
output_dir = Path('runs/detect/v5811_pic')
output_dir.mkdir(parents=True, exist_ok=True)

for img_path in val_img_dir.glob('*.*'):
    if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
        continue
    results = model(img_path, conf=0.3)
    plotted = results[0].plot()  # 默认会绘制置信度
    cv2.imwrite(str(output_dir / img_path.name), plotted)

print(f"完成，结果保存在 {output_dir}")