from ultralytics import YOLO

model = YOLO("yolo11n.pt")

train_results = model.train(
    data="/home/b520/Downloads/yuheping/DetectDataset/data.yaml",
    epochs=20,
    imgsz=640,
    device="0",
    batch=16,
    save=True,
    verbose=True,
    save_period=5,
    project="runs/train",
    workers=0
)

metrics = model.val()

results = model("path/to/image.jpg")
results[0].show()

path = model.export(format = "onnx")