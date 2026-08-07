# YOLO11 微小无人机检测项目

本目录用于整理并复现论文第三章对应的 YOLO11 无人机检测项目。当前阶段的主要任务是：使用作者提交的现存 `best.pt`，在固定验证集或独立测试集上复核结果；目前不进行重新训练。

## 1. 文件来源

当前目录不是完全由作者 ZIP 单独提供，而是由两部分作者材料合并恢复：

- 作者 ZIP：根目录训练/推理脚本、CAA、模型 YAML、YOLOv5 基线目录、示例图片和其他项目素材。
- 服务器原项目 `/home/b520/Downloads/yuheping`：补齐完整定制 `ultralytics/`、完整数据集以及 `runs/` 中的权重、CSV 和参数记录。

由本次整理新增的内容只有：

- `paper_chapter3/`：按要求从论文中单独提取的第三章及低 Token 摘要。
- `reproduction/`：现存权重测试脚本和已经生成的测试结果。
- 本 `README.md`：项目入口说明。

其余代码、模型配置、权重和数据均来自作者材料；目录位置经过整理，但没有重新实现作者模型。

## 2. 目录结构

```text
yolo11/
├── ultralytics/              # 作者定制的 Ultralytics 源码
├── DetectDataset/            # 完整数据集
│   ├── train/
│   ├── valid/
│   └── test/
├── runs/
│   ├── detect/               # YOLOv8、YOLO11、最终 MEDA 等运行
│   └── train/                # CAA、Focaler、早期 MEDA 等运行
├── yolov5/                   # 作者提供的 YOLOv5 基线工程
├── CAA.py                    # CAA 模块
├── yolo11-CAA.yaml           # CAA 网络配置
├── yolo11n-MEDA.yaml         # 早期 MEDA 网络配置
├── yolo11n-MEDA-Pro.yaml     # 最终 MEDA 网络配置
├── train_caa.py              # 作者 CAA 训练脚本
├── train_meda.py             # 作者最终 MEDA 训练脚本
├── paper_chapter3/           # 第三章材料
└── reproduction/            # 本次最小复现材料与结果
```

GitHub 仓库只跟踪 `DetectDataset/data.yaml`，不上传数据集图像与标签；`runs/` 中只跟踪各运行的 `args.yaml` 和 `results.csv`，不上传权重、图表及其他生成文件。完整数据集、权重和运行产物仍保留在当前服务器本地。

## 3. 固定环境

所有正式复核统一使用服务器原项目的 YHP 环境：

```text
Conda: /home/b520/anaconda3/envs/YHP
Python: 3.12.0
PyTorch: 2.10.0+cu128
Ultralytics evaluator: 8.3.241（当前目录定制源码）
GPU: NVIDIA GeForce RTX 3090
```

运行前进入项目根目录，保证优先导入当前 `ultralytics/`：

```bash
cd /home/b520/Downloads/yelin/yolo11
```

## 4. 数据划分与职责

复现使用 [reproduction/data.yaml](reproduction/data.yaml)：

论文第三章称数据集包含 5000 张图像并按 8:1:1 划分，但当前作者材料中可读取到 5638 张图像，实际比例约为 75.13%:14.93%:9.93%。因此，下表描述的是当前交付数据，而不是对论文数量表述的转录；详细核对见 [reproduction/RESULTS.md](reproduction/RESULTS.md)。

| 划分 | 图像 | 目标 | 用途 |
|---|---:|---:|---|
| train | 4236 | 4493 | 反向传播和参数更新 |
| valid | 842 | 993 | 每个 epoch 验证并选择 `best.pt` |
| test | 560 | 608 | 模型和 checkpoint 冻结后的最终性能评估 |

正确流程为：

1. 使用 train 更新模型。
2. 训练期间只根据 valid 指标选择 `best.pt`。
3. 模型、超参数和选择规则冻结后，使用该 `best.pt` 在 test 上评估。
4. 不得根据 test 排名重新选择模型或继续调参。

`best.pt` 是验证集选择的 checkpoint；`last.pt` 只是最后一个训练 epoch，不能默认代替 `best.pt`。

## 5. 论文模型与权重路径

| 论文/消融模型 | 模型配置 | 当前 best.pt |
|---|---|---|
| YOLOv8n | Ultralytics YOLOv8n | `runs/detect/yolo8_100/weights/best.pt` |
| YOLO11n 基线 | `ultralytics/cfg/models/11/yolo11.yaml` | `runs/detect/yolo11_100/weights/best.pt` |
| YOLO11n + CAA | `yolo11-CAA.yaml`、`CAA.py` | `runs/train/train_c_120/weights/best.pt` |
| YOLO11n + CAA + Focaler | 与 CAA 相同拓扑，损失实现在 `ultralytics/utils/loss.py` | `runs/train/train_ic_150_3_val/weights/best.pt` |
| 最终 MEDA | `yolo11n-MEDA-Pro.yaml` | `runs/detect/meda_200/weights/best.pt` |

其他历史运行仍保存在 `runs/train/`。每个运行通常包含：

```text
args.yaml       # 训练配置
results.csv     # 各 epoch 的训练/验证日志
weights/best.pt # 验证集规则选出的 checkpoint
weights/last.pt # 最后一个 epoch
```

历史 `args.yaml` 中仍保留作者服务器的原始绝对路径，这是训练来源记录；当前读取权重应使用上表中的本项目路径。

## 6. 使用现存权重验证或测试

不需要单独编写 `val.py` 或 `test.py`。Ultralytics 使用同一个 `val` 命令，通过 `split` 参数切换数据划分。

YHP 环境中的 `yolo` 启动器默认会找到环境内安装包；为了强制使用本项目的定制 Ultralytics 8.3.241，命令必须带 `PYTHONPATH="$PWD"`。可先核对：

```bash
PYTHONPATH="$PWD" /home/b520/anaconda3/envs/YHP/bin/yolo version
# 应输出 8.3.241
```

验证集（用于复核 `best.pt` 的选择依据）：

```bash
PYTHONPATH="$PWD" /home/b520/anaconda3/envs/YHP/bin/yolo detect val \
  model=runs/detect/meda_200/weights/best.pt \
  data=reproduction/data.yaml split=val \
  imgsz=640 batch=16 device=0 workers=4 plots=False
```

独立测试集只需把一个参数改为 `split=test`：

```bash
PYTHONPATH="$PWD" /home/b520/anaconda3/envs/YHP/bin/yolo detect val \
  model=runs/detect/meda_200/weights/best.pt \
  data=reproduction/data.yaml split=test \
  imgsz=640 batch=16 device=0 workers=4 plots=False
```

终端表头中的 `Box(P)` 就是检测框 Precision，`R` 是 Recall。例如：

```text
all  560  608  0.942  0.956  0.974  0.637
                P      R      mAP50  mAP50-95
```

即 Precision=94.2%、Recall=95.6%、mAP50=97.4%、mAP50:95=63.7%。

独立 `val` 命令默认主要将汇总指标打印到终端。若设置 `plots=False`，默认生成的 `runs/detect/val*` 目录可能是空的。需要同时保存终端记录和评估图表时，可指定输出目录、打开绘图并用 `tee` 保存日志：

```bash
mkdir -p reproduction/results/manual
PYTHONPATH="$PWD" /home/b520/anaconda3/envs/YHP/bin/yolo detect val \
  model=runs/detect/meda_200/weights/best.pt \
  data=reproduction/data.yaml split=test \
  imgsz=640 batch=16 device=0 workers=4 plots=True \
  project=reproduction/results/manual name=meda_test exist_ok=True \
  2>&1 | tee reproduction/results/manual/meda_test.log
```

图表保存在 `reproduction/results/manual/meda_test/`，终端完整输出保存在 `reproduction/results/manual/meda_test.log`。批量复测脚本则会额外生成结构化 CSV 和 JSON。

切换模型时只替换 `model=` 后的权重路径。由于项目根目录含有作者的 `CAA.py`，同样的命令也可读取主要 CAA checkpoint。

一次性复核全部 20 个现存 `best.pt`：

```bash
/home/b520/anaconda3/envs/YHP/bin/python reproduction/evaluate_all_existing_best_on_test.py
```

完整结果见 [reproduction/RESULTS.md](reproduction/RESULTS.md)。

## 7. 训练如何实施

作者原始入口是 `train_meda.py`、`train_caa.py` 和 `example1_train.py`。这些文件保留了作者提交状态，其中部分数据路径仍指向 `/home/b520/Downloads/yuheping`，不要在未核对配置时直接启动。

从现有 `args.yaml` 恢复出的主要训练配置并不统一：

| 模型 | epochs | batch | optimizer | lr0 | AMP | pretrained |
|---|---:|---:|---|---:|---|---|
| YOLOv8n | 100 | 16 | auto | 0.01 | True | True |
| YOLO11n | 100 | 16 | auto | 0.01 | True | True |
| CAA | 120 | 16 | AdamW | 0.001 | False | False |
| CAA + Focaler | 150 | 16 | AdamW | 0.001 | False | False |
| 最终 MEDA | 200 | 16 | AdamW | 0.001 | False | True |

因此这些历史结果可以用于复核作者提交内容，但不能直接视为严格控制变量的公平消融。若以后重新训练，应先冻结统一协议并把新结果写入新的运行名，避免覆盖作者 `runs/`。

训练也使用同一个终端入口，将模式改成 `train`。下面只是调用方式示例，正式训练参数应以预先确定的实验协议为准：

```bash
PYTHONPATH="$PWD" /home/b520/anaconda3/envs/YHP/bin/yolo detect train \
  model=yolo11n-MEDA-Pro.yaml data=reproduction/data.yaml \
  epochs=200 batch=16 imgsz=640 device=0 workers=4 \
  project=runs/retrain name=meda_retrain exist_ok=False
```

训练完成后，读取 `runs/retrain/meda_retrain/weights/best.pt`，先在 valid 上确认选择结果，再按第 6 节的方法在 test 上做一次最终评估。

## 8. 当前复现边界

- 当前已完成的是作者现存权重的 val/test 复核，不是重新训练。
- 最终 MEDA 在独立 test 上的 mAP50:95 为 63.68%，指定 YOLO11n 基线为 61.75%，提升 1.92 个百分点。
- 论文报告的 4.59 个百分点提升不能由这两个最终 `best.pt` 在独立 test 上复现。
- `reproduction/` 中的测试排名只用于审计，不能再反馈为模型选择依据。
