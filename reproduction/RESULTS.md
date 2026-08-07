# 全部现存 best.pt 的独立测试集审计

## 结论

- 当前迁移目录共发现 20 个 `weights/best.pt`，20 个 SHA256 均不同。
- 20 个权重均已在同一个独立 test split 上完成评估，失败数为 0。
- 本次没有训练、没有根据 test 选择 checkpoint，也没有测试任何 `last.pt`。
- test split 固定为 560 images / 608 instances；推理参数统一为 `imgsz=640`、`batch=16`。
- 唯一环境为 `/home/b520/anaconda3/envs/YHP`：Python 3.12.0、PyTorch 2.10.0+cu128、本地定制 Ultralytics 8.3.241、RTX 3090。

## 论文主表对应模型

下表只列已有文件能够明确映射到论文主表/消融链条的模型。数值均为本次独立 test 结果，不是训练日志中的 val 结果。

| 模型 | 当前目录中的 best.pt | P/% | R/% | mAP50/% | mAP50:95/% |
|---|---|---:|---:|---:|---:|
| YOLOv8n | `runs/detect/yolo8_100/weights/best.pt` | 95.42 | 95.90 | 97.70 | 62.80 |
| YOLO11n | `runs/detect/yolo11_100/weights/best.pt` | 96.94 | 95.89 | 97.74 | 61.75 |
| YOLO11n + CAA | `runs/train/train_c_120/weights/best.pt` | 95.69 | 94.92 | 96.58 | 60.96 |
| YOLO11n + CAA + Focaler | `runs/train/train_ic_150_3_val/weights/best.pt` | 92.79 | 95.27 | 96.01 | 60.97 |
| 最终 MEDA | `runs/detect/meda_200/weights/best.pt` | 94.17 | 95.58 | 97.35 | 63.68 |

最终 MEDA 相对指定 YOLO11n 基线的 test mAP50:95 提升为 1.92 个百分点（63.68−61.75），不是论文表述的 4.59 个百分点；P、R、mAP50 分别变化 −2.77、−0.30、−0.39 个百分点。

## 全部历史 best.pt

以下仅用于归档和审计，按 test mAP50:95 排序。不能从这些历史探索运行中再挑最高者作为论文最终模型，否则 test 就被用于模型选择，失去独立测试集的意义。

| 排名 | 运行目录 | 参数量 | P/% | R/% | mAP50/% | mAP50:95/% |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `train/train_meda2_datafiou_150` | 20,061,836 | 93.68 | 95.13 | 97.04 | 63.84 |
| 2 | `train/train_meda2_150` | 20,061,836 | 94.69 | 95.89 | 97.12 | 63.82 |
| 3 | `detect/meda_200` | 10,825,268 | 94.17 | 95.58 | 97.35 | 63.68 |
| 4 | `detect/yolo8_100` | 3,011,043 | 95.42 | 95.90 | 97.70 | 62.80 |
| 5 | `train/train_meda2_datawiou_150` | 20,061,836 | 94.18 | 96.05 | 97.16 | 62.32 |
| 6 | `train/train_yolo_100_3` | 2,590,035 | 95.41 | 95.75 | 97.64 | 62.32 |
| 7 | `train/train_yolo_100` | 2,590,035 | 96.45 | 95.07 | 97.85 | 62.02 |
| 8 | `detect/yolo11_100` | 2,590,035 | 96.94 | 95.89 | 97.74 | 61.75 |
| 9 | `train/meda_yolo` | 2,456,915 | 94.86 | 95.07 | 97.30 | 61.42 |
| 10 | `train/train_meda1_150` | 3,344,035 | 94.82 | 96.31 | 97.47 | 61.02 |
| 11 | `train/train_ic_150_3_val` | 2,777,299 | 92.79 | 95.27 | 96.01 | 60.97 |
| 12 | `train/train_c_120` | 2,777,299 | 95.69 | 94.92 | 96.58 | 60.96 |
| 13 | `train/train_yolo_100_2` | 2,590,035 | 96.34 | 95.72 | 97.44 | 60.92 |
| 14 | `train/train_ci_150` | 2,777,299 | 93.42 | 95.67 | 96.47 | 60.90 |
| 15 | `train/train_ic_150_3` | 2,777,299 | 92.75 | 94.63 | 95.90 | 60.70 |
| 16 | `train/train_meda1_100` | 3,344,035 | 95.93 | 94.24 | 97.17 | 60.54 |
| 17 | `train/train_ic_150_2` | 2,777,299 | 94.79 | 95.81 | 97.09 | 60.36 |
| 18 | `train/train_ci_120` | 2,777,299 | 95.09 | 95.55 | 96.96 | 60.19 |
| 19 | `train/train_c_100` | 2,777,299 | 94.47 | 95.49 | 97.10 | 59.46 |
| 20 | `train/train` | 10,825,268 | 46.96 | 26.15 | 28.60 | 8.71 |

## 兼容性与权重口径

- `train/meda_yolo` 的旧 checkpoint 将项目内 `dysample.py` 序列化成顶层模块名。评估脚本只为它建立了原文件的导入别名。
- `train/train_meda1_100` 和 `train/train_meda1_150` 保存的是旧版 MSEF 对象。原项目 `meda_modules.py` 中仍保留该版本的完整自适应池化前向代码；评估脚本按对象字段自动调用这段旧公式，并在 CSV/JSON 的 `compatibility_adapter` 中标注。
- 兼容处理没有修改 checkpoint、数据、阈值或指标公式。
- CSV 中的 `model_arg` 是各运行 `args.yaml` 记录的训练初始化来源。即使其中两项文字指向原目录的 `last.pt`，本次实际传给评估器并计算 SHA256 的文件始终是当前目录对应运行下的 `weights/best.pt`，没有测试 `last.pt`，也没有拼接多个 checkpoint。

## 可复核产物

- 完整机器可读结果：`reproduction/results/all_existing_best_test/all_best_test_results.json`
- 完整表格：`reproduction/results/all_existing_best_test/all_best_test_results.csv`
- 批量评估脚本：`reproduction/evaluate_all_existing_best_on_test.py`
- 每次 Ultralytics 输出：`reproduction/results/all_existing_best_test/runs/`

## 数据集职责

- train：参数更新。
- val：观察训练、调参并选择各运行的 `best.pt`。
- test：模型及选择规则冻结后做最终评估。此次“全部历史权重上 test”属于一次性完整审计；这些排名不能再反馈到模型设计或最终模型选择。
