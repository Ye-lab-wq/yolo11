# 固定种子 1：200 轮结构消融结果

本文件只比较随机种子 `1` 下的五种结构，不汇总不同种子，也不拼接不同 epoch 的指标。所有模型均从零训练，训练 `200` 轮，以清晰验证集 mAP50–95 选择各自的 `best.pt`。训练配置为 `640×640`、batch 16、AdamW、CIoU、AMP 关闭、`deterministic: true`，其余超参数见各运行的 `args.yaml`。

数据均来自同一个 [`DetectDataset_clean_v2`](../../../DetectDataset_clean_v2/data.yaml)：训练 2,947 张（含 36 张自采视频帧），验证 625 张，完整测试 641 张（含另 16 张自采视频帧、701 个框）。测试中的自采帧与其他帧一起输入同一评估器，不单独加权或更改规则。这里的测试指标全部来自 Ultralytics 检测评估器，不能与另一个 COCO-style 评估器的数值混用。

## 验证集选择的 checkpoint

下表来自每次训练冻结时记录的验证集指标；一行指标对应**同一轮**。`best epoch` 是从 1 开始的训练轮数。原始 `frozen_model_seed1.json` 中的字段名 `best_epoch_zero_based` 不准确，其记录的数值实际与 `results.csv` 的从 1 开始的 `epoch` 列相同。训练后重新运行验证评估器可能与训练日志有很小的数值差异，不应将两者混成一行。

| 模型 | best epoch | 验证 P/% | 验证 R/% | 验证 mAP50/% | 验证 mAP50–95/% |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 140 | 89.94 | 91.40 | 93.76 | 56.63 |
| YOLO11n+P2 | 149 | 90.83 | 91.10 | 94.37 | 57.63 |
| YOLO11n+P2+MSEFPaper | 182 | 90.62 | 91.53 | 95.01 | 58.58 |
| YOLO11n+P2+ADown | 158 | 90.20 | 91.39 | 94.70 | 58.25 |
| YOLO11n+P2+MSEFPaper+ADown | 193 | 92.52 | 92.12 | 94.93 | **58.60** |

## 完整 641 张测试集：固定种子 1

每行均评估该结构由验证集选出的 `best.pt`。P、R、mAP50、AP75、mAP50–95 均是**同一次**完整测试评估的指标。

| 模型 | P/% | R/% | mAP50/% | AP75/% | mAP50–95/% |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 93.69 | 93.25 | 95.20 | 61.84 | 57.90 |
| YOLO11n+P2 | 94.42 | 94.21 | **96.33** | 63.98 | 58.78 |
| YOLO11n+P2+MSEFPaper | 94.03 | **94.30** | 96.15 | 66.27 | 58.65 |
| YOLO11n+P2+ADown | 93.69 | 93.25 | 96.01 | 66.07 | 59.54 |
| YOLO11n+P2+MSEFPaper+ADown | **94.89** | 92.65 | 95.75 | **66.46** | **59.63** |

测试指标相对基线的变化，单位为百分点：

| 模型 | ΔP | ΔR | ΔmAP50 | ΔAP75 | ΔmAP50–95 |
|---|---:|---:|---:|---:|---:|
| +P2 | +0.73 | +0.96 | +1.13 | +2.15 | +0.88 |
| +P2+MSEFPaper | +0.33 | +1.05 | +0.96 | +4.43 | +0.74 |
| +P2+ADown | +0.00 | +0.00 | +0.82 | +4.23 | +1.63 |
| +P2+MSEFPaper+ADown | +1.19 | −0.60 | +0.56 | +4.62 | +1.73 |

按直接父模型比较，种子 1 的 `P2+MSEFPaper` 比 `P2` 的 mAP50–95 **低 0.14** 个百分点；完整模型比 `P2+ADown` **高 0.10** 个百分点，同时 R **低 0.60**、mAP50 **低 0.26** 个百分点。这不能说明 MSEFPaper 稳定有效。该种子下完整模型在验证与测试 mAP50–95 都最高，但测试集此前已用于多次方法分析，因此这是既有结果的描述，不是新的盲测证据；不得把“事后挑到种子 1”表述为预先固定的研究设计。

## 计算开销与源文件

现有 RTX 3090、640 输入、FP32 融合模型前向基准测量如下。FPS 仅包括模型前向，不包括读图、预处理或 NMS，不等于完整检测系统的端到端 FPS。

| 模型 | 参数/M | GFLOPs | FP32 batch-1 FPS |
|---|---:|---:|---:|
| YOLO11n | 2.590 | 6.44 | 173.5 |
| +P2 | 2.904 | 10.83 | 146.3 |
| +P2+MSEFPaper | 2.920 | 11.69 | 143.9 |
| +P2+ADown | 2.278 | 9.47 | 135.7 |
| +P2+MSEFPaper+ADown | 2.295 | 10.33 | 132.2 |

| 模型 | 种子 1 测试 JSON | 权重位置（仅本地，不上传 Git） |
|---|---|---|
| YOLO11n | [`combined_test.json`](../../../runs/final_baseline_yolo11n_200/test_results/seed1/combined_test.json) | `runs/final_baseline_yolo11n_200/train_seed1/weights/best.pt` |
| +P2 | [`combined_test.json`](../../../runs/ablation_200_v1/p2/test_results/seed1/combined_test.json) | `runs/ablation_200_v1/p2/train_seed1/weights/best.pt` |
| +P2+MSEFPaper | [`combined_test.json`](../../../runs/ablation_200_v1/p2_msef_paper/test_results/seed1/combined_test.json) | `runs/ablation_200_v1/p2_msef_paper/train_seed1/weights/best.pt` |
| +P2+ADown | [`combined_test.json`](../../../runs/ablation_200_v1/p2_adown/test_results/seed1/combined_test.json) | `runs/ablation_200_v1/p2_adown/train_seed1/weights/best.pt` |
| +P2+MSEFPaper+ADown | [`combined_test.json`](../../../runs/final_candidate_p2_msef_adown_s0_200/test_results/seed1/combined_test.json) | `runs/final_candidate_p2_msef_adown_s0_200/train_seed1/weights/best.pt` |

各运行目录中还保留 `args.yaml`、`results.csv` 与 `frozen_model_seed1.json`；模型结构 YAML 在 [`reproduction/ablation/models`](../../ablation/models/)。本文件未使用 60 轮模糊训练结果、625 张测试子范围结果或 COCO-style AP。
