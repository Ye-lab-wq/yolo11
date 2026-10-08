# 权重复现与受控消融

本目录分为两部分：作者现存 `best.pt` 的 valid/test 复核，以及基于 `DetectDataset_clean_v2` 的重新训练与模块消融。前者保留作者权重语义；后者的脚本、模型 YAML 和结果集中放在 `ablation/`、`results/`。

文件：

- `evaluate_all_existing_best_on_test.py`：依次测试 `runs/detect/` 和 `runs/train/` 中全部 20 个 `best.pt`；默认测试 test，可用 `--split val` 切换验证集。
- `data.yaml`：指向当前完整 `DetectDataset/` 的复测数据配置。
- `results/all_existing_best_test/`：20 个权重的 CSV、JSON 和 Ultralytics 输出。
- `results/all_existing_best_val/`：同一批权重在验证集上的 CSV、JSON 和 Ultralytics 输出。
- `results/lightweight_main_models.csv`：五个论文主模型在统一 RTX 3090/PyTorch FP32 口径下的参数量、GFLOPs、权重体积、延迟、FPS 和显存。
- `FIRST_STAGE_AUDIT.md`：数据规模/目标尺寸/重复泄漏审计，以及 YOLO11n 按尺寸、IoU 和重复状态分层的 failure analysis。
- `analyze_dataset.py`：可重复生成 `results/dataset_audit.json`。
- `analyze_model_failures.py`：可重复生成 `results/failure_analysis_{split}.json`；只推理，不训练。
- `RESULTS.md`：验证集、测试集、训练日志峰值及使用边界汇总。
- `ablation/`：统一训练、验证集选权重、测试与复杂度测量脚本及预定实验协议。
- `results/clean_v2_200_3seed/RESULTS.md`：五组结构各三种子的 200 轮公开测试与同机测速汇总。
- `results/p2_msef_adown_visual_evidence_seed1/`：P2、MSEF、ADown 的逐模块可视化图和源数据。
- `results/motion_blur_training_3seed/` 与 `results/motion_blur_visual_analysis_3seed/`：三种子动态模糊结果及清晰/模糊场景可视化。

运行环境固定为 `/home/b520/anaconda3/envs/YHP`，命令：

```bash
cd /home/b520/Downloads/yelin/yolo11
/home/b520/anaconda3/envs/YHP/bin/python reproduction/evaluate_all_existing_best_on_test.py

# 验证集
/home/b520/anaconda3/envs/YHP/bin/python reproduction/evaluate_all_existing_best_on_test.py --split val
```

上面的批量复核脚本只测试作者现存 `best.pt`，不训练、不测试 `last.pt`。受控训练使用 `ablation/` 下的独立入口，例如 `run_final_candidate.py`；其权重由 valid 选出，测试 JSON 保存在相应的 `runs/` 目录。已多次查看的重建数据集测试划分应按历史基准报告，外部零微调测试需使用新的来源。
