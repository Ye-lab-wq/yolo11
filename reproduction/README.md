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
- `PAPER_CLAIMS_AUDIT.md`：将重新训练的内部消融、外部零样本与合成模糊实验映射到可写和不可写的论文主张。
- `ablation/`：统一训练、验证集选权重、测试与复杂度测量脚本及预定实验协议。
- `results/clean_v2_200_3seed/RESULTS.md`：五组结构各三种子的 200 轮原始来源测试子集（代码名 `public_test`）与同机测速汇总；它与自采测试子集同属一个重建数据集。
- `results/uav_eagle_external_200_3seed/RESULTS.md`：同 15 个冻结权重在独立 UAV-Eagle 来源上的零样本测试与数据局限性；原始下载和审计协议在 `external_benchmarks/uav_eagle/`。
- `results/p2_msef_adown_visual_evidence_seed1/`：P2、MSEF、ADown 的逐模块可视化图和源数据。
- `results/motion_blur_training_3seed/` 与 `results/motion_blur_visual_analysis_3seed/`：三种子动态模糊结果及清晰/模糊场景可视化。
- `results/msef_redesign_60_s12/RESULTS.md`：MSEF 三步改版与 P2、旧 MSEFPaper 的 60 轮三种子配对复核。
- `results/msef_redesign_seed1_200_v1/RESULTS.md`：同五个结构的 seed 1 / 200 轮探索性完整测试，含两套评估器、验证选模记录和结论边界。

运行环境固定为 `/home/b520/anaconda3/envs/YHP`，命令：

```bash
cd /home/b520/Downloads/yelin/yolo11
/home/b520/anaconda3/envs/YHP/bin/python reproduction/evaluate_all_existing_best_on_test.py

# 验证集
/home/b520/anaconda3/envs/YHP/bin/python reproduction/evaluate_all_existing_best_on_test.py --split val
```

上面的批量复核脚本只测试作者现存 `best.pt`，不训练、不测试 `last.pt`。受控训练使用 `ablation/` 下的独立入口，例如 `run_final_candidate.py`；其权重由 valid 选出，测试 JSON 保存在相应的 `runs/` 目录。已多次查看的重建数据集测试划分应按历史基准报告；UAV-Eagle 作为独立来源零样本测试单独汇报，不与内部测试混为一个数据集。
