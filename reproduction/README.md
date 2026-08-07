# 权重复现

本目录只保存本次“使用作者现存 `best.pt` 在独立 test split 上复测”的最小材料，不包含训练代码或新模型实现。

文件：

- `evaluate_all_existing_best_on_test.py`：依次测试 `runs/detect/` 和 `runs/train/` 中全部 20 个 `best.pt`。
- `data.yaml`：指向当前完整 `DetectDataset/` 的复测数据配置。
- `results/all_existing_best_test/`：20 个权重的 CSV、JSON 和 Ultralytics 输出。
- `RESULTS.md`：测试结果汇总与使用边界。

运行环境固定为 `/home/b520/anaconda3/envs/YHP`，命令：

```bash
cd /home/b520/Downloads/yelin/yolo11
/home/b520/anaconda3/envs/YHP/bin/python reproduction/evaluate_all_existing_best_on_test.py
```

该脚本只测试现存 `best.pt`，不训练、不测试 `last.pt`，也不根据 test 结果重新选择论文最终模型。
