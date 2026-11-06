# 200 轮 seed1：P2 与 ADown 的直接父模型可视证据

> 正文排版优先查看 `journal_qualitative_p2_adown.png` 与 `JOURNAL_STYLE_NOTES.md`。下述 `direct_parent_examples.*` 是包含失败案例的完整核查版，保留作补充材料或内部审计。

## 可查看的图与可复核来源

- `direct_parent_examples.png` 用于快速查看；同图提供 PDF、SVG、600 dpi TIFF。
- `manifest.json` 记录每张原图的相对路径与 SHA256、全部 GT 坐标、目标索引、裁剪范围、模型的显示阈值内和低阈值候选框指标。
- `target_transitions_conf025.csv` 包含**完整重建测试集** 641 张图、701 个 GT 在三个模型下的 IoU 0.50/0.75 一对一匹配状态。
- `reproduction/ablation/build_module_direct_evidence_seed1_200.py` 从已有冻结预测重新生成图及 CSV，不重新推理、训练或改动图像内容。

图中 a、b、c 是同一个 `DetectDataset_clean_v2` 测试集里的三个原始来源样例，并非新的测试集。完整 641 张测试图还含 16 张自采视频帧；以下完整测试集的统计**包含**这 16 张，与以前 200 轮消融的全测试口径一致。图片仅从原始来源选取，以避免在一张图里反复展示数量很少的自采帧。

同一行的父模型与加模块模型使用相同原图、相同原生像素裁剪、相同 GT、相同置信度阈值 0.25。绿色实线是所有可见 GT，红色实线是模型预测；未修改亮度、对比度、清晰度或图中水印。`best.pt` 均由验证集选取，随机种子均为 1，训练上限均为 200 轮。

## 三张图实际能说明什么

| 面板 | 直接对照 | 目标尺度 | 观测 | 允许的表述 |
|---|---|---:|---|---|
| a | YOLO11n → +P2 | 20.5 px | 基线在阈值 0.25 下未检出该 GT；它仍有 IoU 0.779、置信度 0.079 的低分候选框。P2 框 IoU 0.747、置信度 0.387。 | P2 在此例主要提高了该小目标的**检出分数**，并非改善其框定位。 |
| b | +P2 → +P2+ADown | 29.2 px | P2 IoU 0.612 → ADown IoU 0.777；两者均已在 IoU 0.50 检出。 | ADown 在此例使框跨过严格的 0.75 IoU 门槛；**不是**新增检出。 |
| c | +P2 → +P2+ADown | 62.1 px | P2 IoU 0.777、置信度 0.545；ADown 在 0.25 阈值下无匹配框，但有 IoU 0.832、置信度 0.013 的低分候选框。 | ADown 也可能导致分数坍塌；不能宣称对所有目标都有效。 |

这里的“未检出”严格指显示阈值 0.25 且与所选 GT 的 IoU ≥0.50。它**不是**说模型内部没有产生任何框。第一幅源帧含两处 GT，图中完整原图均绘制 GT，但裁剪专门围绕第 0 个小目标；过阈值的其他目标框不构成该目标被检出。

## 全测试集的系统性核对，不把示例当总体

对 701 个 GT，按每张图的预测置信度降序，分别在 IoU 0.50 和 0.75 做一对一贪心匹配。这个固定阈值的目标转换表**不是 COCO AP**：它不积分置信度排序、不能代替完整 PR 曲线，也不校正视频相邻帧间的依赖。它的作用是核查所选图是否与总体转换方向矛盾。

| 范围与门槛 | 直接加模块对照 | 新匹配目标 | 失去匹配目标 | 判断 |
|---|---|---:|---:|---|
| 16–32 px，126 个 GT，IoU 0.50 | YOLO11n → P2 | 9 | 0 | P2 的小目标检出案例与这批目标的净转换方向一致。 |
| 16–32 px，126 个 GT，IoU 0.75 | YOLO11n → P2 | 9 | 10 | P2 **不能**被说成改善了严格定位。 |
| 16–32 px，126 个 GT，IoU 0.50 | P2 → P2+ADown | 2 | 8 | ADown **不能**被说成提高了固定阈值的小目标召回。 |
| 16–32 px，126 个 GT，IoU 0.75 | P2 → P2+ADown | 11 | 9 | 存在严格定位改善，但净转换很小，不能独立证明普遍增益。 |
| 全部 701 个 GT，IoU 0.75 | P2 → P2+ADown | 45 | 39 | 与 AP75 升高方向一致，但该计数不是 AP75。 |

来自**同一完整测试集、同一个 COCO-style 评估器**的连续阈值指标提供更强的总体证据：YOLO11n → P2 的 AP50 为 93.61% → 95.70%，16–32 px AP 为 36.61% → 39.02%，但 AP75 为 60.53% → 59.56%；P2 → P2+ADown 的 AP75 为 59.56% → 64.17%，AP50:95 为 56.82% → 57.86%，同时参数量 2.904M → 2.278M、GFLOPs 10.8 → 9.5。完整原始数据与计算口径见 `reproduction/results/p2_msef_adown_visual_evidence_seed1/figure_source_data.csv` 和 `FIGURE_NOTES.md`。尺寸分组采用等效方形边长（以 GT 框面积开方），并非直接使用 COCO 官方 small/medium 边界。

## SCI 审稿视角的结论边界

1. **P2 有较可辩护的针对性收益**：同 seed 的总体 AP50 和 16–32 px AP 提升；一对一目标转换显示 16–32 px 的 IoU 0.50 新增 9、丢失 0；图 a 说明其中一种具体表现是低分小目标跨过显示阈值。不能从这一张图推导“P2 解决所有小目标”或“P2 定位更准”。
2. **ADown 有条件地有效**：同 seed 的严格定位 AP75 上升，且参数和计算量下降；图 b 说明何种框变化可能贡献 AP75。但 IoU 0.50 下 16–32 px 新增 2、丢失 8，图 c 又是分数坍塌。更关键的是，先前 200 轮三种子中 ADown 相对 P2 的 AP50:95 增益**并不方向稳定**；不能把 seed1 的图写成“ADown 稳定改善召回/定位”。
3. **MSEF 不在本图的有效性结论内**。当前同协议全模型 `P2+MSEF+ADown` 相比 `P2+ADown` 的 AP50:95 和 AP75 均下降，故不能靠挑一张 MSEF 正例证明它值得加入；若论文最终不使用 MSEF，应如实报告筛选和消融。
4. 图 a 的源视频帧保留原有片内品牌标识和画中画，没有擦除水印。提交期刊前必须核查该帧及其他源图的使用许可；若不允许发表，应使用有权使用的独立样本重作图，而不是抹掉标识。相邻视频帧可能造成样例相关性，不能把“9 个获益 GT”解释成 9 个独立视频事件。
5. 所有例图是在查看测试集后选出的，属于**探索性、说明性**证据；测试集已被多次用于模型研究，不能当作盲法独立验证。真正更强的“证明”需要冻结结构后在新来源/未参与方法选择的数据上复测；当前图与消融仅支持所述数据集和协议下的有限主张。

**建议图注（英文草案）：** “Direct-parent qualitative contrasts for validation-selected 200-epoch seed-1 models. (a) A 20-pixel drone is below the 0.25 display threshold with YOLO11n despite a well-localized low-score candidate, but is detected by YOLO11n+P2. (b) For a different 29-pixel target already detected by P2, adding ADown yields a tighter box, crossing IoU 0.75. (c) A counterexample in which adding ADown suppresses a valid P2 detection below the display threshold. Teal boxes denote ground truth, red boxes denote displayed predictions, and amber boxes mark identical native-pixel crops. Examples were selected post hoc and are illustrative rather than representative; full-test metrics and target-transition counts are reported separately.”

## 图件 QA

绘图语言为 Python/Matplotlib，PDF 和 SVG 保留可编辑文本；最终幅宽 182.9 mm。源代码静态检查 21 项通过、0 警告、0 失败；PDF 实际最小字高 6.2 pt；最终图多面板对齐检查通过；PDF 文字碰撞检查 0 失败、0 警告。原图与逐行裁剪已经人工检查，图中不含图像增强。期刊投稿前仍须进行图像来源权利核查。
