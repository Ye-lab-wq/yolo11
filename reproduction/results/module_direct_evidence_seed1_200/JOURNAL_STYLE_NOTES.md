# 同领域图版借鉴与新版定性图

## 推荐查看

`journal_qualitative_p2_adown.png` 是当前**优先用于小论文内部排版讨论**的版本；同图另有 PDF、可编辑文字的 SVG 和 600 dpi TIFF。旧的 `direct_parent_examples.*` 保留作为完整核查图，包含 ADown 的失败边界；它不再是推荐的正文版式。两版使用同一批冻结预测，**没有重训、重新挑权重或改变测试数据**。

新版的图面只放两种互补的直接父模型对照：

- **a，P2：**YOLO11n 对一个约 20 px 的 GT 在置信度 0.25 下没有匹配检测；+P2 的对应框 IoU 0.747、置信度 0.387。基线并非完全没有候选框：其低分候选 IoU 0.779、置信度 0.079。该例说明的是**分数跨阈值**，不是框定位提高。
- **b，ADown：**对另一约 29 px 的 GT，P2 已检出，IoU 0.612；+P2+ADown 的框 IoU 0.777，两者置信度均约 0.57。该例说明的是**框定位**，不是新增检出。

两个例子均出自 `DetectDataset_clean_v2` 的同一测试划分。所有图框绘制在未增强的原始 RGB 帧上；同一行四格使用同一来源帧、同一 GT、同一原生像素局部区域，完整画面 640 px 与局部 128 px 显示宽度相等，因此标注为 5×。完整画面绘出所有 GT 和所有置信度 ≥0.25 的预测；局部画面绘出局部 GT 与中心落在局部的过阈值预测。黄色矩形是局部来源，青绿色实线为 GT，红色实线为预测。主图故意不添加未经验证的 Grad-CAM：热力图不能仅因“更集中”就证明某个模块的机制。

## 借鉴了哪些正式出版的图版

以下是**版式参考，不是性能对标，也未复制任何论文图片或绘图元素**：

1. [Wang 等，*ISPRS Journal of Photogrammetry and Remote Sensing* 2026，ZoomDet，图 6](https://doi.org/10.1016/j.isprsjprs.2026.01.036)（[作者公开稿](https://arxiv.org/pdf/2602.07512)）：按同一来源帧排列基线与不同方法，并把原图和预测分区排列；少数行标明放大区域。这启发我们保持来源帧、目标位置和局部区域严格一致，同时只保留本研究必需的模型列。
2. [Li 等，CVPR 2025，PointSR，图 5](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_PointSR_Self-Regularized_Point_Supervision_for_Drone-View_Object_Detection_CVPR_2025_paper.pdf)：同一帧跨方法保持相同位置，列标题直接标模型，并明确区分 GT 与检测结果。我们不照搬其七列密集排版。
3. [Chen 等，*Sensors* 2026，ADA-YOLO，图 9](https://doi.org/10.3390/s26123908)：同一场景的基线/改进模型并排，在各自图内展示相同目标区的放大细节。我们借鉴“固定场景 + 固定放大区域”的比较逻辑，但将放大图放在单独相邻列，避免遮盖原图和标注。
4. [FDA_YOLOv8，*Scientific Reports* 2026，图 9](https://www.nature.com/articles/s41598-026-51902-3)：按场景为行、方法为列，让读者横向查同一目标的结果。我们保留其行/列可比性，但为本项目的单类微小无人机增加显式的原生像素局部放大。

参考文献的科学主张、评价数据集和期刊地位与本项目均不等价；这些文献只能指导如何**清楚、诚实地呈现已有预测**。当前尚未确定目标投稿期刊，故 182.9 mm 双栏幅宽和 6.2 pt 最小文字只是通用工作版式，正式投稿需按目标期刊作者指南重新核对。

## 旧版为什么不适合正文，新版仍有什么限制

旧版每行“完整图 + 三个孤立方形裁剪”，横向留白较大；模型名和较长状态文字散在图中，读者必须来回看 GT 与预测，而且 P2 的直接父模型作用被三模型并列稀释。新版按**直接父模型**成对排布，完整场景与等比例局部细节重复在两模型上，预测框细线、颜色和阈值一致；结论数字放在图外，正式图注可以独立解释。

新版本仍是**看过测试结果后选择的说明性图片**，不是独立验证；不能用它单独证明模块普遍有效。全测试集 641 图/701 GT 的 AP 和一对一目标转换仍以 `FIGURE_NOTES.md`、`figure_source_data.csv`、`target_transitions_conf025.csv` 为准。尤其 ADown 在 16–32 px 的固定阈值 IoU 0.50 匹配是新增 2、失去 8，旧核查图中的置信度坍塌反例不应在论文中隐去。MSEF 亦不在这张图的主张内。

图 a 的源帧保留原视频品牌标识和画中画，**投稿前须确认该图的发表使用权**；不能通过擦除标识来规避来源问题。所有源帧相邻视频关系及选例的事后性也应在 Methods/图注中说明。

## 可直接改写的英文图注草案

“**Fig. X | Direct-parent qualitative comparisons for P2 and ADown.** a A 20-pixel-equivalent-side drone is not matched by YOLO11n at the fixed display confidence of 0.25, but is detected after adding P2 (IoU 0.747, confidence 0.387). A geometrically valid baseline candidate remains below the display threshold (confidence 0.079). b For a different 29-pixel target already detected by P2, adding ADown improves the displayed box IoU from 0.612 to 0.777. Each method pair uses the same native-pixel frame and 128-pixel crop, shown at equal physical width (5× display magnification). Teal boxes mark all visible ground-truth objects, red boxes denote detections at confidence ≥0.25, and amber boxes mark the cropped region. Images were selected post hoc from the same rebuilt dataset test split and illustrate the observed behaviors rather than estimating their frequency. The complete test-set results and a counterexample are reported separately.”

## 技术 QA

- 绘图源：`reproduction/ablation/build_journal_qualitative_p2_adown.py`；只读取 `raw_predictions.json` 和原始图/标签，不重新运行网络。
- 图幅 182.9 × 130.8 mm；Python/Matplotlib 是唯一绘图后端。PDF、SVG、300 dpi PNG、600 dpi TIFF 已输出。
- 静态检查 21 项通过、0 警告、0 失败；PDF 最小文字 6.2 pt；八个可比图格的对齐检查通过（8 项比较，0 豁免）；PDF 文字/框线碰撞检查 0 失败、0 警告。
- 已按最终图大小逐格目检：同一行来源一致、缩放一致、两个模型的预测框不互换；行间标题与状态不重叠，图例在图外；原视频自带文字及画中画未被擦除。
