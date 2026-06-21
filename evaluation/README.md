# 评估数据集

`real_manifest.csv` 是固定照片回归集。新增照片时，应以不可变记录的形式填写图片路径、
预期 FEN 和来源标识。该清单中的照片不得参与训练，否则端到端评估结果会失真。

使用以下命令执行分阶段评估：

```bash
python scripts/evaluate_stages.py \
  --model data/models/checkpoint_standard_v4.pth \
  --manifest evaluation/standard_manifest.csv \
  --device cuda \
  --json-output evaluation/standard_v4_report.json
```

`standard_manifest.csv` 是合成标准棋盘评估集，覆盖经典、木质、浅色塑料三种样式，
以及初始局面、空棋盘和随机中盘。该清单用于模型晋级评估，不参与训练。

若整盘完全正确率不足，先运行逐格诊断：

```bash
python scripts/diagnose_eval_errors.py \
  --model data/models/checkpoint_standard_v4.pth \
  --manifest evaluation/standard_manifest.csv \
  --device cuda \
  --json-output evaluation/standard_v4_diagnosis.json
```

当前晋级门槛：

- 棋盘检测成功率 >= 99%
- 网格接受率 >= 99%
- 格子级准确率 >= 98.5%
- 整盘完全正确率 >= 80%
- 三张人工确认预览图全部完全正确

## 当前 v4 结果

`data/models/checkpoint_standard_v4.pth` 在 `standard_manifest.csv` 上的结果：

- 检测成功率：60/60
- 网格接受率：60/60
- 格子级准确率：5398/5400（99.96%）
- 整盘完全正确率：58/60（96.67%）
- 分风格：classic 20/20，wood 20/20，plastic 18/20

早期 `data/raw` 生成图不符合当前标准棋盘要求，相关 legacy 兼容清单和报告已退役；
后续固定评估以 `standard_manifest.csv` 和 `real_manifest.csv` 为准。

`real_manifest.csv` 真实照片回归集当前结果为：检测 18/18、网格接受 18/18、
格子级准确率 1296/1620（80.00%）、整盘完全正确 1/18。主要剩余错误仍集中在
边缘交点的有子判空，后续应优先补充真实风格和大棋子尺度的数据增强。
