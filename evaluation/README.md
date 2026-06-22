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

本目录只保留当前 `standard-v4` 标准评估和 `real-v4` 真实照片回归结果。历史实验报告
已删除；后续固定评估以 `standard_manifest.csv` 和 `real_manifest.csv` 为准。

`real_manifest.csv` 真实照片回归集当前 v4 单模型结果为：检测 18/18、网格接受
18/18、格子级准确率 1409/1620（86.98%）、整盘完全正确 5/18。
`checkpoint_standard_v6.pth` 单模型标准集为 5400/5400、真实集为 1307/1620，
暂不单模型晋级。当前最佳候选为 v4/v5/v6=0.55/0.05/0.40 融合：标准集
5400/5400、60/60，真实集 1505/1620（92.90%）、整盘 9/18。低饱和真实木盘
视觉占位补全已拉回剩余 wood/initial 红上漏判，最佳融合诊断中 `piece_to_empty`
为 85 个、`edge:piece_to_empty` 为 55 个。后续应继续补充不污染评估集的真实训练照片，
并加强 unknown/plastic/screen 与非初始 wood 场景。
