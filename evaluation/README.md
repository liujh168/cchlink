# 评估数据集

`real_manifest.csv` 是固定照片回归集。新增照片时，应以不可变记录的形式填写图片路径、
预期 FEN 和来源标识。该清单中的照片不得参与训练，否则端到端评估结果会失真。

使用以下命令执行分阶段评估：

```bash
python scripts/evaluate_stages.py --model data/models/checkpoint_v8.pth
```
