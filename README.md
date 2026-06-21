# 中国象棋棋盘 FEN 识别

拍照识别中国象棋棋盘，生成国际通用的 FEN 棋谱字符串。

## 技术方案

**传统 CV + 深度学习**，分四个阶段：

1. **棋盘定位与校正** — Canny 边缘检测 → 最大四边形轮廓查找 → 透视变换 → 450×500 标准棋盘
2. **交点定位** — 检测 9×10 个棋盘交点，以交点为中心截取棋子区域
3. **棋子识别** — MobileNetV3-Small（15 分类：7 红子 + 7 黑子 + 空）
4. **方向与规则分析** — 自动判断红方方向，统一为红方在下，并执行保守纠错和静态合法性校验
5. **FEN 组装** — 按规则拼接成 FEN 字符串

## 项目结构

```
cchlink/
├── data/
│   ├── raw/                          # 原始棋盘照片
│   ├── pieces/                       # 棋子训练数据
│   │   ├── red/                      #   红方：帅仕相俥马炮兵
│   │   └── black/                    #   黑方：将士象车马炮卒
│   ├── generated/                    # 数据增强样本
│   └── models/                       # 训练好的模型权重
│
├── src/
│   ├── preprocess/
│   │   ├── board_detector.py         #   Canny + 最大四边形查找
│   │   └── perspective.py            #   透视变换
│   ├── segmentation/
│   │   └── grid_splitter.py          #   9×10 格子切分
│   ├── recognition/
│   │   ├── model.py                  #   MobileNetV3-Small 分类器
│   │   ├── dataset.py                #   数据加载 + 数据增强
│   │   └── predictor.py              #   推理接口
│   ├── fen/
│   │   ├── fen_builder.py            #   FEN 字符串组装
│   │   └── rules.py                  #   方向判断、保守纠错、静态合法性校验
│   ├── analysis.py                   #   结构化分析结果
│   ├── artifacts.py                  #   可视化与调试产物
│   └── pipeline.py                   #   总管线
│
├── scripts/
│   ├── train.py                      # 训练脚本
│   ├── predict.py                    # CLI 单张推理
│   ├── batch.py                      # CLI 递归批量推理
│   ├── test_fen.py                   # FEN 组装单元测试
│   └── test_imports.py               # 模块导入验证
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 棋子类别映射

| 序号 | 类别 | FEN | 序号 | 类别 | FEN |
|------|------|-----|------|------|-----|
| 0 | 红帅 | K | 7 | 黑将 | k |
| 1 | 红仕 | A | 8 | 黑士 | a |
| 2 | 红相 | B | 9 | 黑象 | b |
| 3 | 红俥 | R | 10 | 黑车 | r |
| 4 | 红马 | N | 11 | 黑马 | n |
| 5 | 红炮 | C | 12 | 黑炮 | c |
| 6 | 红兵 | P | 13 | 黑卒 | p |
| — | — | — | 14 | 空 | 数字 |

## 环境要求

- Python 3.10+
- 依赖安装：`pip install -e .`
- 开发依赖：`pip install -e ".[dev]"`

核心依赖：OpenCV、PyTorch、torchvision、NumPy、Pillow

## 使用方法

### 训练模型

```bash
# 将训练数据放入 data/pieces/ 后执行
python scripts/train.py -d data/pieces -o data/models/checkpoint.pth -e 30
```

推荐使用按整盘分组的数据集，避免同一棋盘泄漏到训练集和验证集：

```bash
python scripts/generate_grouped_data.py -o data/pieces_grouped_v4 -n 3000
python scripts/audit_dataset.py data/pieces_grouped_v4 --against-manifest evaluation/standard_manifest.csv
python scripts/train_v2.py -d data/pieces_grouped_v4 -o data/models/checkpoint_standard_v4.pth
```

当前标准数据版本为 `standard-v4`。它沿用已确认的真实棋盘模板：正确初始布局、
红黑九宫 X、楚河汉界，以及距交点 6px、短臂 5px 的紧凑炮兵卒定位标记。v4
还会将部分完整棋盘先放入透视、光照和噪声场景，再按已知四角矫正回标准棋盘后
切出训练图块，使训练输入更接近端到端推理。v4 在此基础上修正了 `red_top`
样本生成：先渲染正常棋盘再整图旋转 180°，让倒拍时棋子文字方向也与评测/真实照片一致。

参数说明：
- `-d` / `--data`：训练数据根目录（必填）
- `-o` / `--output`：模型保存路径（必填）
- `-e` / `--epochs`：训练轮数（默认 30）
- `-b` / `--batch_size`：批次大小（默认 32）
- `--lr`：学习率（默认 0.001）
- `--device`：训练设备 cpu / cuda（默认 cpu）

### 识别棋盘

```bash
python scripts/predict.py photo.jpg -m data/models/checkpoint.pth -v
python scripts/predict.py photo.jpg -m data/models/checkpoint.pth --visualize-dir output/visual
python scripts/predict.py photo.jpg -m data/models/checkpoint.pth --debug-dir output/debug
```

参数说明：
- `image`：棋盘图片路径（位置参数）
- `-m` / `--model`：模型权重路径（必填）
- `-d` / `--device`：推理设备（默认 cpu）
- `-v` / `--verbose`：打印详细识别结果
- `--visualize-dir`：保存原图棋盘四角和规范方向棋盘识别叠加图
- `--debug-dir`：显式保存候选棋盘、校正棋盘、网格、90 格接触表和完整分析 JSON

默认识别流程不会写入磁盘。最终 FEN 始终规范化为黑方在上、红方在下；异常静态棋局仍会返回结果，并在结构化警告中说明问题。

### Python 结构化分析接口

```python
from src.pipeline import Pipeline

pipeline = Pipeline("data/models/checkpoint.pth")
result = pipeline.analyze(image_rgb, debug_dir="output/debug/example")

print(result.fen)          # 规则修正后的规范方向 FEN
print(result.raw_fen)      # 方向规范化后、规则修正前的 FEN
print(result.orientation)  # red_bottom / red_top / unknown
print(result.warnings)
print(result.corrections)
```

`AnalysisResult` 还包含棋盘与网格置信度、90 个交点的类别/名称/置信度、棋盘角点和网格交点坐标。`Pipeline.run()` 继续只返回最终 FEN，`Pipeline.run_verbose()` 继续返回字典并包含新增分析字段。

### 批量识别

```bash
python scripts/batch.py photos --model data/models/checkpoint.pth --output results.csv
python scripts/batch.py photos --model data/models/checkpoint.pth --output results.csv \
  --visualize-dir output/visual --debug-dir output/debug
```

批处理会递归扫描常见图片格式，单张读取或识别失败不会终止任务。CSV 包含相对路径、状态、最终/原始 FEN、方向、棋盘/网格置信度、警告代码、错误信息和处理耗时。

### 验证导入

```bash
python scripts/test_imports.py
```

### 测试 FEN

```bash
python scripts/test_fen.py
```

### 固定照片分阶段评估

```bash
python scripts/evaluate_stages.py --model data/models/checkpoint_standard_v4.pth \
  --manifest evaluation/standard_manifest.csv \
  --device cuda \
  --json-output evaluation/standard_v4_report.json
```

评估会分别报告棋盘检测通过率、网格通过率、90 格准确率、整盘完全匹配率和延迟。
`evaluation/real_manifest.csv` 是固定照片回归集，严禁将其中照片或其派生样本用于训练。

如需定位剩余错误，可运行逐格诊断：

```bash
python scripts/diagnose_eval_errors.py \
  --model data/models/checkpoint_standard_v4.pth \
  --manifest evaluation/standard_manifest.csv \
  --device cuda \
  --json-output evaluation/standard_v4_diagnosis.json
```

模型晋级建议同时满足：标准评测格子级准确率不低于 98.5%，整盘完全正确率不低于
80%，空棋盘/木质初始/塑料倾斜三张人工确认预览全部完全正确，且旧版回归集下降
不超过 2 个百分点。

`checkpoint_standard_v4.pth` 已通过该标准评测门槛：标准评测 5398/5400 格正确
（99.96%），58/60 盘完全正确（96.67%）；三张人工确认预览全部完全正确。旧版
兼容清单格子级准确率为 324/360（90.00%），相对旧基线下降约 1.67 个百分点。

## 开发状态

| 模块 | 状态 |
|------|------|
| 项目骨架 + 依赖 | ✅ |
| 棋盘检测与透视校正 | ✅ |
| 9×10 格子分割 | ✅ |
| CNN 模型定义 + 数据集 | ✅ |
| FEN 字符串组装 | ✅ |
| Pipeline 管线 | ✅ |
| 训练脚本 | ✅ |
| CLI 推理入口 | ✅ |
| pytest + GitHub Actions CI | ✅ |
| 按棋盘分组训练数据 | ✅ |
| 固定照片分阶段评估 | ✅ |
| 方向规范化 + 静态合法性校验 | ✅ |
| 批量推理 + 可视化 + 显式调试 | ✅ |

## 许可证

MIT
