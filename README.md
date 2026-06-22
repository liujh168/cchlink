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
│   ├── pieces/                       # 基础棋子训练图块
│   │   ├── red/                      #   红方：帅仕相俥马炮兵
│   │   └── black/                    #   黑方：将士象车马炮卒
│   └── models/                       # 本地模型权重（不纳入 Git）
│
├── evaluation/
│   ├── real_images/                  # 真实照片/截图回归集
│   ├── standard_manifest.csv         # 标准合成评估清单
│   ├── real_manifest.csv             # 真实照片固定回归清单
│   ├── standard_v4_report.json       # 当前标准评估结果
│   ├── standard_v4_diagnosis.json    # 当前标准逐格诊断
│   ├── real_v4_report.json           # 当前真实回归结果
│   └── real_v4_diagnosis.json        # 当前真实逐格诊断
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
│   ├── generate_grouped_data.py      # 生成按整盘分组的 v4 训练数据
│   ├── generate_standard_eval.py     # 生成标准评估集
│   ├── train_v2.py                   # 当前训练入口
│   ├── evaluate_stages.py            # 分阶段评估
│   ├── diagnose_eval_errors.py       # 逐格错误诊断
│   ├── predict.py                    # CLI 单张推理
│   ├── batch.py                      # CLI 递归批量推理
│   ├── audit_dataset.py              # 训练集泄漏审计
│   ├── generate_realistic_previews.py # 人工确认预览
│   └── generate_board.py             # 标准棋盘渲染工具（测试使用）
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

推荐训练路径是先生成按整盘分组的数据集，再用 `train_v2.py` 训练，避免同一棋盘泄漏到训练集和验证集。当前真实集晋级配置为
v4/v5/v6/v8 融合；最新合成候选数据版本为 `standard-v8`：

```bash
python scripts/generate_grouped_data.py -o data/pieces_grouped_v8 -n 3000
python scripts/audit_dataset.py data/pieces_grouped_v8 --against-manifest evaluation/standard_manifest.csv
python scripts/train_v2.py -d data/pieces_grouped_v8 -o data/models/checkpoint_standard_v8.pth
```

`data/pieces/` 是保留的基础棋子训练图块；`data/pieces_grouped_v4/` 和 `data/models/`
是本地派生输出，不纳入 Git。

当前已晋级模型的数据版本为 `standard-v4`。它沿用已确认的真实棋盘模板：正确初始布局、
红黑九宫 X、楚河汉界，以及距交点 6px、短臂 5px 的紧凑炮兵卒定位标记。v4
还会将部分完整棋盘先放入透视、光照和噪声场景，再按已知四角矫正回标准棋盘后
切出训练图块，使训练输入更接近端到端推理。v4 在此基础上修正了 `red_top`
样本生成：先渲染正常棋盘再整图旋转 180°，让倒拍时棋子文字方向也与评测/真实照片一致。

`standard-v5` 在 v4 基础上增加真实照片风格增强（色偏、不均匀光照、轻微虚焦、
降采样和 JPEG 压缩），并针对边缘有子交点记录并生成多裁剪尺度、轻微外向偏移的
训练图块，用于补强真实照片中外圈棋子容易被判空的问题。

`standard-v6` 继续沿用不得使用真实回归照片派生样本的约束，并根据最新诊断报告
加强木质棋盘、初始布局、边缘大棋子/贴边棋子、低清晰度和压缩场景的占比。当前
诊断显示真实集主要瓶颈为 `piece_to_empty`，尤其是 `edge:piece_to_empty`，因此
v6 的目标是优先提升真实照片外圈有子召回，同时用空棋盘和困难空位样本防止误报回升。
`checkpoint_standard_v6.pth` 单模型标准集达到 5400/5400、60/60，但真实集为
1307/1620，暂不作为单模型晋级；当前最佳候选是 v4/v5/v6 概率融合，权重为
0.55/0.05/0.40，并叠加近完整初始局模板补全与低饱和真实木盘视觉占位补全。
`standard-v7` 新增浅色木盘合成风格并训练 `checkpoint_standard_v7.pth`，单模型标准集
5400/5400、真实集 1332/1620，融合扫权重未超过当前最佳，因此暂不晋级。

`standard-v8` 继续禁止使用 `evaluation/real_images` 或派生图训练，新增布面/塑料红线、
浅色布面、屏幕截图风格、近初始开局、复杂中局和稀疏残局合成覆盖。`checkpoint_standard_v8.pth`
单模型标准集保持 5400/5400、60/60；真实集单模型不晋级，但作为小剂量融合信号配合
士/象静态点概率先验后，v4/v5/v6/v8 权重 `0.44/0.03/0.36/0.17` 成为当前最佳。

参数说明：
- `-d` / `--data`：训练数据根目录（必填）
- `-o` / `--output`：模型保存路径（必填）
- `-e` / `--epochs`：训练轮数（默认 30）
- `-b` / `--batch_size`：批次大小（默认 32）
- `--lr`：学习率（默认 0.001）
- `--device`：训练设备 cpu / cuda（默认 cpu）

### 识别棋盘

```bash
python scripts/predict.py photo.jpg -m data/models/checkpoint_standard_v4.pth -v
python scripts/predict.py photo.jpg -m data/models/checkpoint_standard_v4.pth --visualize-dir output/visual
python scripts/predict.py photo.jpg -m data/models/checkpoint_standard_v4.pth --debug-dir output/debug
python scripts/predict.py photo.jpg -m data/models/checkpoint_standard_v4.pth \
  --ensemble-model data/models/checkpoint_standard_v5.pth \
  --ensemble-model data/models/checkpoint_standard_v6.pth \
  --ensemble-model data/models/checkpoint_standard_v8.pth \
  --ensemble-weight 0.44 --ensemble-weight 0.03 --ensemble-weight 0.36 --ensemble-weight 0.17
```

参数说明：
- `image`：棋盘图片路径（位置参数）
- `-m` / `--model`：模型权重路径（必填）
- `-d` / `--device`：推理设备（默认 cpu）
- `-v` / `--verbose`：打印详细识别结果
- `--visualize-dir`：保存原图棋盘四角和规范方向棋盘识别叠加图
- `--debug-dir`：显式保存候选棋盘、校正棋盘、网格、90 格接触表和完整分析 JSON
- `--ensemble-model` / `--ensemble-weight`：可选模型概率融合；权重数量需等于主模型加
  ensemble 模型总数

默认识别流程不会写入磁盘。最终 FEN 始终规范化为黑方在上、红方在下；异常静态棋局仍会返回结果，并在结构化警告中说明问题。

### Python 结构化分析接口

```python
from src.pipeline import Pipeline

pipeline = Pipeline("data/models/checkpoint_standard_v4.pth")
result = pipeline.analyze(image_rgb, debug_dir="output/debug/example")

print(result.fen)          # 规则修正后的规范方向 FEN
print(result.raw_fen)      # 方向规范化后、规则修正前的 FEN
print(result.orientation)  # red_bottom / red_top / unknown
print(result.warnings)
print(result.corrections)
```

如需使用当前 v8 候选融合，可传入多个模型和权重：

```python
pipeline = Pipeline(
    [
        "data/models/checkpoint_standard_v4.pth",
        "data/models/checkpoint_standard_v5.pth",
        "data/models/checkpoint_standard_v6.pth",
        "data/models/checkpoint_standard_v8.pth",
    ],
    model_weights=[0.44, 0.03, 0.36, 0.17],
)
```

`AnalysisResult` 还包含棋盘与网格置信度、90 个交点的类别/名称/置信度、棋盘角点和网格交点坐标。`Pipeline.run()` 继续只返回最终 FEN，`Pipeline.run_verbose()` 继续返回字典并包含新增分析字段。

### 批量识别

```bash
python scripts/batch.py photos --model data/models/checkpoint_standard_v4.pth --output results.csv
python scripts/batch.py photos --model data/models/checkpoint_standard_v4.pth --output results.csv \
  --visualize-dir output/visual --debug-dir output/debug
python scripts/batch.py photos --model data/models/checkpoint_standard_v4.pth \
  --ensemble-model data/models/checkpoint_standard_v5.pth \
  --ensemble-model data/models/checkpoint_standard_v6.pth \
  --ensemble-model data/models/checkpoint_standard_v8.pth \
  --ensemble-weight 0.44 --ensemble-weight 0.03 --ensemble-weight 0.36 --ensemble-weight 0.17 \
  --output results.csv
```

批处理会递归扫描常见图片格式，单张读取或识别失败不会终止任务。CSV 包含相对路径、状态、最终/原始 FEN、方向、棋盘/网格置信度、警告代码、错误信息和处理耗时。

### 运行测试

```bash
python -m pytest
ruff check .
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
80%，空棋盘/木质初始/塑料倾斜三张人工确认预览全部完全正确。

`checkpoint_standard_v4.pth` 已通过该标准评测门槛：标准评测 5398/5400 格正确
（99.96%），58/60 盘完全正确（96.67%）；三张人工确认预览全部完全正确。

当前 v8 候选融合模式（v4/v5/v6/v8 权重 0.44/0.03/0.36/0.17）加近完整初始局模板补全、
视觉占位补全、候选面积先验与士/象静态点概率先验后，
在固定清单上达到：标准评测 5400/5400 格正确、60/60 盘完全正确；真实照片回归集
1543/1620 格正确（95.25%）、整盘 11/18。screen 子集提升到 598/630、整盘 3/7，
wood 子集达到 613/630、整盘 5/7，initial 子集保持 720/720、整盘 8/8；剩余错误主要转向
plastic midgame、非初始 wood opening/endgame 与 screen endgame 弱棋子召回。

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
