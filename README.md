# 中国象棋棋盘 FEN 识别

拍照识别中国象棋棋盘，生成国际通用的 FEN 棋谱字符串。

## 技术方案

**传统 CV + 深度学习**，分四个阶段：

1. **棋盘定位与校正** — Canny 边缘检测 → 最大四边形轮廓查找 → 透视变换 → 450×500 标准棋盘
2. **交点定位** — 检测 9×10 个棋盘交点，以交点为中心截取棋子区域
3. **棋子识别** — MobileNetV3-Small（15 分类：7 红子 + 7 黑子 + 空）
4. **FEN 组装** — 按规则拼接成 FEN 字符串

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
│   │   └── fen_builder.py            #   FEN 字符串组装
│   └── pipeline.py                   #   总管线：串联以上 4 个模块
│
├── scripts/
│   ├── train.py                      # 训练脚本
│   ├── predict.py                    # CLI 单张推理
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
python scripts/generate_grouped_data.py -o data/pieces_grouped -n 500
python scripts/audit_dataset.py data/pieces_grouped
python scripts/train_v2.py -d data/pieces_grouped -o data/models/grouped.pth
```

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
```

参数说明：
- `image`：棋盘图片路径（位置参数）
- `-m` / `--model`：模型权重路径（必填）
- `-d` / `--device`：推理设备（默认 cpu）
- `-v` / `--verbose`：打印详细识别结果

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
python scripts/evaluate_stages.py --model data/models/checkpoint_v8.pth
```

评估会分别报告棋盘检测通过率、网格通过率、90 格准确率、整盘完全匹配率和延迟。

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
| 批量推理 + 棋规后处理 | ✅ |

## 许可证

MIT
