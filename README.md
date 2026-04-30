# EuroSAT RGB MLP 分类实验

本项目基于 NumPy 手工实现多层感知机（MLP），用于 EuroSAT RGB 遥感图像土地覆盖分类。代码不依赖 PyTorch、TensorFlow、JAX 等自动微分框架，模型前向传播、反向传播、交叉熵损失、L2 正则化、SGD/动量 SGD/Adam、学习率衰减、最优权重保存、超参数搜索和测试混淆矩阵均由项目代码实现。

## 代码结构

 5 个核心模块，和 1 个可视化分析模块：

```text
.
├── data_loader.py   # 数据加载与预处理：读取图像、标签编码、分层划分、展平、标准化统计、数据增强
├── model.py         # 模型定义：Linear、激活函数、Dropout、CrossEntropyLoss、SGD/Adam、MLP
├── train.py         # 训练循环：训练/验证、反向传播、学习率衰减、早停、最优权重保存、曲线保存
├── test.py          # 测试评估：加载最优权重、输出 accuracy、打印/保存混淆矩阵及热力图
├── search.py        # 超参数查找：初始随机搜索与精细化随机搜索，记录 CSV
└── analysis.py      # 辅助分析：第一层权重可视化与测试集错例分析
```


| 模块 | 文件 |
|---|---|
| 数据加载与预处理 | `data_loader.py` |
| 模型定义 | `model.py` |
| 训练循环 | `train.py` |
| 测试评估 | `test.py` |
| 超参数查找 | `search.py` |

## 环境依赖

建议使用 Python 3.8+。主要依赖：

```bash
pip install numpy pillow scikit-learn matplotlib
```


## 数据准备

将 EuroSAT RGB 数据集放在 `EuroSAT_RGB/` 下，并保持类别子目录结构：

```text
EuroSAT_RGB/
  AnnualCrop/
  Forest/
  HerbaceousVegetation/
  Highway/
  Industrial/
  Pasture/
  PermanentCrop/
  Residential/
  River/
  SeaLake/
```

`data_loader.py` 会将图像读取为 RGB，resize 到 `64 x 64`，像素归一化到 `[0, 1]`，并按类别分层划分训练集、验证集和测试集。

## 训练

基础训练命令：

```bash
python train.py --epochs 20 --batch-size 128 --lr 1e-3 --optimizer sgd --hidden-dims 256,128 --dropout 0.1,0.1 --model-path saved_model.pkl
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--hidden-dims` | 隐藏层维度，如 `1024,512,256` |
| `--activation` | 激活函数，可选 `relu`、`sigmoid`、`tanh` |
| `--optimizer` | 优化器，可选 `sgd`、`momentum`、`adam` |
| `--lr` | 初始学习率 |
| `--weight-decay` | L2 正则化系数 |
| `--dropout` | 各隐藏层 Dropout 比例 |
| `--augment` | 启用随机水平翻转和 0/90/180/270 度旋转 |
| `--plot-curves` | 保存训练/验证 loss 曲线、验证 accuracy 曲线和 history JSON |
| `--plot-path` | 曲线和 history 的保存前缀 |
| `--model-path` | 保存验证集表现最优的模型权重 |

最终增强版实验命令：

```bash
python train.py --epochs 60 --batch-size 128 --lr 0.002 --optimizer momentum --momentum 0.9 --hidden-dims 1024,512,256 --dropout 0.15,0.15,0.15 --weight-decay 1e-4 --patience 10 --lr-patience 4 --lr-decay 0.5 --min-lr 1e-5 --clip-norm 5.0 --augment --plot-curves --plot-path result/final_aug/final_trial1 --model-path result/final_aug/model/final_trial1_aug_60ep.pkl
```

不启用数据增强：

```bash
python train.py --epochs 60 --batch-size 128 --lr 0.002 --optimizer momentum --momentum 0.9 --hidden-dims 1024,512,256 --dropout 0.15,0.15,0.15 --weight-decay 1e-4 --patience 10 --lr-patience 4 --lr-decay 0.5 --min-lr 1e-5 --clip-norm 5.0 --no-augment --plot-curves --plot-path result/final_noaug/final_trial1 --model-path result/final_noaug/model/final_trial1_noaug_60ep.pkl
```

## 超参数搜索

`search.py` 同时支持初始搜索和精细化搜索。

初始搜索：

```bash
python search.py --mode initial --n-iter 10 --csv-path search_results.csv --overwrite
```

精细化搜索：

```bash
python search.py --mode finetune --n-iter 10 --csv-path finetune_results.csv --overwrite
```

搜索会记录每组超参数对应的验证集 loss 和验证集 accuracy。搜索空间包括隐藏层大小、学习率、正则化强度、Dropout、batch size 和是否启用数据增强。

## 测试与混淆矩阵

加载训练好的最优权重并在独立测试集上评估：

```bash
python test.py --model-path result/final_aug/model/final_trial1_aug_60ep.pkl --hidden-dims 1024,512,256 --dropout 0.15,0.15,0.15 --activation relu
```

保存混淆矩阵 CSV 和热力图：

```bash
python test.py --model-path result/final_aug/model/final_trial1_aug_60ep.pkl --hidden-dims 1024,512,256 --dropout 0.15,0.15,0.15 --activation relu --save-confusion-csv result/final_aug/confusion_matrix.csv --save-confusion-plot result/final_aug/confusion_matrix.png --save-normalized-confusion-plot result/final_aug/confusion_matrix_normalized.png
```

## 可视化分析

生成第一层权重可视化和错例分析：

```bash
python analysis.py --model-path result/final_aug/model/final_trial1_aug_60ep.pkl --num-neurons 16 --num-samples 5
```

该脚本会生成：

| 文件 | 说明 |
|---|---|
| `weight_visualization.png` | 第一层隐藏层权重恢复成图像尺寸后的可视化 |
| `misclassified_analysis.png` | 测试集中若干分类错误样本及其预测标签 |

## 已有结果

当前目录中已有最终结果文件：

| 实验 | 测试 loss | 测试 accuracy | 目录 |
|---|---:|---:|---|
| 启用数据增强 | `0.9365` | `0.7251` | `result/final_aug/` |
| 不启用数据增强 | `1.1528` | `0.6839` | `result/final_noaug/` |

## 最终结果对应命令

最终模型对应的训练命令：

```bash
python train.py --epochs 60 --batch-size 128 --lr 0.002 --optimizer momentum --momentum 0.9 --hidden-dims 1024,512,256 --dropout 0.15,0.15,0.15 --weight-decay 1e-4 --patience 10 --lr-patience 4 --lr-decay 0.5 --min-lr 1e-5 --clip-norm 5.0 --augment --plot-curves --plot-path result/final_aug/final_trial1 --model-path result/final_aug/model/final_trial1_aug_60ep.pkl
```

最终模型对应的测试与混淆矩阵可视化命令：

```bash
python test.py --model-path result/final_aug/model/final_trial1_aug_60ep.pkl --hidden-dims 1024,512,256 --dropout 0.15,0.15,0.15 --activation relu --save-confusion-csv result/final_aug/confusion_matrix.csv --save-confusion-plot result/final_aug/confusion_matrix.png --save-normalized-confusion-plot result/final_aug/confusion_matrix_normalized.png
```

noaug最终模型对应的训练命令：

```bash
python train.py --epochs 60 --batch-size 128 --lr 0.002 --optimizer momentum --momentum 0.9 --hidden-dims 1024,512,256 --dropout 0.15,0.15,0.15 --weight-decay 1e-4 --patience 10 --lr-patience 4 --lr-decay 0.5 --min-lr 1e-5 --clip-norm 5.0 --no-augment --plot-curves --plot-path result/final_noaug/final_trial1 --model-path result/final_noaug/model/final_trial1_noaug_60ep.pkl
```

noaug最终模型对应的测试与混淆矩阵可视化命令：

```bash
python test.py --model-path result/final_noaug/model/final_trial1_noaug_60ep.pkl --hidden-dims 1024,512,256 --dropout 0.15,0.15,0.15 --activation relu --save-confusion-csv result/final_noaug/confusion_matrix.csv --save-confusion-plot result/final_noaug/confusion_matrix.png --save-normalized-confusion-plot result/final_noaug/confusion_matrix_normalized.png
```

最终模型的权重可视化与错例分析命令：

```bash
python analysis.py --model-path result/final_aug/model/final_trial1_aug_60ep.pkl --num-neurons 16 --num-samples 5
```
