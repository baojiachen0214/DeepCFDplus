# 数据目录

该目录包含训练 DeepCFD+ 模型所需的数据集文件。

## 必需文件

- `dataX.pkl`：输入数据（几何表示）
- `dataY.pkl`：输出数据（速度和压力场）

## 数据格式

数据以 pickle 格式存储，结构如下：

- `dataX.pkl`：包含作为符号距离函数（SDF）表示的输入几何形状
- `dataY.pkl`：包含相应的速度（Ux, Uy）和压力（p）场

## 数据来源

原始数据集可从 [Zenodo](https://zenodo.org/record/3666056) 下载。

## 使用方法

训练脚本（`main.py`）默认会自动从该目录加载这些文件。