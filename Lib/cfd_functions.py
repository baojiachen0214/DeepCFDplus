import torch.nn as nn
import numpy as np
import os
from matplotlib import pyplot as plt


def split_tensors(*tensors, ratio):
    """
    按比例分割张量

    Args:
        *tensors: 待分割的张量列表
        ratio: 分割比例

    Returns:
        分割后的两部分张量
    """
    assert len(tensors) > 0
    split1, split2 = [], []
    count = len(tensors[0])
    for tensor in tensors:
        assert len(tensor) == count
        split1.append(tensor[:int(len(tensor) * ratio)])
        split2.append(tensor[int(len(tensor) * ratio):])
    if len(tensors) == 1:
        split1, split2 = split1[0], split2[0]
    return split1, split2


def initialize(model, gain=1, std=0.02):
    """
    初始化模型权重

    Args:
        model: 待初始化的模型
        gain: Xavier初始化增益系数
        std: 偏置初始化标准差
    """
    for module in model.modules():
        if type(module) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            nn.init.xavier_normal_(module.weight, gain)
            if module.bias is not None:
                nn.init.normal_(module.bias, 0, std)


def _plot_subplot(ax, data, title, cmap, vmin, vmax, ylabel=None, colorbar_orientation='horizontal',
                  colorbar_aspect=20, colorbar_pad=0.02, title_fontsize=20, label_fontsize=16, title_pad=5):
    """
    绘制单个子图的通用函数

    Args:
        ax: matplotlib轴对象
        data: 要绘制的数据
        title: 子图标题
        cmap: 颜色映射
        vmin: 颜色映射的最小值
        vmax: 颜色映射的最大值
        ylabel: Y轴标签
        colorbar_orientation: 颜色条方向
        colorbar_aspect: 颜色条长宽比
        colorbar_pad: 颜色条间距
        title_fontsize: 标题字体大小
        label_fontsize: 标签字体大小
        title_pad: 标题间距

    Returns:
        matplotlib图像对象
    """
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', extent=[0, 260, 0, 120])
    ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    cbar = plt.colorbar(im, orientation=colorbar_orientation, aspect=colorbar_aspect, pad=colorbar_pad)
    return im, cbar


def _calculate_ranges(sample_y, out_y, error, s, compare_data=None):
    """
    计算数据范围用于颜色映射

    Args:
        sample_y: 真实数据
        out_y: 预测数据
        error: 误差数据
        s: 样本索引
        compare_data: 对比数据（可选）

    Returns:
        (ranges, error_ranges) 元组
    """
    if compare_data is None:
        # 计算每个变量的数值范围，用于颜色映射
        ranges = []
        for i in range(3):
            min_val = min(np.min(sample_y[s, i, :, :]), np.min(out_y[s, i, :, :]))
            max_val = max(np.max(sample_y[s, i, :, :]), np.max(out_y[s, i, :, :]))
            ranges.append((min_val, max_val))

        error_ranges = []
        for i in range(3):
            min_val = np.min(error[s, i, :, :])
            max_val = np.max(error[s, i, :, :])
            error_ranges.append((min_val, max_val))
    else:
        # 对比模式 - 每行显示一个变量，共三行
        # 计算数值范围
        ranges = []
        for i in range(3):
            min_val = min(
                np.min(sample_y[s, i, :, :]),
                np.min(out_y[s, i, :, :]),
                np.min(compare_data['out_y'][s, i, :, :])
            )
            max_val = max(
                np.max(sample_y[s, i, :, :]),
                np.max(out_y[s, i, :, :]),
                np.max(compare_data['out_y'][s, i, :, :])
            )
            ranges.append((min_val, max_val))

        error_ranges = []
        for i in range(3):
            min_val = min(
                np.min(error[s, i, :, :]),
                np.min(compare_data['error'][s, i, :, :])
            )
            max_val = max(
                np.max(error[s, i, :, :]),
                np.max(compare_data['error'][s, i, :, :])
            )
            error_ranges.append((min_val, max_val))

    return ranges, error_ranges


def visualize(sample_y, out_y, error, s, model_name="Model", compare_data=None,
              figsize_scaling=1.2,
              font_size=10,
              color_map='jet',
              title_fontsize=20,
              label_fontsize=16,
              tick_fontsize=12,
              colorbar_orientation='horizontal',
              colorbar_aspect=20,
              colorbar_pad=0.02,
              title_pad=5,
              show_title=True,
              tight_layout_pad=1.08,
              timestamp=None):
    """
    可视化CFD结果 - 3x3网格布局版本

    Args:
        sample_y: 真实数据 [N, 3, H, W]
        out_y: 预测数据 [N, 3, H, W]
        error: 误差数据 [N, 3, H, W]
        s: 样本索引
        model_name: 模型名称
        compare_data (可选): 用于对比的字典
        figsize_scaling: 图像尺寸缩放因子
        font_size: 基础字体大小
        color_map: 配色方案
        title_fontsize: 标题字体大小
        label_fontsize: 标签字体大小
        tick_fontsize: 刻度字体大小
        colorbar_orientation: 颜色条方向 ('horizontal' 或 'vertical')
        colorbar_aspect: 颜色条长宽比
        colorbar_pad: 颜色条间距
        title_pad: 标题间距
        show_title: 是否显示标题
        tight_layout_pad: 紧凑布局间距
        timestamp: 时间戳，用于创建统一的结果目录
    """
    # 设置学术风格参数，支持中文字体
    plt.rcParams.update({
        'font.size': font_size,
        'font.family': ['DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif'],
        'axes.unicode_minus': False,
        'axes.linewidth': 1.0,
        'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize,
        'axes.labelsize': label_fontsize,
        'axes.titlesize': title_fontsize,
    })

    # 计算数据范围
    ranges, error_ranges = _calculate_ranges(sample_y, out_y, error, s, compare_data)

    if compare_data is None:
        # 创建3x3网格布局
        fig_width = 15 * figsize_scaling
        fig_height = 10 * figsize_scaling
        fig = plt.figure(figsize=(fig_width, fig_height))
        if show_title:
            fig.suptitle(f'CFD Field Prediction: {model_name}',
                         fontsize=title_fontsize + 2, fontweight='bold', y=0.98)

        # 第一行：Ux变量 (真实值, 预测值, 误差)
        ax1 = plt.subplot(3, 3, 1)
        _plot_subplot(ax1, np.transpose(sample_y[s, 0, :, :]), 'Ground Truth', color_map,
                      ranges[0][0], ranges[0][1], 'Ux', colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        # 预测 Ux
        ax2 = plt.subplot(3, 3, 2)
        _plot_subplot(ax2, np.transpose(out_y[s, 0, :, :]), f'{model_name}', color_map,
                      ranges[0][0], ranges[0][1], None, colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        # 误差 Ux
        ax3 = plt.subplot(3, 3, 3)
        _plot_subplot(ax3, np.transpose(error[s, 0, :, :]), 'Absolute Error', color_map,
                      error_ranges[0][0], error_ranges[0][1], None, colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        # 第二行：Uy变量 (真实值, 预测值, 误差)
        ax4 = plt.subplot(3, 3, 4)
        _plot_subplot(ax4, np.transpose(sample_y[s, 1, :, :]), '', color_map,
                      ranges[1][0], ranges[1][1], 'Uy', colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        # 预测 Uy
        ax5 = plt.subplot(3, 3, 5)
        _plot_subplot(ax5, np.transpose(out_y[s, 1, :, :]), '', color_map,
                      ranges[1][0], ranges[1][1], None, colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        # 误差 Uy
        ax6 = plt.subplot(3, 3, 6)
        _plot_subplot(ax6, np.transpose(error[s, 1, :, :]), '', color_map,
                      error_ranges[1][0], error_ranges[1][1], None, colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        # 第三行：p变量 (真实值, 预测值, 误差)
        ax7 = plt.subplot(3, 3, 7)
        _plot_subplot(ax7, np.transpose(sample_y[s, 2, :, :]), '', color_map,
                      ranges[2][0], ranges[2][1], 'p', colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        # 预测 p
        ax8 = plt.subplot(3, 3, 8)
        _plot_subplot(ax8, np.transpose(out_y[s, 2, :, :]), '', color_map,
                      ranges[2][0], ranges[2][1], None, colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        # 误差 p
        ax9 = plt.subplot(3, 3, 9)
        _plot_subplot(ax9, np.transpose(error[s, 2, :, :]), '', color_map,
                      error_ranges[2][0], error_ranges[2][1], None, colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

    else:
        # 对比模式 - 每行显示一个变量，共三行
        model1_name = model_name
        model2_name = compare_data['model_name']

        # 创建3x4网格布局
        fig_width = 18 * figsize_scaling
        fig_height = 10 * figsize_scaling
        fig = plt.figure(figsize=(fig_width, fig_height))
        if show_title:
            fig.suptitle(f'CFD Field Prediction Comparison',
                         fontsize=title_fontsize + 2, fontweight='bold', y=0.98)

        # 第一行：Ux变量
        ax1 = plt.subplot(3, 4, 1)
        _plot_subplot(ax1, np.transpose(sample_y[s, 0, :, :]), 'Ground Truth', color_map,
                      ranges[0][0], ranges[0][1], 'Ux', colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        ax2 = plt.subplot(3, 4, 2)
        _plot_subplot(ax2, np.transpose(out_y[s, 0, :, :]), f'{model1_name}', color_map,
                      ranges[0][0], ranges[0][1], None, colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        ax3 = plt.subplot(3, 4, 3)
        _plot_subplot(ax3, np.transpose(compare_data['out_y'][s, 0, :, :]), f'{model2_name}', color_map,
                      ranges[0][0], ranges[0][1], None, colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        ax4 = plt.subplot(3, 4, 4)
        _plot_subplot(ax4, np.transpose(error[s, 0, :, :]), 'Error Comparison', color_map,
                      error_ranges[0][0], error_ranges[0][1], None, colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        # 第二行：Uy变量
        ax5 = plt.subplot(3, 4, 5)
        _plot_subplot(ax5, np.transpose(sample_y[s, 1, :, :]), '', color_map,
                      ranges[1][0], ranges[1][1], 'Uy', colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        ax6 = plt.subplot(3, 4, 6)
        _plot_subplot(ax6, np.transpose(out_y[s, 1, :, :]), '', color_map,
                      ranges[1][0], ranges[1][1], None, colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        ax7 = plt.subplot(3, 4, 7)
        _plot_subplot(ax7, np.transpose(compare_data['out_y'][s, 1, :, :]), '', color_map,
                      ranges[1][0], ranges[1][1], None, colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        ax8 = plt.subplot(3, 4, 8)
        _plot_subplot(ax8, np.transpose(error[s, 1, :, :]), '', color_map,
                      error_ranges[1][0], error_ranges[1][1], None, colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        # 第三行：p变量
        ax9 = plt.subplot(3, 4, 9)
        _plot_subplot(ax9, np.transpose(sample_y[s, 2, :, :]), '', color_map,
                      ranges[2][0], ranges[2][1], 'p', colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        ax10 = plt.subplot(3, 4, 10)
        _plot_subplot(ax10, np.transpose(out_y[s, 2, :, :]), '', color_map,
                      ranges[2][0], ranges[2][1], None, colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        ax11 = plt.subplot(3, 4, 11)
        _plot_subplot(ax11, np.transpose(compare_data['out_y'][s, 2, :, :]), '', color_map,
                      ranges[2][0], ranges[2][1], None, colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

        ax12 = plt.subplot(3, 4, 12)
        _plot_subplot(ax12, np.transpose(error[s, 2, :, :]), '', color_map,
                      error_ranges[2][0], error_ranges[2][1], None, colorbar_orientation, colorbar_aspect,
                      colorbar_pad, title_fontsize, label_fontsize, title_pad)

    plt.tight_layout(pad=tight_layout_pad)

    # 使用传入的时间戳创建统一的结果目录
    if timestamp is not None:
        save_dir = f"./results/{timestamp}"
    else:
        save_dir = "./results"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 根据模型名称和样本索引生成文件名
    if compare_data is None:
        filename = f"{model_name}_可视化结果.png"
    else:
        filename = f"{model_name}_对比可视化结果.png"

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存到: {save_path}")

    plt.show()
    plt.close()
