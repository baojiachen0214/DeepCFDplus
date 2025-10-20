import json
import pickle
from Lib.train_model import *
from Lib.cfd_functions import *
from torch.utils.data import TensorDataset
from Models.UNet import UNet
from Models.UNetEx import UNetEx
from Models.UNetExAvg import UNetEx as UNetExAvg
from Models.UNetExMod import UNetExMod
import logging
from datetime import datetime
import numpy as np
import yaml
import traceback
import torch.nn.functional as F
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.use('Agg')  # 使用非交互式后端，如果最后希望得到交互式绘图，请将此处的“Agg”改为“TkAgg”


def setup_logging():
    """设置日志系统"""
    # 创建logs目录
    log_dir = "./logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 设置日志格式
    log_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)

    # 配置根日志（避免重复）
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | %(message)s',
            handlers=[
                logging.FileHandler(log_path, mode='w', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    return logging.getLogger(__name__)


def log_model_info(model, model_name, device, **kwargs):
    """记录模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("=" * 80)
    logger.info(f"{'模型架构':<12} : {model_name}")
    logger.info(f"{'总参数量':<12} : {total_params:,}")
    logger.info(f"{'可训练参数量':<12} : {trainable_params:,}")
    logger.info(f"{'训练设备':<12} : {device}")
    logger.info("-" * 80)

    for key, value in kwargs.items():
        logger.info(f"{key:<12} : {value}")
    logger.info("=" * 80)


def log_training_progress_simple(epoch, total_epochs, train_loss, val_loss, train_mse, val_mse):
    """简化版训练进度记录"""
    progress = (epoch) / total_epochs * 100  # 修复轮次计数问题
    train_rmse = np.sqrt(train_mse)
    val_rmse = np.sqrt(val_mse)

    if epoch == 1:
        # 使用固定宽度的中文表头，确保对齐
        logger.info("轮次     | 进度     | 训练损失      | 验证损失      | 训练RMSE      | 验证RMSE      ")
        logger.info("-" * 80)

    # 使用固定宽度格式化，确保中文日志对齐
    logger.info(
        f"{epoch:<8} | {progress:>7.0f}% | {train_loss:>12.6f} | {val_loss:>12.6f} | {train_rmse:>12.6f} | {val_rmse:>12.6f}")


def log_final_results_simple(train_mse, val_mse, train_loss, val_loss, epochs_trained):
    """简化版最终结果记录"""
    logger.info("=" * 80)
    logger.info("训练总结")
    logger.info("=" * 80)
    logger.info(f"{'总轮次':<12} : {epochs_trained}")
    logger.info(f"{'最终训练损失':<12} : {train_loss:.6f}")
    logger.info(f"{'最终验证损失':<12} : {val_loss:.6f}")
    logger.info(f"{'最终训练MSE':<12} : {train_mse:.6f}")
    logger.info(f"{'最终验证MSE':<12} : {val_mse:.6f}")
    train_rmse = np.sqrt(train_mse)
    val_rmse = np.sqrt(val_mse)
    logger.info(f"{'最终训练RMSE':<12} : {train_rmse:.6f}")
    logger.info(f"{'最终验证RMSE':<12} : {val_rmse:.6f}")
    logger.info("=" * 80)


def save_results_simple(train_loss_curve, val_loss_curve, train_mse_curve, val_mse_curve, config, model_name, timestamp,
                        extra_metrics=None):
    """简化版结果保存"""
    results = {
        'train_loss_curve': train_loss_curve,
        'val_loss_curve': val_loss_curve,
        'train_mse_curve': train_mse_curve,
        'val_mse_curve': val_mse_curve,
        'config': config,
        'model_name': model_name,
        'timestamp': datetime.now().isoformat()
    }

    # 创建统一的结果目录
    simulation_directory = f"./results/{timestamp}"
    if not os.path.exists(simulation_directory):
        os.makedirs(simulation_directory)

    results_path = os.path.join(simulation_directory,
                                f"{model_name}_训练结果.json")
    with open(results_path, "w", encoding='utf-8') as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    logger.info(f"结果已保存至: {results_path}")

    # 保存训练过程数据为CSV
    epochs = list(range(1, len(train_loss_curve) + 1))
    train_rmse_curve = [np.sqrt(mse) for mse in train_mse_curve]
    val_rmse_curve = [np.sqrt(mse) for mse in val_mse_curve]

    # 准备数据用于CSV
    training_data_dict = {
        '轮次': epochs,
        '训练损失': train_loss_curve,
        '验证损失': val_loss_curve,
        '训练MSE': train_mse_curve,
        '验证MSE': val_mse_curve,
        '训练RMSE': train_rmse_curve,
        '验证RMSE': val_rmse_curve
    }

    # 添加额外指标（如果有）
    if extra_metrics:
        # 将英文键转换为中文键
        chinese_keys = {
            'Train Ux MSE': '训练Ux MSE',
            'Val Ux MSE': '验证Ux MSE',
            'Train Uy MSE': '训练Uy MSE',
            'Val Uy MSE': '验证Uy MSE',
            'Train p MSE': '训练p MSE',
            'Val p MSE': '验证p MSE'
        }
        for key, value in extra_metrics.items():
            chinese_key = chinese_keys.get(key, key)
            training_data_dict[chinese_key] = value

    training_data = pd.DataFrame(training_data_dict)

    csv_path = os.path.join(simulation_directory, f"{model_name}_训练数据.csv")
    training_data.to_csv(csv_path, index=False)
    logger.info(f"训练数据已保存至: {csv_path}")

    # 绘制训练过程图表
    # 设置英文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 如果有额外指标，为每个变量创建单独的图表
    if extra_metrics and all(key in extra_metrics for key in [
        'Train Ux MSE', 'Val Ux MSE', 'Train Uy MSE', 'Val Uy MSE', 'Train p MSE', 'Val p MSE'
    ]):
        train_ux_mse = extra_metrics['Train Ux MSE']
        val_ux_mse = extra_metrics['Val Ux MSE']
        train_uy_mse = extra_metrics['Train Uy MSE']
        val_uy_mse = extra_metrics['Val Uy MSE']
        train_p_mse = extra_metrics['Train p MSE']
        val_p_mse = extra_metrics['Val p MSE']

        # 计算每个变量的RMSE
        train_ux_rmse = [np.sqrt(mse) for mse in train_ux_mse]
        val_ux_rmse = [np.sqrt(mse) for mse in val_ux_mse]
        train_uy_rmse = [np.sqrt(mse) for mse in train_uy_mse]
        val_uy_rmse = [np.sqrt(mse) for mse in val_uy_mse]
        train_p_rmse = [np.sqrt(mse) for mse in train_p_mse]
        val_p_rmse = [np.sqrt(mse) for mse in val_p_mse]

        # 为每个变量创建单独的图表，训练和验证曲线分开显示
        # 创建3行4列的图表布局 (3个变量 x 2种指标 x 2种曲线类型 = 12张图)
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'Training Curves - {model_name}', fontsize=16, fontweight='bold')

        # Ux MSE
        axes[0, 0].plot(epochs, train_ux_mse, label='Train', linewidth=2)
        axes[0, 0].set_title('Ux Train MSE', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, val_ux_mse, label='Validation', linewidth=2, color='orange')
        axes[0, 1].set_title('Ux Validation MSE', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Ux RMSE
        axes[0, 2].plot(epochs, train_ux_rmse, label='Train', linewidth=2, color='green')
        axes[0, 2].set_title('Ux Train RMSE', fontsize=14)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('RMSE')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        axes[0, 3].plot(epochs, val_ux_rmse, label='Validation', linewidth=2, color='red')
        axes[0, 3].set_title('Ux Validation RMSE', fontsize=14)
        axes[0, 3].set_xlabel('Epoch')
        axes[0, 3].set_ylabel('RMSE')
        axes[0, 3].legend()
        axes[0, 3].grid(True, alpha=0.3)

        # Uy MSE
        axes[1, 0].plot(epochs, train_uy_mse, label='Train', linewidth=2)
        axes[1, 0].set_title('Uy Train MSE', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(epochs, val_uy_mse, label='Validation', linewidth=2, color='orange')
        axes[1, 1].set_title('Uy Validation MSE', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Uy RMSE
        axes[1, 2].plot(epochs, train_uy_rmse, label='Train', linewidth=2, color='green')
        axes[1, 2].set_title('Uy Train RMSE', fontsize=14)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('RMSE')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        axes[1, 3].plot(epochs, val_uy_rmse, label='Validation', linewidth=2, color='red')
        axes[1, 3].set_title('Uy Validation RMSE', fontsize=14)
        axes[1, 3].set_xlabel('Epoch')
        axes[1, 3].set_ylabel('RMSE')
        axes[1, 3].legend()
        axes[1, 3].grid(True, alpha=0.3)

        # p MSE
        axes[2, 0].plot(epochs, train_p_mse, label='Train', linewidth=2)
        axes[2, 0].set_title('p Train MSE', fontsize=14)
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('MSE')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        axes[2, 1].plot(epochs, val_p_mse, label='Validation', linewidth=2, color='orange')
        axes[2, 1].set_title('p Validation MSE', fontsize=14)
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('MSE')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

        # p RMSE
        axes[2, 2].plot(epochs, train_p_rmse, label='Train', linewidth=2, color='green')
        axes[2, 2].set_title('p Train RMSE', fontsize=14)
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].set_ylabel('RMSE')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)

        axes[2, 3].plot(epochs, val_p_rmse, label='Validation', linewidth=2, color='red')
        axes[2, 3].set_title('p Validation RMSE', fontsize=14)
        axes[2, 3].set_xlabel('Epoch')
        axes[2, 3].set_ylabel('RMSE')
        axes[2, 3].legend()
        axes[2, 3].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(simulation_directory, f"{model_name}_训练曲线.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # 默认图表（现有行为）
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_loss_curve, label='Train Loss', linewidth=2)
        plt.plot(epochs, val_loss_curve, label='Validation Loss', linewidth=2)
        plt.title('Loss Curves', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(epochs, train_mse_curve, label='Train MSE', linewidth=2)
        plt.plot(epochs, val_mse_curve, label='Validation MSE', linewidth=2)
        plt.title('MSE Curves', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.plot(epochs, train_rmse_curve, label='Train RMSE', linewidth=2)
        plt.plot(epochs, val_rmse_curve, label='Validation RMSE', linewidth=2)
        plt.title('RMSE Curves', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(simulation_directory, f"{model_name}_训练曲线.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    logger.info(f"训练曲线已保存至: {plot_path}")


def compare_models_error(trained_models, sample_index=0, timestamp=None):
    """
    对比多个模型的绝对误差
    
    Args:
        trained_models: 训练好的模型字典
        sample_index: 样本索引
        timestamp: 时间戳
    """
    if len(trained_models) < 2:
        logger.info("至少需要两个模型才能进行对比。")
        return

    logger.info(f"正在对比 {len(trained_models)} 个模型的绝对误差...")

    # 获取真实值（假设所有模型的 test_y 是相同的）
    first_trainer = list(trained_models.values())[0][0]  # 获取第一个训练器
    sample_x, sample_y = first_trainer.test_x[:10].to(first_trainer.device), first_trainer.test_y[:10].to(
        first_trainer.device)
    true_values = sample_y.cpu().numpy()

    # 收集所有模型的预测和误差
    model_errors = {}
    model_names = []

    for model_name, (trainer, model, _, _) in trained_models.items():
        model_names.append(model_name)
        trainer.model.eval()
        with torch.no_grad():
            # UNet系列模型
            out = trainer.model(sample_x).cpu()
            out_spatial = out.numpy()
        error = np.abs(true_values - out_spatial)
        model_errors[model_name] = error

    # 动态计算统一的颜色范围 - 使用百分位数方法确保更好的颜色分布
    all_errors = np.concatenate([error.flatten() for error in model_errors.values()])
    # 使用99%分位数作为最大值，避免异常值影响整体显示效果
    vmax = np.percentile(all_errors, 99)
    # 使用1%分位数作为最小值
    vmin = np.percentile(all_errors, 1)
    logger.info(f"统一颜色标尺范围: {vmin:.4f} - {vmax:.4f}")

    # 动态计算布局
    n_models = len(model_names)
    variables = ['Ux', 'Uy', 'p']
    variable_indices = [0, 1, 2]

    # 创建对比图表，将三个变量(Ux, Uy, p)放在同一张图中
    fig_width = min(5 * n_models, 20)  # 限制最大宽度
    fig_height = 10  # 减小高度以减少行间距
    fig, axes = plt.subplots(3, n_models, figsize=(fig_width, fig_height))
    fig.suptitle('Model Absolute Error Comparison', fontsize=24, y=0.95)

    # 绘制每个模型的误差图
    for var_idx, var_name in zip(variable_indices, variables):
        for i, model_name in enumerate(model_names):
            ax = axes[var_idx, i] if n_models > 1 else axes[var_idx]

            # 显示误差
            im = ax.imshow(np.transpose(model_errors[model_name][sample_index, var_idx, :, :]),
                           cmap='jet', vmin=vmin, vmax=vmax, origin='lower', extent=[0, 260, 0, 120])

            # 设置模型名称（只在第一行显示），不加粗
            if var_idx == 0:
                ax.set_title(model_name, fontsize=18, pad=10)

            # 不显示坐标轴标签
            ax.set_xticks([])
            ax.set_yticks([])

            # 只在第一列显示变量名称
            if i == 0:
                ax.set_ylabel(var_name, fontsize=16, rotation=90, labelpad=20)

            # 为每个子图添加颜色条，减小颜色条与图像的距离
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0.1)  # 减小pad值使颜色条更靠近图像
            cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
            cbar.set_label(f'{var_name} Absolute Error', fontsize=10)
            cbar.ax.tick_params(labelsize=8)

    # 调整子图间距，使布局更紧凑
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # 保存图表
    if timestamp is not None:
        save_dir = f"./results/{timestamp}"
    else:
        save_dir = "./results"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = "模型误差对比.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"模型绝对误差对比图已保存至: {save_path}")


class ModelTrainer:
    """统一的模型训练器"""

    # 类变量用于存储已加载的数据
    _loaded_data = None
    _data_path = None

    def __init__(self, model_name, data_path="./", timestamp=None, **kwargs):
        self.model_name = model_name
        self.data_path = data_path
        self.timestamp = timestamp
        self.kwargs = kwargs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载数据（添加异常处理）
        try:
            # 检查是否已经加载过相同路径的数据
            if ModelTrainer._loaded_data is None or ModelTrainer._data_path != data_path:
                self.x = pickle.load(open(f"{self.data_path}dataX.pkl", "rb"))
                self.y = pickle.load(open(f"{self.data_path}dataY.pkl", "rb"))
                self.x = torch.FloatTensor(self.x)
                self.y = torch.FloatTensor(self.y)
                logger.info(f"数据加载完成: x.shape={self.x.shape}, y.shape={self.y.shape}")

                # 保存加载的数据和路径
                ModelTrainer._loaded_data = (self.x, self.y)
                ModelTrainer._data_path = data_path
            else:
                # 使用已加载的数据
                self.x, self.y = ModelTrainer._loaded_data
                logger.info(f"使用已加载的数据: x.shape={self.x.shape}, y.shape={self.y.shape}")
        except Exception as e:
            logger.error(f"数据加载失败: {e}\n{traceback.format_exc()}")
            raise

        self.save_every_n_epochs = self.kwargs.get('save_every_n_epochs', 0)
        self.save_best_every_n_epochs = self.kwargs.get('save_best_every_n_epochs', 1)
        # 使用统一的检查点目录
        base_save_dir = self.kwargs.get('save_dir', './checkpoints')
        self.save_dir = os.path.join(base_save_dir, self.timestamp, self.model_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 根据模型类型设置数据处理方式
        # UNet系列都是处理 (B, C, H, W) 格式，都需要通道权重
        if model_name in ["UNet", "UNetEx", "UNetExAvg", "UNetExMod"]:
            self.channels_weights = torch.sqrt(
                torch.mean(self.y.permute(0, 2, 3, 1).reshape((-1, 3)) ** 2, dim=0)
            ).view(1, -1, 1, 1).to(self.device)
        else:
            self.channels_weights = 1.0

    def prepare_data(self, train_ratio=0.7):
        """准备训练和验证数据（增强形状检查）"""
        try:
            # 根据模型类型处理数据形状
            # UNet系列: (N, C, H, W) -> (N, C, H_pad, W_pad)
            # 检查并填充尺寸，使其能被 4 整除 (因为有2次下采样)
            _, _, H, W = self.x.shape
            pad_h = (4 - H % 4) % 4
            pad_w = (4 - W % 4) % 4
            if pad_h > 0 or pad_w > 0:
                # 使用反射填充 (reflect padding) 通常比零填充效果好
                x_padded = F.pad(self.x, (0, pad_w, 0, pad_h), mode='reflect')
                y_padded = F.pad(self.y, (0, pad_w, 0, pad_h), mode='reflect')
                logger.info(f"数据填充: 从 ({H}, {W}) 填充到 ({H + pad_h}, {W + pad_w})")
            else:
                x_padded = self.x
                y_padded = self.y

            x_seq = x_padded
            y_seq = y_padded
            self.input_dim = x_padded.shape[1]
            self.spatial_shape = (x_padded.shape[1], x_padded.shape[2], x_padded.shape[3])  # (C, H_pad, W_pad)
            # 保存原始尺寸，用于后续可视化时裁剪回原尺寸
            self.original_shape = (H, W)

            logger.info(f"处理后数据形状: x_seq={x_seq.shape}, y_seq={y_seq.shape}")

            # 分割数据
            train_data, val_data = split_tensors(x_seq, y_seq, ratio=train_ratio)
            train_ratio_adjusted = 0.9
            train_data_adjusted, test_data = split_tensors(train_data[0], train_data[1], ratio=train_ratio_adjusted)

            self.train_dataset = TensorDataset(train_data_adjusted[0], train_data_adjusted[1])
            self.val_dataset = TensorDataset(val_data[0], val_data[1])
            self.test_dataset = TensorDataset(test_data[0], test_data[1])

            # 获取测试数据
            test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=len(self.test_dataset),
                                                      shuffle=False)
            for batch in test_loader:
                self.test_x, self.test_y = batch
                break

            logger.info(
                f"训练样本数: {len(self.train_dataset)}, 验证样本数: {len(self.val_dataset)}, 测试样本数: {len(self.test_dataset)}")
        except Exception as e:
            logger.error(f"数据准备失败: {e}\n{traceback.format_exc()}")
            raise

    def create_model(self):
        """创建模型（添加异常处理）"""
        try:
            if self.model_name == "UNet":
                lr = self.kwargs.get('lr', 0.001)
                kernel_size = self.kwargs.get('kernel_size', 3)
                filters = self.kwargs.get('filters', [16, 32, 64])
                bn = self.kwargs.get('bn', True)
                wn = self.kwargs.get('wn', True)
                wd = self.kwargs.get('wd', 0.005)
                layers = self.kwargs.get('layers', 2)

                self.model = UNet(3, 3, filters=filters, kernel_size=kernel_size, layers=layers,
                                  batch_norm=bn, weight_norm=wn)
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
                self.loss_func = self.unet_loss_func
                self.batch_size = self.kwargs.get('batch_size', 64)
                self.epochs = self.kwargs.get('epochs', 1000)
                self.patience = self.kwargs.get('patience', 25)

            elif self.model_name == "UNetEx":
                lr = self.kwargs.get('lr', 0.001)
                kernel_size = self.kwargs.get('kernel_size', 5)
                filters = self.kwargs.get('filters', [8, 16, 32, 32])
                bn = self.kwargs.get('bn', False)
                wn = self.kwargs.get('wn', False)
                wd = self.kwargs.get('wd', 0.005)
                layers = self.kwargs.get('layers', 3)

                self.model = UNetEx(3, 3, filters=filters, kernel_size=kernel_size, layers=layers,
                                    batch_norm=bn, weight_norm=wn)
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
                self.loss_func = self.unet_loss_func
                self.batch_size = self.kwargs.get('batch_size', 64)
                self.epochs = self.kwargs.get('epochs', 1000)
                self.patience = self.kwargs.get('patience', 25)

            elif self.model_name == "UNetExAvg":
                lr = self.kwargs.get('lr', 0.001)
                kernel_size = self.kwargs.get('kernel_size', 3)
                filters = self.kwargs.get('filters', [16, 32, 64])
                bn = self.kwargs.get('bn', True)
                wn = self.kwargs.get('wn', True)
                wd = self.kwargs.get('wd', 0.005)
                layers = self.kwargs.get('layers', 2)

                self.model = UNetExAvg(3, 3, filters=filters, kernel_size=kernel_size, layers=layers,
                                       batch_norm=bn, weight_norm=wn)
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
                self.loss_func = self.unet_loss_func
                self.batch_size = self.kwargs.get('batch_size', 64)
                self.epochs = self.kwargs.get('epochs', 1000)
                self.patience = self.kwargs.get('patience', 25)

            elif self.model_name == "UNetExMod":
                lr = self.kwargs.get('lr', 0.001)
                kernel_size = self.kwargs.get('kernel_size', 3)
                filters = self.kwargs.get('filters', [16, 32, 64])
                bn = self.kwargs.get('bn', True)
                wn = self.kwargs.get('wn', True)
                wd = self.kwargs.get('wd', 0.005)
                layers = self.kwargs.get('layers', 3)

                self.model = UNetExMod(3, 3, filters=filters, kernel_size=kernel_size, layers=layers,
                                       batch_norm=bn, weight_norm=wn)
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
                self.loss_func = self.unet_loss_func
                self.batch_size = self.kwargs.get('batch_size', 64)
                self.epochs = self.kwargs.get('epochs', 1000)
                self.patience = self.kwargs.get('patience', 25)

            else:
                raise ValueError(f"不支持的模型: {self.model_name}")

            # 为UNet系列模型添加通用指标
            if self.model_name in ["UNet", "UNetEx", "UNetExAvg", "UNetExMod"]:
                self.extra_metrics = {
                    'm_ux_name': 'Ux MSE',
                    'm_ux_on_batch': lambda scope: float(
                        torch.sum((scope["output"][:, 0, :, :] - scope["batch"][1][:, 0, :, :]) ** 2)),
                    'm_ux_on_epoch': lambda scope: sum(scope["list"]) / len(scope["dataset"]),
                    'm_uy_name': 'Uy MSE',
                    'm_uy_on_batch': lambda scope: float(
                        torch.sum((scope["output"][:, 1, :, :] - scope["batch"][1][:, 1, :, :]) ** 2)),
                    'm_uy_on_epoch': lambda scope: sum(scope["list"]) / len(scope["dataset"]),
                    'm_p_name': 'p MSE',
                    'm_p_on_batch': lambda scope: float(
                        torch.sum((scope["output"][:, 2, :, :] - scope["batch"][1][:, 2, :, :]) ** 2)),
                    'm_p_on_epoch': lambda scope: sum(scope["list"]) / len(scope["dataset"]),
                }

            # 移动模型到设备
            self.model = self.model.to(self.device)

            # 记录模型信息
            log_model_info(
                self.model,
                self.model_name,
                self.device,
                学习率=self.kwargs.get('lr', 0.001),
                权重衰减=self.kwargs.get('wd', 0.0),
                批次大小=self.batch_size,
                训练轮次=self.epochs,
                输入维度=getattr(self, 'input_dim', 'N/A')
            )
        except Exception as e:
            logger.error(f"模型创建失败: {e}\n{traceback.format_exc()}")
            raise

    def unet_loss_func(self, model, batch):
        """UNet系列模型损失函数（分别加权通道）"""
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        output = model(x)
        B, C, H, W = output.shape

        # 计算每个通道的 MSE，并 reshape 为 (B,1,H,W)
        lossu = ((output[:, 0] - y[:, 0]) ** 2).unsqueeze(1)  # (B,1,H,W)
        lossv = ((output[:, 1] - y[:, 1]) ** 2).unsqueeze(1)
        lossp = ((output[:, 2] - y[:, 2]) ** 2).unsqueeze(1)  # 统一用 **2 (MSE)

        # 分别加权（channels_weights: (1,3,1,1））
        weighted_lossu = lossu / self.channels_weights[:, 0:1, :, :]
        weighted_lossv = lossv / self.channels_weights[:, 1:2, :, :]
        weighted_lossp = lossp / self.channels_weights[:, 2:3, :, :]

        # 求和并 mean
        loss = (weighted_lossu + weighted_lossv + weighted_lossp).mean()
        return loss, output

    def visualize_results(self, timestamp):
        """可视化结果（添加异常处理）"""
        try:
            logger.info("执行推理以进行可视化...")
            self.model.eval()
            with torch.no_grad():
                # UNet系列模型
                out_padded = self.model(self.test_x[:10].to(self.device))
                # 裁剪回原始尺寸
                if hasattr(self, 'original_shape'):
                    orig_H, orig_W = self.original_shape
                    out = out_padded[:, :, :orig_H, :orig_W]
                    true_y_vis = self.test_y[:10][:, :, :orig_H, :orig_W]
                else:
                    out = out_padded
                    true_y_vis = self.test_y[:10]
                error = torch.abs(out.cpu() - true_y_vis.cpu())
                visualize(true_y_vis.cpu().numpy(), out.cpu().numpy(), error.numpy(), 0,
                          model_name=self.model_name, timestamp=timestamp)

            logger.info(f"{self.model_name} 可视化完成")
            # 返回预测结果和误差用于模型对比
            return out.cpu().numpy(), error.numpy()
        except Exception as e:
            logger.error(f"可视化失败: {e}\n{traceback.format_exc()}")
            return None, None

    def train(self, resume_path=None):
        """Train model (enhanced exception handling)"""
        logger.info(f"开始训练 {self.model_name}...")

        train_loss_curve = []
        val_loss_curve = []
        train_mse_curve = []
        val_mse_curve = []

        # Per-variable metric curves for UNet series
        train_ux_mse_curve = []
        val_ux_mse_curve = []
        train_uy_mse_curve = []
        val_uy_mse_curve = []
        train_p_mse_curve = []
        val_p_mse_curve = []

        def after_epoch(scope):
            train_loss_curve.append(scope["train_loss"])
            val_loss_curve.append(scope["val_loss"])
            train_mse = scope.get("train_mse", scope["train_loss"])
            val_mse = scope.get("val_mse", scope["val_loss"])
            train_mse_curve.append(train_mse)
            val_mse_curve.append(val_mse)

            # Collect per-variable metrics for UNet series
            if self.model_name in ["UNet", "UNetEx", "UNetExAvg", "UNetExMod"]:
                train_ux_mse_curve.append(scope.get("train_ux", 0))
                val_ux_mse_curve.append(scope.get("val_ux", 0))
                train_uy_mse_curve.append(scope.get("train_uy", 0))
                val_uy_mse_curve.append(scope.get("val_uy", 0))
                train_p_mse_curve.append(scope.get("train_p", 0))
                val_p_mse_curve.append(scope.get("val_p", 0))

            try:
                current_epoch = scope.get("epoch", len(train_loss_curve))
                total_epochs = scope.get("epochs", self.epochs)

                log_training_progress_simple(
                    epoch=current_epoch,
                    total_epochs=total_epochs,
                    train_loss=scope["train_loss"],
                    val_loss=scope["val_loss"],
                    train_mse=train_mse,
                    val_mse=val_mse
                )
            except Exception as e:
                logger.warning(f"after_epoch回调错误: {e}")

        # Prepare training parameters
        train_kwargs = {
            'model': self.model,
            'loss_func': self.loss_func,
            'train_dataset': self.train_dataset,
            'val_dataset': self.val_dataset,
            'optimizer': self.optimizer,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'patience': self.patience,
            'device': self.device,
            'resume_path': resume_path,
            'save_every_n_epochs': self.save_every_n_epochs,
            'save_best_every_n_epochs': self.save_best_every_n_epochs,
            'save_dir': self.save_dir,
            'm_mse_name': "总MSE",
            'm_mse_on_batch': lambda scope: float(torch.sum((scope["output"] - scope["batch"][1]) ** 2)),
            'm_mse_on_epoch': lambda scope: sum(scope["list"]) / len(scope["dataset"]),
            'after_epoch': after_epoch
        }

        # Add model-specific metrics
        if self.model_name in ["UNet", "UNetEx", "UNetExAvg", "UNetExMod"]:
            train_kwargs.update(self.extra_metrics)

        try:
            # Train model
            result = train_model(**train_kwargs)

            # Process return values
            if len(result) == 5:
                self.model, train_metrics, train_loss, val_metrics, val_loss = result
                train_mse = train_metrics.get('mse', train_loss) if isinstance(train_metrics, dict) else train_loss
                val_mse = val_metrics.get('mse', val_loss) if isinstance(val_metrics, dict) else val_loss
            else:
                self.model = result[0]
                train_loss = result[2] if len(result) > 2 else 0
                val_loss = result[4] if len(result) > 4 else 0
                train_mse = train_loss
                val_mse = val_loss

            # Save results
            extra_metrics_data = None
            if self.model_name in ["UNet", "UNetEx", "UNetExAvg", "UNetExMod"]:
                # Collect per-variable metrics for CSV
                extra_metrics_data = {
                    'Train Ux MSE': train_ux_mse_curve,
                    'Val Ux MSE': val_ux_mse_curve,
                    'Train Uy MSE': train_uy_mse_curve,
                    'Val Uy MSE': val_uy_mse_curve,
                    'Train p MSE': train_p_mse_curve,
                    'Val p MSE': val_p_mse_curve
                }

            config = {
                "训练损失": train_loss,
                "验证损失": val_loss,
                "训练MSE": train_mse,
                "验证MSE": val_mse,
                "模型名称": self.model_name,
                "输入维度": getattr(self, 'input_dim', 'N/A'),
                "空间形状": getattr(self, 'spatial_shape', 'N/A'),
                **self.kwargs
            }
            save_results_simple(train_loss_curve, val_loss_curve, train_mse_curve, val_mse_curve, config,
                                self.model_name, self.timestamp,
                                extra_metrics_data)

            # Record final results
            log_final_results_simple(train_mse, val_mse, train_loss, val_loss, len(train_loss_curve))

            return self.model, {'mse': train_mse}, {'mse': val_mse}
        except Exception as e:
            logger.error(f"训练失败: {e}\n{traceback.format_exc()}")
            raise


def load_config(config_path):
    """加载YAML配置文件（添加异常处理）"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"配置加载失败: {e}\n{traceback.format_exc()}")
        raise FileNotFoundError(f"YAML文件 {config_path} 不存在或格式错误。请检查 config.yaml。")


def run_training(_config, logger):
    """执行训练流程（增强鲁棒性）"""
    if logger is None:
        raise ValueError("Logger为全局配置！")

    # 为本次运行创建统一的时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        data_path = _config.get('data_path', './data/')  # 默认值
        logger.info(f"使用数据路径: {data_path}")

        # 构建训练配置
        models_config = _config.get('models', {})
        training_configs = []
        for model_name, model_config in models_config.items():
            config_dict = {
                'model_name': model_name,
                'data_path': data_path,
                'timestamp': timestamp,
                **model_config
            }
            training_configs.append(config_dict)

        if not training_configs:
            raise ValueError("YAML中未指定模型!")

        # 恢复训练配置（默认值）
        training_section = _config.get('training', {})
        resume_model_name = training_section.get('resume_model_name')
        resume_path = training_section.get('resume_path')
        train_ratio = training_section.get('train_ratio', 0.7)

        # 训练所有模型
        trained_models = {}
        logger.info("=" * 80)
        logger.info("开始模型训练流程")
        logger.info("=" * 80)

        for config_item in training_configs:
            model_name = config_item['model_name']
            logger.info(f"\n>>> 准备训练模型: {model_name} <<<")

            try:
                trainer = ModelTrainer(**config_item)
                trainer.prepare_data(train_ratio=train_ratio)
                trainer.create_model()

                if resume_path and resume_model_name == model_name and os.path.exists(resume_path):
                    logger.info(f"从 {resume_path} 恢复 {model_name} 训练...")
                    model, train_metrics, val_metrics = trainer.train(resume_path=resume_path)
                else:
                    logger.info(f"从头开始训练 {model_name}...")
                    model, train_metrics, val_metrics = trainer.train(resume_path=None)

                trainer.visualize_results(timestamp)
                trained_models[model_name] = (trainer, model, train_metrics, val_metrics)
                logger.info(f"<<< {model_name} 训练完成 >>>\n")
            except Exception as e:
                logger.error(f"{model_name} 训练失败: {e}\n{traceback.format_exc()}")
                continue  # 跳过失败模型，继续下一个

        # 生成模型误差对比图
        if len(trained_models) >= 2:
            compare_models_error(trained_models, sample_index=0, timestamp=timestamp)

        logger.info("=" * 80)
        logger.info("所有模型训练完成!")
        logger.info("=" * 80)
        return trained_models
    except Exception as e:
        logger.error(f"训练流程执行失败: {e}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    # 加载配置并运行训练
    try:
        user_config = load_config('config.yaml')
        logger = setup_logging()  # 全局 logger
        logger.info("深度CFD训练程序启动")
        logger.info("=" * 80)
        trained_models = run_training(user_config, logger)
        logger.info("程序正常退出")
    except Exception as e:
        # 特别处理logger可能未定义的情况
        try:
            logger.error(f"主程序执行失败: {e}\n{traceback.format_exc()}")
        except:
            print(f"主程序执行失败: {e}\n{traceback.format_exc()}")
