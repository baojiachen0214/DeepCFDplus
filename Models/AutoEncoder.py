import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    # 新版本PyTorch使用parametrizations
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    # 兼容旧版本PyTorch
    from torch.nn.utils import weight_norm


def create_layer(in_channels, out_channels, kernel_size, wn=True, bn=True,
                 activation=nn.ReLU, convolution=nn.Conv2d):
    """
    创建网络层
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        wn: 是否使用权重归一化
        bn: 是否使用批归一化
        activation: 激活函数
        convolution: 卷积类型（Conv2d或ConvTranspose2d）
        
    Returns:
        网络层Sequential模型
    """
    assert kernel_size % 2 == 1
    layer = []
    conv = convolution(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
    if wn:
        conv = weight_norm(conv)
    layer.append(conv)
    if activation is not None:
        layer.append(activation())
    if bn:
        layer.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layer)


class AutoEncoder(nn.Module):
    """
    自编码器模型
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, filters=[16, 32, 64],
                 weight_norm=True, batch_norm=True, activation=nn.ReLU, final_activation=None):
        """
        初始化自编码器
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            filters: 编码器各层滤波器数量
            weight_norm: 是否使用权重归一化
            batch_norm: 是否使用批归一化
            activation: 激活函数
            final_activation: 最终激活函数
        """
        super().__init__()
        assert len(filters) > 0
        encoder = []
        decoder = []
        for i in range(len(filters)):
            if i == 0:
                encoder_layer = create_layer(in_channels, filters[i], kernel_size, weight_norm, batch_norm, activation, nn.Conv2d)
                decoder_layer = create_layer(filters[i], out_channels, kernel_size, weight_norm, False, final_activation, nn.ConvTranspose2d)
            else:
                encoder_layer = create_layer(filters[i-1], filters[i], kernel_size, weight_norm, batch_norm, activation, nn.Conv2d)
                decoder_layer = create_layer(filters[i], filters[i-1], kernel_size, weight_norm, batch_norm, activation, nn.ConvTranspose2d)
            encoder = encoder + [encoder_layer]
            decoder = [decoder_layer] + decoder
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            编码-解码后的输出张量
        """
        return self.decoder(self.encoder(x))