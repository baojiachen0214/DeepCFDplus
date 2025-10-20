import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    # 新版本PyTorch使用parametrizations
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    # 兼容旧版本PyTorch
    from torch.nn.utils import weight_norm
from Models.AutoEncoder import create_layer


def create_encoder_block(in_channels, out_channels, kernel_size, wn=True, bn=True,
                         activation=nn.ReLU, layers=2):
    """
    创建编码器块
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        wn: 是否使用权重归一化
        bn: 是否使用批归一化
        activation: 激活函数
        layers: 块内层数
        
    Returns:
        编码器块Sequential模型
    """
    encoder = []
    for i in range(layers):
        _in = out_channels
        _out = out_channels
        if i == 0:
            _in = in_channels
        encoder.append(create_layer(_in, _out, kernel_size, wn, bn, activation, nn.Conv2d))
    return nn.Sequential(*encoder)


def create_decoder_block(in_channels, out_channels, kernel_size, wn=True, bn=True,
                         activation=nn.ReLU, layers=2, final_layer=False):
    """
    创建解码器块
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        wn: 是否使用权重归一化
        bn: 是否使用批归一化
        activation: 激活函数
        layers: 块内层数
        final_layer: 是否为最后一层
        
    Returns:
        解码器块Sequential模型
    """
    decoder = []
    for i in range(layers):
        _in = in_channels
        _out = in_channels
        _bn = bn
        _activation = activation
        if i == 0:
            _in = in_channels * 2
        if i == layers - 1:
            _out = out_channels
            if final_layer:
                _bn = False
                _activation = None
        decoder.append(create_layer(_in, _out, kernel_size, wn, _bn, _activation, nn.ConvTranspose2d))
    return nn.Sequential(*decoder)


def create_encoder(in_channels, filters, kernel_size, wn=True, bn=True, activation=nn.ReLU, layers=2):
    """
    创建完整的编码器
    
    Args:
        in_channels: 输入通道数
        filters: 各层滤波器数量列表
        kernel_size: 卷积核大小
        wn: 是否使用权重归一化
        bn: 是否使用批归一化
        activation: 激活函数
        layers: 每个块的层数
        
    Returns:
        编码器Sequential模型
    """
    encoder = []
    for i in range(len(filters)):
        if i == 0:
            encoder_layer = create_encoder_block(in_channels, filters[i], kernel_size, wn, bn, activation, layers)
        else:
            encoder_layer = create_encoder_block(filters[i-1], filters[i], kernel_size, wn, bn, activation, layers)
        encoder = encoder + [encoder_layer]
    return nn.Sequential(*encoder)


def create_decoder(out_channels, filters, kernel_size, wn=True, bn=True, activation=nn.ReLU, layers=2):
    """
    创建完整的解码器
    
    Args:
        out_channels: 输出通道数
        filters: 各层滤波器数量列表
        kernel_size: 卷积核大小
        wn: 是否使用权重归一化
        bn: 是否使用批归一化
        activation: 激活函数
        layers: 每个块的层数
        
    Returns:
        解码器Sequential模型
    """
    decoder = []
    for i in range(len(filters)):
        if i == 0:
            decoder_layer = create_decoder_block(filters[i], out_channels, kernel_size, wn, bn, activation, layers, final_layer=True)
        else:
            decoder_layer = create_decoder_block(filters[i], filters[i-1], kernel_size, wn, bn, activation, layers, final_layer=False)
        decoder = [decoder_layer] + decoder
    return nn.Sequential(*decoder)


class UNet(nn.Module):
    """
    标准UNet模型
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, filters=[16, 32, 64], layers=2,
                 weight_norm=True, batch_norm=True, activation=nn.ReLU, final_activation=None):
        """
        初始化UNet模型
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            filters: 编码器各层滤波器数量
            layers: 每个编码器/解码器块的层数
            weight_norm: 是否使用权重归一化
            batch_norm: 是否使用批归一化
            activation: 激活函数
            final_activation: 最终激活函数
        """
        super().__init__()
        assert len(filters) > 0
        self.final_activation = final_activation
        self.encoder = create_encoder(in_channels, filters, kernel_size, weight_norm, batch_norm, activation, layers)
        self.decoder = create_decoder(out_channels, filters, kernel_size, weight_norm, batch_norm, activation, layers)

    def encode(self, x):
        """
        编码过程
        
        Args:
            x: 输入张量
            
        Returns:
            编码结果、特征张量、索引和尺寸信息
        """
        tensors = []
        indices = []
        sizes = []
        for encoder in self.encoder:
            x = encoder(x)
            sizes.append(x.size())
            tensors.append(x)
            x, ind = F.max_pool2d(x, 2, 2, return_indices=True)
            indices.append(ind)
        return x, tensors, indices, sizes

    def decode(self, x, tensors, indices, sizes):
        """
        解码过程
        
        Args:
            x: 编码结果
            tensors: 特征张量列表
            indices: 最大池化索引列表
            sizes: 尺寸信息列表
            
        Returns:
            解码结果
        """
        for decoder in self.decoder:
            tensor = tensors.pop()
            size = sizes.pop()
            ind = indices.pop()
            x = F.max_unpool2d(x, ind, 2, 2, output_size=size)
            x = torch.cat([tensor, x], dim=1)
            x = decoder(x)
        return x

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            模型输出
        """
        x, tensors, indices, sizes = self.encode(x)
        x = self.decode(x, tensors, indices, sizes)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x