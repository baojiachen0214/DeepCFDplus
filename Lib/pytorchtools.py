import numpy as np
import torch


class EarlyStopping:
    """早停机制：如果验证损失在给定的耐心值后没有改善，则停止训练"""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): 上次验证损失改善后等待的轮数
                            默认值: 7
            verbose (bool): 如果为True，则在每次验证损失改善时打印消息
                            默认值: False
            delta (float): 监控量的最小变化，以符合改善条件
                            默认值: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'早停计数器: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """验证损失减少时保存模型"""
        if self.verbose:
            print(f'验证损失减少 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 正在保存模型 ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
