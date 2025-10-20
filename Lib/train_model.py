import copy
import torch
import time
import os


def generate_metrics_list(metrics_def):
    """
    根据指标定义生成指标列表
    
    Args:
        metrics_def: 指标定义字典
        
    Returns:
        包含各指标名称和空列表的字典
    """
    metrics_list = {}
    for name in metrics_def.keys():
        metrics_list[name] = []
    return metrics_list


def epoch(scope, loader, on_batch=None, training=False):
    """
    执行一个完整的训练/验证周期
    
    Args:
        scope: 包含模型、优化器等的上下文字典
        loader: 数据加载器
        on_batch: 每个批次执行的回调函数
        training: 是否为训练模式
        
    Returns:
        总损失值和各项指标
    """
    model = scope["model"]
    optimizer = scope["optimizer"]
    loss_func = scope["loss_func"]
    metrics_def = scope["metrics_def"]
    scope = copy.copy(scope)
    scope["loader"] = loader

    metrics_list = generate_metrics_list(metrics_def)
    total_loss = 0

    # 设置模型模式
    if training:
        model.train()
    else:
        model.eval()

    for tensors in loader:
        # 处理批次数据
        if "process_batch" in scope and scope["process_batch"] is not None:
            tensors = scope["process_batch"](tensors)
        if "device" in scope and scope["device"] is not None:
            tensors = [tensor.to(scope["device"]) for tensor in tensors]

        # 前向传播和损失计算
        loss, output = loss_func(model, tensors)

        # 训练模式下进行反向传播和参数更新
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        scope["batch"] = tensors
        scope["loss"] = loss
        scope["output"] = output
        scope["batch_metrics"] = {}

        # 计算各项指标
        for name, metric in metrics_def.items():
            value = metric["on_batch"](scope)
            scope["batch_metrics"][name] = value
            metrics_list[name].append(value)

        if on_batch is not None:
            on_batch(scope)

    scope["metrics_list"] = metrics_list
    metrics = {}

    # 计算周期平均指标
    for name in metrics_def.keys():
        scope["list"] = scope["metrics_list"][name]
        metrics[name] = metrics_def[name]["on_epoch"](scope)

    return total_loss, metrics


def save_checkpoint(scope, filepath, checkpoint_type="epoch"):
    """
    保存训练检查点
    
    Args:
        scope: 包含训练状态的上下文字典
        filepath: 检查点文件保存路径
        checkpoint_type: 检查点类型（当前或最佳）
    """
    checkpoint = {
        'epoch': scope.get('epoch', 0),
        'model_state_dict': scope['model'].state_dict(),
        'optimizer_state_dict': scope['optimizer'].state_dict(),
        'best_val_loss': scope.get('best_val_loss', float('inf')),
        'train_loss_curve': scope.get('train_loss_curve', []),
        'val_loss_curve': scope.get('val_loss_curve', []),
    }
    torch.save(checkpoint, filepath)
    print(f"检查点 ({checkpoint_type}) 已保存到 {filepath}")


def load_checkpoint(filepath, model, optimizer):
    """
    从检查点加载状态
    
    Args:
        filepath: 检查点文件路径
        model: 模型对象
        optimizer: 优化器对象
        
    Returns:
        检查点数据字典
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"检查点已从 {filepath} 加载")
    return checkpoint


def train(scope, train_dataset, val_dataset, patience=10, batch_size=256, print_function=print, eval_model=None,
          on_train_batch=None, on_val_batch=None, on_train_epoch=None, on_val_epoch=None, after_epoch=None):
    """
    执行模型训练
    
    Args:
        scope: 包含训练配置的上下文字典
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        patience: 早停耐心值
        batch_size: 批次大小
        print_function: 打印函数
        eval_model: 模型评估函数
        on_train_batch: 训练批次回调
        on_val_batch: 验证批次回调
        on_train_epoch: 训练周期回调
        on_val_epoch: 验证周期回调
        after_epoch: 周期后回调
        
    Returns:
        最佳模型和相关指标
    """
    epochs = scope["epochs"]
    model = scope["model"]
    metrics_def = scope["metrics_def"]
    scope = copy.copy(scope)

    scope["best_train_metric"] = None
    scope["best_train_loss"] = float("inf")
    scope["best_val_metrics"] = None
    scope["best_val_loss"] = float("inf")
    scope["best_model"] = None

    # 获取保存频率参数
    save_every_n_epochs = scope.get("save_every_n_epochs", 0)
    save_best_every_n_epochs = scope.get("save_best_every_n_epochs", 1)

    # 使用传入的保存目录
    save_dir = scope.get("save_dir", "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # 恢复训练逻辑
    start_epoch = 1
    resume_path = scope.get("resume_path", None)
    if resume_path is not None and os.path.exists(resume_path):
        checkpoint = load_checkpoint(resume_path, model, scope["optimizer"])
        start_epoch = checkpoint['epoch'] + 1
        scope["best_val_loss"] = checkpoint['best_val_loss']
        scope["train_loss_curve"] = checkpoint.get('train_loss_curve', [])
        scope["val_loss_curve"] = checkpoint.get('val_loss_curve', [])
        print_function(f"从第 {start_epoch} 轮恢复训练，加载自 {resume_path}")

    # 初始化损失曲线
    if 'train_loss_curve' not in scope:
        scope["train_loss_curve"] = []
    if 'val_loss_curve' not in scope:
        scope["val_loss_curve"] = []

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    skips = 0

    for epoch_id in range(start_epoch, epochs + 1):
        scope["epoch"] = epoch_id
        epoch_start_time = time.time()

        # 训练阶段
        scope["dataset"] = train_dataset
        train_start_time = time.time()
        train_loss, train_metrics = epoch(scope, train_loader, on_train_batch, training=True)
        train_end_time = time.time()
        scope["train_loss"] = train_loss
        scope["train_metrics"] = train_metrics
        if on_train_epoch is not None:
            on_train_epoch(scope)
        del scope["dataset"]

        # 验证阶段
        scope["dataset"] = val_dataset
        with torch.no_grad():
            val_start_time = time.time()
            val_loss, val_metrics = epoch(scope, val_loader, on_val_batch, training=False)
            val_end_time = time.time()
        scope["val_loss"] = val_loss
        scope["val_metrics"] = val_metrics
        if on_val_epoch is not None:
            on_val_epoch(scope)
        del scope["dataset"]

        # 日志记录
        epoch_end_time = time.time()
        lr = scope["optimizer"].param_groups[0]["lr"]

        # 更新损失曲线
        scope["train_loss_curve"].append(train_loss)
        scope["val_loss_curve"].append(val_loss)

        log_str = (
            f"轮次 [{epoch_id}/{epochs}] | "
            f"训练损失: {train_loss:.6f} | "
            f"验证损失: {val_loss:.6f} | "
            f"学习率: {lr:.2e} | "
            f"训练时间: {train_end_time - train_start_time:.2f}s | "
            f"验证时间: {val_end_time - val_start_time:.2f}s | "
            f"总时间: {epoch_end_time - epoch_start_time:.2f}s"
        )
        print_function(log_str)

        # 打印详细指标并传递给after_epoch回调
        for name in metrics_def.keys():
            # 将每个指标的训练和验证值添加到scope中，供after_epoch使用
            scope[f"train_{name}"] = train_metrics[name]
            scope[f"val_{name}"] = val_metrics[name]
            log_str = (f"  训练 {metrics_def[name]['name']}: {train_metrics[name]:.6f} "
                       f"| 验证 {metrics_def[name]['name']}: {val_metrics[name]:.6f}")
            print_function(log_str)

        # 模型选择和保存
        is_best = None
        if eval_model is not None:
            is_best = eval_model(scope)
        if is_best is None:
            is_best = val_loss < scope["best_val_loss"]

        # 根据频率保存当前模型
        if save_every_n_epochs > 0 and epoch_id % save_every_n_epochs == 0:
            current_checkpoint_path = os.path.join(save_dir, f"current_model_epoch_{epoch_id}.pth")
            save_checkpoint(scope, current_checkpoint_path, checkpoint_type="current")

        # 根据频率和性能保存最佳模型
        if is_best and (save_best_every_n_epochs > 0 and epoch_id % save_best_every_n_epochs == 0):
            scope["best_train_metric"] = train_metrics
            scope["best_train_loss"] = train_loss
            scope["best_val_metrics"] = val_metrics
            scope["best_val_loss"] = val_loss
            scope["best_model"] = copy.deepcopy(model)
            best_checkpoint_path = os.path.join(save_dir, f"best_model_checkpoint.pth")
            save_checkpoint(scope, best_checkpoint_path, checkpoint_type="best")
            print_function(">>> 已保存新的最佳模型! <<<")
            skips = 0
        elif is_best:
            # 如果是最佳模型但不满足保存频率，则更新最佳指标但不保存
            scope["best_train_metric"] = train_metrics
            scope["best_train_loss"] = train_loss
            scope["best_val_metrics"] = val_metrics
            scope["best_val_loss"] = val_loss
            scope["best_model"] = copy.deepcopy(model)
            print_function(">>> 新的最佳模型 (由于频率设置未保存) <<<")
            skips = 0
        else:
            skips += 1
            if skips >= patience:
                print_function(f"在 {patience} 轮无改善后触发早停机制。")
                break

        if after_epoch is not None:
            after_epoch(scope)

    return scope["best_model"], scope["best_train_metric"], scope["best_train_loss"], \
        scope["best_val_metrics"], scope["best_val_loss"]


def train_model(model, loss_func, train_dataset, val_dataset, optimizer, process_batch=None, eval_model=None,
                on_train_batch=None, on_val_batch=None, on_train_epoch=None, on_val_epoch=None, after_epoch=None,
                epochs=100, batch_size=256, patience=10, device=None, resume_path=None,
                save_every_n_epochs=0, save_best_every_n_epochs=1, save_dir="./checkpoints",
                **kwargs):
    """
    模型训练入口函数
    
    Args:
        model: 待训练的模型
        loss_func: 损失函数
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        optimizer: 优化器
        process_batch: 批次处理函数
        eval_model: 模型评估函数
        on_train_batch: 训练批次回调
        on_val_batch: 验证批次回调
        on_train_epoch: 训练周期回调
        on_val_epoch: 验证周期回调
        after_epoch: 周期后回调
        epochs: 训练轮数
        batch_size: 批次大小
        patience: 早停耐心值
        device: 训练设备
        resume_path: 恢复训练路径
        save_every_n_epochs: 每N轮保存当前模型
        save_best_every_n_epochs: 每N轮检查并保存最佳模型
        save_dir: 模型保存目录
        **kwargs: 其他参数
        
    Returns:
        训练结果
    """
    if device is not None:
        model = model.to(device)
    scope = {"model": model, "loss_func": loss_func, "train_dataset": train_dataset, "val_dataset": val_dataset,
             "optimizer": optimizer, "process_batch": process_batch, "epochs": epochs, "batch_size": batch_size,
             "device": device, "resume_path": resume_path, "save_every_n_epochs": save_every_n_epochs,
             "save_best_every_n_epochs": save_best_every_n_epochs, "save_dir": save_dir}

    # 处理指标定义
    metrics_def = {}
    names = []
    for key in kwargs.keys():
        parts = key.split("_")
        if len(parts) == 3 and parts[0] == "m":
            if parts[1] not in names:
                names.append(parts[1])
    for name in names:
        if ("m_" + name + "_name" in kwargs
                and "m_" + name + "_on_batch" in kwargs
                and "m_" + name + "_on_epoch" in kwargs):
            metrics_def[name] = {
                "name": kwargs["m_" + name + "_name"],
                "on_batch": kwargs["m_" + name + "_on_batch"],
                "on_epoch": kwargs["m_" + name + "_on_epoch"],
            }
        else:
            print("警告: " + name + " 指标不完整!")
    scope["metrics_def"] = metrics_def

    return train(scope, train_dataset, val_dataset, eval_model=eval_model, on_train_batch=on_train_batch,
                 on_val_batch=on_val_batch, on_train_epoch=on_train_epoch, on_val_epoch=on_val_epoch,
                 after_epoch=after_epoch,
                 batch_size=batch_size, patience=patience)
