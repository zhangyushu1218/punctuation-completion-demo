"""
训练日志模块
提供训练过程中的日志记录和进度追踪功能
"""
import logging
import os
from datetime import datetime
from transformers import TrainerCallback


def setup_training_logger(log_dir="./logs"):
    """
    设置训练日志记录器
    
    Args:
        log_dir: 日志文件保存目录
        
    Returns:
        tuple: (logger, log_file_path)
    """
    # 创建logs目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # 配置logger
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    logger.handlers.clear()
    
    # 文件handler - 保存详细日志
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                   datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # 控制台handler - 显示关键信息
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger, log_file


class TrainingLogCallback(TrainerCallback):
    """
    自定义回调，用于记录训练过程中的详细信息
    
    在每次 logging_steps 时自动记录：
    - Step: 当前训练步数
    - Epoch: 当前训练轮次
    - Loss: 批次损失值
    - Learning Rate: 当前学习率
    - Grad Norm: 梯度范数（如果可用）
    """
    
    def __init__(self, logger):
        """
        初始化训练日志回调
        
        Args:
            logger: logging.Logger 实例
        """
        self.logger = logger
        self.training_logs = []  # 存储所有训练日志
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        在每次 logging_steps 时触发
        
        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
            logs: 包含训练指标的字典
        """
        if logs is not None:
            # 提取关键信息
            log_entry = {
                "step": state.global_step,
                "epoch": round(logs.get("epoch", 0), 4),
                "loss": round(logs.get("loss", 0), 4),
                "learning_rate": logs.get("learning_rate", 0),
            }
            
            # 如果有梯度范数则记录
            if "grad_norm" in logs:
                log_entry["grad_norm"] = round(logs["grad_norm"], 4)
            
            # 保存到列表
            self.training_logs.append(log_entry)
            
            # 格式化输出到日志文件
            log_parts = [
                f"Step: {log_entry['step']}",
                f"Epoch: {log_entry['epoch']}",
                f"Loss: {log_entry['loss']}",
                f"LR: {log_entry['learning_rate']:.3e}"
            ]
            
            if "grad_norm" in log_entry:
                log_parts.append(f"Grad Norm: {log_entry['grad_norm']}")
            
            log_message = " | ".join(log_parts)
            self.logger.info(f"  [训练进度] {log_message}")
    
    def get_training_logs(self):
        """
        获取所有记录的训练日志
        
        Returns:
            list: 训练日志列表
        """
        return self.training_logs
    
    def get_logs_summary(self):
        """
        获取训练日志摘要
        
        Returns:
            dict: 包含日志统计信息的字典
        """
        if not self.training_logs:
            return {
                "total_steps": 0,
                "logs_count": 0,
                "logs_sample": []
            }
        
        return {
            "total_steps": len(self.training_logs),
            "logs_sample": self.training_logs[:10],  # 前10条作为示例
            "logs_count": len(self.training_logs)
        }
