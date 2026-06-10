"""
训练线程 - 在后台线程中执行模型训练，避免阻塞UI
支持 FeatureBank（特征库）和 SuperSimpleNet（神经网络）两种模式
"""
from PyQt5.QtCore import QThread, pyqtSignal
from torch.utils.data import DataLoader

from core.trainer import Trainer
from core.fb_trainer import FBTrainer
from core.dataset import TrainDataset
from config.config import TrainingConfig
from utils.logger import get_logger

logger = get_logger("training_thread")


class TrainingThread(QThread):
    """
    后台训练线程

    自动选择训练方式:
      - method="feature_bank": 构建特征库（默认，推荐，AUROC≈1.0）
      - method="super_simple_net": 传统神经网络训练

    Signals:
        epoch_finished: (epoch, total_epochs, loss)
        loss_updated: (losses_list)
        status_updated: (status_text)
        training_finished: (success, message)
        error_occurred: (error_message)
        training_history: (history_text)
    """

    epoch_finished = pyqtSignal(int, int, float)
    loss_updated = pyqtSignal(list)
    status_updated = pyqtSignal(str)
    training_finished = pyqtSignal(bool, str)
    error_occurred = pyqtSignal(str)
    training_history = pyqtSignal(str)

    # FeatureBank 特有的完成信号
    fb_training_finished = pyqtSignal(bool, str)

    def __init__(self, config: TrainingConfig, parent=None):
        super().__init__(parent)
        self.cfg = config
        self.trainer = None       # Trainer 或 FBTrainer
        self.train_loader = None
        self._model_name = ""
        self._method = "feature_bank"  # 默认使用 FeatureBank

    def set_model_name(self, name: str):
        """设置自定义模型名称"""
        self._model_name = name.strip()

    def set_method(self, method: str):
        """设置训练方法: 'feature_bank' 或 'super_simple_net'"""
        if method in ("feature_bank", "super_simple_net"):
            self._method = method
            logger.info(f"训练方法已设置: {method}")

    def setup(self) -> bool:
        """初始化训练器和数据加载器"""
        try:
            # 验证数据路径
            if not self.cfg.data_path:
                raise ValueError("请先选择训练数据文件夹")

            # 创建数据集
            dataset = TrainDataset(
                data_path=self.cfg.data_path,
                image_size=self.cfg.image_size,
                use_dataset_norm=True,
            )

            if len(dataset) == 0:
                raise ValueError("训练数据集中没有图像文件")

            self.train_loader = TrainDataset.get_dataloader(
                dataset,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
            )

            # 根据方法创建训练器
            if self._method == "feature_bank":
                self.trainer = FBTrainer(self.cfg)
                self.status_updated.emit(
                    f"FeatureBank 模式 | 样本数: {len(dataset)} | "
                    f"骨干网络: {self.cfg.backbone} | "
                    f"无需梯度训练，只需构建特征库"
                )
            else:
                self.trainer = Trainer(self.cfg)
                self.trainer.build_model()
                self.status_updated.emit(
                    f"SuperSimpleNet 模式 | 样本数: {len(dataset)} | "
                    f"Batch大小: {self.cfg.batch_size} | "
                    f"设备: {self.trainer.device}"
                )

            # 设置模型名称
            if self._model_name:
                self.trainer.set_model_name(self._model_name)

            # 注册回调
            self.trainer.on_epoch_end = lambda e, t, l: self.epoch_finished.emit(e, t, l)
            self.trainer.on_loss_updated = lambda losses: self.loss_updated.emit(losses)
            self.trainer.on_status_changed = lambda msg: self.status_updated.emit(msg)

            return True

        except Exception as e:
            error_msg = f"初始化失败: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False

    def run(self):
        """线程主函数 - 执行训练"""
        try:
            if self.trainer is None:
                if not self.setup():
                    return

            self.status_updated.emit("训练开始...")
            self.trainer.train(self.train_loader)

            # 保存模型
            if hasattr(self.trainer, 'save_model'):
                # FeatureBank 模式：保存模型文件
                success, msg, path = self.trainer.save_model()
                model_name = self.trainer.model_name or "FeatureBank检测模型"
                total_time = self.trainer.total_time

                if success:
                    msg = (
                        f"特征库构建完成! 模型: {model_name} | "
                        f"样本数: {len(self.trainer._ok_image_paths)} | "
                        f"阈值: {self.trainer._suggested_threshold:.4f} | "
                        f"耗时: {total_time:.1f}s"
                    )
                self.training_finished.emit(success, msg)

                # 发射历史信息
                history_text = self.trainer.get_operation_history()
                self.training_history.emit(history_text)
            else:
                # SuperSimpleNet 模式：原有逻辑
                total_time = self.trainer.total_time
                best_loss = self.trainer.best_loss
                model_name = self.trainer.model_name or "未命名"
                msg = (
                    f"训练完成! 模型名称: {model_name} | "
                    f"总耗时: {total_time:.1f}s | "
                    f"最佳Loss: {best_loss:.6f}"
                )
                history_text = self.trainer.op_history.get_formatted_history()
                self.training_history.emit(history_text)
                self.training_finished.emit(True, msg)

        except Exception as e:
            error_msg = f"训练异常: {str(e)}"
            logger.error(error_msg)
            self.training_finished.emit(False, error_msg)

    def pause(self):
        """暂停训练"""
        if self.trainer:
            self.trainer.pause()

    def resume(self):
        """继续训练"""
        if self.trainer:
            self.trainer.resume()

    def stop(self):
        """停止训练"""
        if self.trainer:
            self.trainer.stop()
