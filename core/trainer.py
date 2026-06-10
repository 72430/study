"""
训练器模块 - 实现模型训练、暂停/继续/终止控制、损失曲线可视化数据生成
集成模型管理器，支持自定义模型命名保存
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Callable, Optional

from model.sup_simple_net import SuperSimpleNet
from config.config import TrainingConfig
from core.model_manager import ModelManager, OperationHistory
from utils.logger import get_logger

logger = get_logger("trainer")


class Trainer:
    """
    模型训练器

    支持:
      - 训练控制（开始/暂停/继续/终止）
      - 自定义模型命名保存（含完整元数据）
      - 进度回调、损失记录、自动保存最佳模型
    """

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.device = self._resolve_device(config.device)

        # 模型管理器
        self.model_manager = ModelManager(config.checkpoint_dir)
        self.op_history = OperationHistory()

        # 模型
        self.model: Optional[SuperSimpleNet] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.criterion = nn.MSELoss()  # 使用MSE Loss简单有效

        # 训练状态
        self.current_epoch = 0
        self.total_epochs = config.epochs
        self.train_losses = []          # 记录每个epoch的loss
        self.epoch_times = []           # 记录每个epoch耗时
        self.best_loss = float("inf")
        self.start_time = 0.0
        self.total_time = 0.0

        # 自定义模型名称（训练前由用户设置）
        self.model_name: str = ""

        # 控制标志
        self._is_paused = False
        self._is_stopped = False
        self._is_training = False

        # 回调函数（用于UI更新）
        self.on_epoch_end: Optional[Callable] = None
        self.on_loss_updated: Optional[Callable] = None
        self.on_status_changed: Optional[Callable] = None

    def _resolve_device(self, device_str: str) -> torch.device:
        """解析设备字符串"""
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def set_model_name(self, name: str):
        """
        设置自定义模型名称

        Args:
            name: 模型名称（训练完成后将以此名称保存）
        """
        self.model_name = name.strip()
        logger.info(f"模型名称已设置: {self.model_name}")

    def validate_model_name(self, name: str) -> tuple:
        """
        验证模型名称（委托给ModelManager）

        Args:
            name: 模型名称

        Returns:
            (is_valid, message)
        """
        return self.model_manager.validate_name(name)

    def build_model(self):
        """构建模型并移动到目标设备"""
        logger.info(f"构建模型: backbone={self.cfg.backbone}, "
                     f"embedding_dim={self.cfg.embedding_dim}")
        self.model = SuperSimpleNet(
            backbone=self.cfg.backbone,
            embedding_dim=self.cfg.embedding_dim,
            pretrained=self.cfg.pretrained,
        ).to(self.device)

        # 只优化映射网络的参数
        trainable_params = filter(
            lambda p: p.requires_grad, self.model.parameters()
        )
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        # 学习率调度器
        if self.cfg.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.epochs
            )
        elif self.cfg.lr_scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.lr_step_size,
                gamma=self.cfg.lr_gamma,
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=5
            )

        logger.info(f"模型已创建，设备: {self.device}")

    def train(self, train_loader: DataLoader):
        """
        执行模型训练

        Args:
            train_loader: 训练数据加载器
        """
        if self.model is None:
            self.build_model()

        self._is_training = True
        self._is_paused = False
        self._is_stopped = False
        self.model.set_train_mode()
        self._last_train_loader = train_loader  # 保存引用，用于后续计算参考嵌入

        self.start_time = time.time()
        logger.info(f"训练开始，共 {self.total_epochs} 个epoch")
        self.op_history.add_record("模型训练", "开始", f"总轮次: {self.total_epochs}")

        for epoch in range(self.current_epoch, self.total_epochs):
            # 检查暂停/终止状态
            while self._is_paused:
                time.sleep(0.1)
                if self._is_stopped:
                    break
            if self._is_stopped:
                logger.info("训练被用户终止")
                self.op_history.add_record("模型训练", "终止", f"完成轮次: {epoch}")
                break

            epoch_start = time.time()
            epoch_loss = self._train_one_epoch(train_loader, epoch)
            epoch_time = time.time() - epoch_start

            # 记录指标
            self.train_losses.append(epoch_loss)
            self.epoch_times.append(epoch_time)
            self.current_epoch = epoch + 1

            # 更新学习率
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(epoch_loss)
            else:
                self.scheduler.step()

            # 保存最佳模型（使用model_manager按名称保存）
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self._save_named_model(is_best=True)

            # 回调通知
            if self.on_epoch_end:
                self.on_epoch_end(epoch + 1, self.total_epochs, epoch_loss)
            if self.on_loss_updated:
                self.on_loss_updated(self.train_losses)

            # 日志
            logger.info(
                f"Epoch [{epoch + 1}/{self.total_epochs}] | "
                f"Loss: {epoch_loss:.6f} | "
                f"Time: {epoch_time:.2f}s | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

        self.total_time = time.time() - self.start_time
        self._is_training = False

        # 训练完成时保存最终模型
        if not self._is_stopped:
            self._save_named_model(is_best=False)
            self.op_history.add_record(
                "模型训练", "成功",
                f"名称: {self.model_name or '默认'} | "
                f"最佳Loss: {self.best_loss:.6f} | "
                f"耗时: {self.total_time:.1f}s"
            )
            logger.info(
                f"训练完成! 总耗时: {self.total_time:.2f}s, "
                f"最佳Loss: {self.best_loss:.6f}"
            )
        else:
            self.op_history.add_record(
                "模型训练", "已终止",
                f"最佳Loss: {self.best_loss:.6f}"
            )

    def _train_one_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """训练一个epoch"""
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(self.device)

            # 前向传播
            embeddings = self.model(images)

            # 对于无监督学习，我们希望正常样本的嵌入向量尽可能紧凑
            center = embeddings.mean(dim=0, keepdim=True).detach()
            loss = self.criterion(embeddings, center.expand_as(embeddings))

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # 每10个batch输出一次进度
            if (batch_idx + 1) % 10 == 0 and self.on_status_changed:
                progress = (epoch * num_batches + batch_idx + 1) / \
                           (self.total_epochs * num_batches) * 100
                self.on_status_changed(
                    f"训练中... Epoch {epoch + 1}/{self.total_epochs}, "
                    f"Batch {batch_idx + 1}/{num_batches}, "
                    f"Loss: {loss.item():.6f}"
                )

        return total_loss / num_batches

    def _save_named_model(self, is_best: bool = False):
        """
        使用自定义名称保存模型（通过ModelManager）

        Args:
            is_best: 是否为最佳模型
        """
        model_name = self.model_name.strip() if self.model_name.strip() else (
            f"best_model_{is_best}" if is_best else "latest_model"
        )

        success, msg, path = self.model_manager.prepare_save(
            model_name=model_name,
            config=self.cfg,
            best_loss=self.best_loss,
            current_epoch=self.current_epoch,
            total_epochs=self.total_epochs,
            train_losses=self.train_losses,
            total_time=self.total_time,
            state_dict=self.model.mapping_net.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
        )

        if success and is_best:
            # 同时保存部署格式（含版本信息及参考嵌入）
            from core.model_manager import SYSTEM_VERSION

            # 计算参考嵌入并保存
            ref_embedding = None
            try:
                self.model.set_eval_mode()
                all_embs = []
                # 使用最后保存的数据加载器重新计算（如果有）
                if hasattr(self, '_last_train_loader') and self._last_train_loader is not None:
                    with torch.no_grad():
                        for imgs, _ in self._last_train_loader:
                            all_embs.append(self.model(imgs.to(self.device)).cpu())
                    if all_embs:
                        ref_embedding = torch.cat(all_embs, dim=0).mean(dim=0)
            except Exception:
                pass

            deploy_path = os.path.join(
                self.cfg.checkpoint_dir, "model_deploy.pth"
            )
            deploy_data = {
                "model_state_dict": self.model.mapping_net.state_dict(),
                "system_version": SYSTEM_VERSION,
                "config": self.cfg.to_dict(),
                "name": model_name,
            }
            if ref_embedding is not None:
                deploy_data["reference_embedding"] = ref_embedding
            torch.save(deploy_data, deploy_path)

    def _save_checkpoint(self, is_best: bool = False):
        """
        [保留] 原始保存方法 - 保持向下兼容
        新代码请使用 _save_named_model()
        """
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

        suffix = "best" if is_best else "latest"
        checkpoint = {
            "model_state_dict": self.model.mapping_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.cfg.to_dict(),
            "epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "train_losses": self.train_losses,
            "system_version": "1.0.0",
        }
        path = os.path.join(
            self.cfg.checkpoint_dir,
            f"sup_simple_net_{suffix}.pth",
        )
        torch.save(checkpoint, path)

    # ==================== 训练控制 ====================

    def pause(self):
        """暂停训练"""
        if self._is_training:
            self._is_paused = True
            logger.info("训练已暂停")
            self.op_history.add_record("训练控制", "暂停", f"Epoch: {self.current_epoch}")
            if self.on_status_changed:
                self.on_status_changed("训练已暂停")

    def resume(self):
        """继续训练"""
        if self._is_training:
            self._is_paused = False
            logger.info("训练已继续")
            self.op_history.add_record("训练控制", "继续", f"Epoch: {self.current_epoch}")
            if self.on_status_changed:
                self.on_status_changed("训练已继续")

    def stop(self):
        """终止训练"""
        self._is_stopped = True
        self._is_paused = False
        self._is_training = False
        logger.info("训练已终止")
        self.op_history.add_record("训练控制", "终止", f"已训练: {self.current_epoch}轮")
        if self.on_status_changed:
            self.on_status_changed("训练已终止")

    # ==================== 属性接口 ====================

    @property
    def is_training(self) -> bool:
        return self._is_training

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    def get_current_lr(self) -> float:
        """获取当前学习率"""
        if self.optimizer:
            return self.optimizer.param_groups[0]["lr"]
        return 0.0

    def load_checkpoint(self, path: str):
        """加载检查点"""
        if not os.path.exists(path):
            logger.warning(f"检查点文件不存在: {path}")
            return

        # 使用ModelManager进行完整性验证
        success, msg, checkpoint = self.model_manager.load_with_validation(
            path, map_location=str(self.device)
        )
        if not success:
            logger.error(f"加载检查点失败: {msg}")
            return

        if self.model is None:
            self.build_model()

        state_dict = checkpoint.get("model_state_dict")
        if state_dict:
            self.model.mapping_net.load_state_dict(state_dict)

        if checkpoint.get("optimizer_state_dict") and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.train_losses = checkpoint.get("train_losses", [])

        metadata = checkpoint.get("metadata", {})
        name_info = metadata.get("name", os.path.basename(path))
        logger.info(f"检查点已加载: {name_info} (Epoch {self.current_epoch})")
        self.op_history.add_record("加载检查点", "成功", f"文件: {os.path.basename(path)}")
