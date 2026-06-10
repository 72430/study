"""
FeatureBank 训练器 - 替代 Trainer 作为 UI 的训练后端

"训练" = 构建特征库（无需梯度下降）
保存格式兼容 model_deploy.pth，可被 Inferencer 加载
"""
import os
import time
from datetime import datetime
import torch
import numpy as np
from glob import glob
from pathlib import Path
from typing import Callable, Optional

from core.feature_bank import FeatureBankDetector
from core.dataset import TrainDataset
from core.model_manager import ModelManager
from config.config import TrainingConfig
from utils.logger import get_logger

logger = get_logger("fb_trainer")


class FBTrainer:
    """
    FeatureBank 训练器

    实现与 Trainer 相同的对外接口（train/save/stop等），
    但内部使用特征库构建替代梯度下降训练。
    """

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型管理器（用于注册模型到注册表）
        self.model_manager = ModelManager(config.checkpoint_dir)

        # 检测器（训练完成后持有）
        self.detector: Optional[FeatureBankDetector] = None

        # 训练状态（兼容原有接口）
        self.current_epoch = 0
        self.total_epochs = 1        # 特征库只需1轮
        self.train_losses = [0.0]    # 无实际loss
        self.epoch_times = []
        self.best_loss = 0.0
        self.start_time = 0.0
        self.total_time = 0.0
        self.model_name = ""

        # 控制标志
        self._is_training = False
        self._is_stopped = False

        # 回调函数
        self.on_epoch_end: Optional[Callable] = None
        self.on_loss_updated: Optional[Callable] = None
        self.on_status_changed: Optional[Callable] = None

        # 特征库元数据
        self._ok_image_paths = []
        self._feature_bank = None
        self._ok_score_mean = 0.0
        self._ok_score_std = 0.0
        self._suggested_threshold = 0.05

    def set_model_name(self, name: str):
        """设置模型名称"""
        self.model_name = name.strip()
        logger.info(f"模型名称已设置: {self.model_name}")

    def build_model(self):
        """构建检测器（兼容 Trainer 接口）"""
        self.detector = FeatureBankDetector(
            backbone=self.cfg.backbone,
            n_neighbors=3,
            device="auto",
        )
        logger.info(f"FeatureBank 检测器已创建 | backbone={self.cfg.backbone}")

    def train(self, train_loader):
        """
        构建特征库（替代传统训练）

        Args:
            train_loader: 数据加载器（仅用于获取图像路径）
        """
        if self.detector is None:
            self.build_model()

        self._is_training = True
        self.start_time = time.time()

        # 从 dataloader 的 dataset 获取图像路径
        dataset = train_loader.dataset
        ok_paths = dataset.image_paths if hasattr(dataset, 'image_paths') else []

        if not ok_paths:
            logger.error("无OK样本图像路径")
            self._is_training = False
            return

        self._ok_image_paths = ok_paths

        # 通知开始
        if self.on_status_changed:
            self.on_status_changed(
                f"构建特征库: {len(ok_paths)} 张OK样本, backbone={self.cfg.backbone}"
            )

        # 提取特征并构建特征库
        try:
            stats = self.detector.fit(ok_paths)
            self._feature_bank = self.detector.feature_bank
            self._ok_score_mean = stats.get("ok_score_mean", 0.0)
            self._ok_score_std = stats.get("ok_score_std", 0.0)
            self._suggested_threshold = stats.get("suggested_threshold", 0.05)
        except Exception as e:
            logger.error(f"特征库构建失败: {e}")
            self._is_training = False
            raise

        self.total_time = time.time() - self.start_time
        self._is_training = False

        # 通知完成
        if self.on_epoch_end:
            self.on_epoch_end(1, 1, 0.0)
        if self.on_status_changed:
            self.on_status_changed(
                f"特征库构建完成 | {len(ok_paths)}张OK样本 → "
                f"shape={self._feature_bank.shape} | "
                f"阈值建议={self._suggested_threshold:.4f} | "
                f"耗时={self.total_time:.1f}s"
            )

        logger.info(
            f"特征库构建完成! 耗时: {self.total_time:.1f}s, "
            f"样本数: {len(ok_paths)}"
        )

    def save_model(self, save_path: str = None):
        """
        保存特征库模型

        Args:
            save_path: 保存路径，None时使用默认路径

        Returns:
            (success, message, path)
        """
        if self._feature_bank is None:
            return False, "特征库为空，请先训练", ""

        if save_path is None:
            save_path = os.path.join(
                self.cfg.checkpoint_dir, "model_deploy.pth"
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存完整模型（兼容 Inferencer 加载）
        save_data = {
            "model_state_dict": {},  # 无权重，占位
            "system_version": "1.0.0",
            "method": "feature_bank",
            "config": self.cfg.to_dict(),
            "name": self.model_name or "FeatureBank检测模型",
            "reference_embedding": self._feature_bank,
            "ok_score_mean": self._ok_score_mean,
            "ok_score_std": self._ok_score_std,
            "suggested_threshold": self._suggested_threshold,
            "ok_sample_count": len(self._ok_image_paths),
            "backbone": self.cfg.backbone,
            "embedding_dim": (
                self._feature_bank.shape[1]
                if self._feature_bank is not None else 0
            ),
        }
        torch.save(save_data, save_path)
        logger.info(f"模型已保存: {save_path} ({os.path.getsize(save_path)/1024/1024:.2f}MB)")

        # 注册到模型管理器，使其出现在模型列表中
        try:
            metadata = {
                "name": self.model_name or "FeatureBank检测模型",
                "backbone": self.cfg.backbone,
                "method": "feature_bank",
                "embedding_dim": save_data["embedding_dim"],
                "ok_sample_count": save_data["ok_sample_count"],
                "ok_score_mean": float(self._ok_score_mean),
                "ok_score_std": float(self._ok_score_std),
                "suggested_threshold": float(self._suggested_threshold),
                "system_version": "1.0.0",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.model_manager.meta_manager.register_model(save_path, metadata)
            logger.info("模型已注册到模型管理器")
        except Exception as e:
            logger.warning(f"模型注册失败（不影响使用）: {e}")

        return True, f"模型已保存: {save_path}", save_path

    # ========== 控制接口（兼容） ==========

    def pause(self):
        """暂停（无操作，兼容接口）"""
        logger.debug("FeatureBank 训练不支持暂停")

    def resume(self):
        """继续（无操作，兼容接口）"""
        logger.debug("FeatureBank 训练不支持继续")

    def stop(self):
        """停止"""
        self._is_stopped = True
        logger.info("特征库构建已终止")

    def get_operation_history(self) -> str:
        """获取操作历史（简化版）"""
        return (
            f"FeatureBank 模型: {self.model_name}\n"
            f"骨干网络: {self.cfg.backbone}\n"
            f"OK样本数: {len(self._ok_image_paths)}\n"
            f"特征维度: {self._feature_bank.shape[1] if self._feature_bank is not None else 0}\n"
            f"OK得分均值: {self._ok_score_mean:.4f}\n"
            f"建议阈值: {self._suggested_threshold:.4f}\n"
            f"耗时: {self.total_time:.1f}s"
        )
