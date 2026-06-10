"""
数据集加载模块 - 支持OK样本的训练集加载和推理图像批量加载
"""
import os
import cv2
import numpy as np
from glob import glob
from PIL import Image
from typing import List, Tuple, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.logger import get_logger

logger = get_logger("dataset")


class TrainDataset(Dataset):
    """
    训练数据集 - 仅加载OK（正常）样本
    支持: jpg, jpeg, png, bmp, tiff
    """

    # 支持的图像格式
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

    def __init__(
        self,
        data_path: str,
        image_size: tuple = (224, 224),
        mean: list = None,
        std: list = None,
        use_dataset_norm: bool = False,
    ):
        """
        Args:
            data_path: 包含OK样本的文件夹路径
            image_size: 目标图像尺寸 (height, width)
            mean: 自定义RGB均值（None时使用ImageNet默认值）
            std: 自定义RGB标准差（None时使用ImageNet默认值）
            use_dataset_norm: 是否自动计算数据集的均值和标准差
        """
        self.data_path = data_path
        self.image_size = image_size

        # 扫描文件夹获取所有图像文件路径
        self.image_paths = self._scan_images(data_path)

        # 确定Normalization参数
        if use_dataset_norm and len(self.image_paths) > 0:
            mean, std = self._compute_dataset_stats(self.image_paths)
            logger.info(f"使用数据集自适应归一化: mean={mean}, std={std}")
        elif mean is None:
            mean = [0.485, 0.456, 0.406]  # ImageNet 默认
            std = [0.229, 0.224, 0.225]

        # 定义图像预处理流水线
        self.transform = transforms.Compose([
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        logger.info(f"训练数据集加载完成: {len(self.image_paths)} 个OK样本")

    def _scan_images(self, data_path: str) -> List[str]:
        """
        递归扫描文件夹，收集所有支持的图像文件

        Args:
            data_path: 文件夹路径

        Returns:
            图像文件路径列表

        Raises:
            FileNotFoundError: 路径不存在时
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据路径不存在: {data_path}")

        image_paths = []
        for ext in self.IMG_EXTENSIONS:
            # 递归搜索所有子文件夹
            pattern = os.path.join(data_path, "**", f"*{ext}")
            found = glob(pattern, recursive=True)
            image_paths.extend(found)

        if len(image_paths) == 0:
            logger.warning(f"在 {data_path} 中未找到任何图像文件")
        else:
            logger.info(f"扫描到 {len(image_paths)} 个图像文件")

        return sorted(image_paths)

    def __len__(self) -> int:
        """返回数据集样本数量"""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        获取单个样本

        Args:
            idx: 索引

        Returns:
            (图像张量, 文件名)
        """
        img_path = self.image_paths[idx]
        try:
            # 使用PIL加载图像
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image, os.path.basename(img_path)
        except Exception as e:
            logger.error(f"加载图像失败 [{img_path}]: {str(e)}")
            # 返回空白张量作为替代
            blank = torch.zeros(3, self.image_size[0], self.image_size[1])
            return blank, os.path.basename(img_path)

    @staticmethod
    def _compute_dataset_stats(image_paths: list, sample_limit: int = 100) -> tuple:
        """
        从数据集中计算RGB均值和标准差（用于自适应归一化）

        Args:
            image_paths: 图像路径列表
            sample_limit: 最多计算样本数

        Returns:
            (mean_list, std_list) 每个长度为3
        """
        import cv2
        means, stds = [], []
        sample_paths = image_paths[:sample_limit]
        for f in sample_paths:
            img = cv2.imread(f)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                means.append(img_rgb.mean(axis=(0, 1)))
                stds.append(img_rgb.std(axis=(0, 1)))
        if not means:
            return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        mean = np.mean(means, axis=0).tolist()
        std = np.mean(stds, axis=0).tolist()
        # 归一化到[0,1]范围（因为ToTensor会除以255）
        mean = [round(m / 255.0, 4) for m in mean]
        std = [round(s / 255.0, 4) for s in std]
        return mean, std

    @staticmethod
    def get_dataloader(
        dataset: "TrainDataset",
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """创建DataLoader"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,  # Windows建议设为False
            drop_last=True,    # 丢弃最后一个不完整的batch
        )


class InferenceDataset(Dataset):
    """
    推理数据集 - 加载待检测的图像（可来自子文件夹）
    """

    def __init__(self, data_path: str, image_size: tuple = (224, 224)):
        """
        Args:
            data_path: 图像文件夹路径
            image_size: 目标图像尺寸
        """
        self.data_path = data_path
        self.image_size = image_size
        self.image_paths = []
        self._scan_recursive(data_path)

        # 推理时仅做resize和归一化，不做数据增强
        self.transform = transforms.Compose([
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        logger.info(f"推理数据集加载完成: {len(self.image_paths)} 个图像")

    def _scan_recursive(self, data_path: str):
        """递归扫描所有子文件夹中的图像"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"路径不存在: {data_path}")

        for ext in TrainDataset.IMG_EXTENSIONS:
            pattern = os.path.join(data_path, "**", f"*{ext}")
            found = glob(pattern, recursive=True)
            self.image_paths.extend(found)
        self.image_paths = sorted(self.image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        """
        返回: (图像张量, 文件名, 完整路径)
        """
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            # 保存原始图像用于可视化
            original = np.array(image)
            original = cv2.resize(original, self.image_size)
            image_tensor = self.transform(image)
            return image_tensor, os.path.basename(img_path), img_path
        except Exception as e:
            logger.error(f"加载图像失败 [{img_path}]: {str(e)}")
            blank = torch.zeros(3, self.image_size[0], self.image_size[1])
            return blank, os.path.basename(img_path), img_path
