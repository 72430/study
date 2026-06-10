"""
配置层 - 集中管理所有可配置参数
支持从配置文件/环境变量加载，并可在运行时通过UI动态修改
"""
import os
import json
import copy
from pathlib import Path


class TrainingConfig:
    """训练参数配置类，管理所有与训练相关的超参数"""

    def __init__(self):
        # ---------- 特征库参数 ----------
        self.data_path: str = ""                # 训练数据文件夹路径（仅OK样本）
        self.image_size: tuple = (224, 224)     # 输入图像尺寸 (height, width)
        self.backbone: str = "resnet50"         # 骨干网络: resnet18/34/50, wide_resnet50_2

        # ---------- 推理参数 ----------
        self.threshold: float = 0.05            # 异常判定阈值（由训练自动计算）

        # ---------- 路径参数 ----------
        self.checkpoint_dir: str = str(
            Path(__file__).resolve().parent.parent / "checkpoints"
        )
        self.log_dir: str = str(
            Path(__file__).resolve().parent.parent / "logs"
        )

        # ---------- 设备参数 ----------
        self.device: str = "auto"               # auto/cuda/cpu

    def to_dict(self) -> dict:
        """将配置转换为字典（用于保存/序列化）"""
        return copy.deepcopy(self.__dict__)

    def from_dict(self, cfg_dict: dict):
        """从字典加载配置"""
        for key, value in cfg_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save(self, file_path: str):
        """保存配置到JSON文件"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    def __repr__(self) -> str:
        return f"TrainingConfig({self.to_dict()})"


class UIConfig:
    """UI界面配置"""

    def __init__(self):
        self.window_title: str = "SuperSimpleNet - 工业无监督缺陷检测系统"
        self.window_size: tuple = (1280, 860)       # 默认窗口大小
        self.fullscreen: bool = False                # 全屏模式
        self.style: str = "industrial"              # 主题风格
        self.language: str = "zh_CN"                # 界面语言

        # 颜色定义
        self.color_ok: str = "#00C853"               # OK判定颜色 (绿色)
        self.color_ng: str = "#FF1744"               # NG判定颜色 (红色)
        self.color_bg_dark: str = "#1E1E2E"          # 深色背景
        self.color_bg_light: str = "#F5F5F5"         # 浅色背景
        self.color_accent: str = "#2196F3"           # 强调色
        self.color_text: str = "#FFFFFF"             # 文字颜色


# 全局单例配置
training_cfg = TrainingConfig()
ui_cfg = UIConfig()
