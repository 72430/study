"""
模型管理器 - 统一的模型命名、保存、加载、验证与历史管理
提供完整的生命周期管理: 命名验证 → 保存(含元数据) → 加载(含完整性校验) → 历史追溯
"""
import os
import re
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch

from config.config import TrainingConfig
from utils.logger import get_logger

logger = get_logger("model_manager")

# 系统版本常量，用于版本兼容性检查
SYSTEM_VERSION = "1.0.0"
SUPPORTED_VERSION_PREFIX = "1."


class ModelNameValidator:
    """模型名称验证器 - 校验命名规范与唯一性"""

    # 允许的字符: 中文、英文字母、数字、下划线、短横线、点、空格、括号
    NAME_PATTERN = re.compile(r'^[\u4e00-\u9fa5a-zA-Z0-9_\-\.\s\(\)（）\u3001\uFF08\uFF09]+$')

    @classmethod
    def validate(cls, name: str) -> Tuple[bool, str]:
        """
        验证模型名称是否合法

        Args:
            name: 待验证的模型名称

        Returns:
            (is_valid, error_message) 元组
        """
        if not name or not name.strip():
            return False, "模型名称不能为空"

        name = name.strip()

        if len(name) < 3:
            return False, f"模型名称长度不能少于3个字符（当前: {len(name)}个）"

        if len(name) > 50:
            return False, f"模型名称长度不能超过50个字符（当前: {len(name)}个）"

        if not cls.NAME_PATTERN.match(name):
            return False, (
                "模型名称仅支持中英文、数字及以下符号:\n"
                "下划线(_) 短横线(-) 点(.) 空格( ) 括号()（）"
            )

        return True, ""

    @classmethod
    def sanitize_filename(cls, name: str) -> str:
        """将模型名称转换为安全的文件名字符"""
        # 替换空格为下划线，移除不安全的文件名字符
        safe = name.strip().replace(" ", "_")
        safe = re.sub(r'[<>:"/\\|?*]', '', safe)
        return safe[:100]  # 限制最大长度


class ModelMetaManager:
    """模型元数据管理器 - 维护模型注册表，记录所有已保存的模型信息"""

    def __init__(self, registry_dir: str = None):
        """
        Args:
            registry_dir: 元数据注册表存储目录，默认在checkpoints下
        """
        if registry_dir is None:
            registry_dir = str(
                Path(__file__).resolve().parent.parent / "checkpoints"
            )
        self.registry_dir = registry_dir
        self.registry_file = os.path.join(registry_dir, "model_registry.json")
        os.makedirs(registry_dir, exist_ok=True)

    def load_registry(self) -> Dict[str, dict]:
        """加载模型注册表"""
        if not os.path.exists(self.registry_file):
            return {}
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"注册表文件损坏，将重建: {str(e)}")
            return {}

    def save_registry(self, registry: Dict[str, dict]):
        """保存模型注册表"""
        try:
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存注册表失败: {str(e)}")

    def is_name_unique(self, name: str, exclude_file: str = None) -> bool:
        """
        检查模型名称是否唯一

        Args:
            name: 模型名称
            exclude_file: 排除的文件名（更新时使用）

        Returns:
            是否唯一
        """
        registry = self.load_registry()
        for file_key, meta in registry.items():
            if meta.get("name") == name:
                if exclude_file and file_key == exclude_file:
                    continue
                return False
        return True

    def register_model(
        self, file_path: str, metadata: dict
    ):
        """注册一个模型到注册表"""
        registry = self.load_registry()
        file_key = os.path.basename(file_path)
        registry[file_key] = metadata
        self.save_registry(registry)
        logger.info(f"模型已注册: {metadata.get('name', 'unknown')} -> {file_key}")

    def get_model_info(self, file_path: str) -> Optional[dict]:
        """获取指定模型的元数据"""
        registry = self.load_registry()
        file_key = os.path.basename(file_path)
        return registry.get(file_key)

    def list_all_models(self) -> List[dict]:
        """列出所有已注册的模型信息（按时间降序）"""
        registry = self.load_registry()
        models = []
        for file_key, meta in registry.items():
            meta["_file_key"] = file_key
            models.append(meta)
        # 按时间降序排列
        models.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return models

    def delete_model_record(self, file_path: str):
        """从注册表删除模型记录"""
        registry = self.load_registry()
        file_key = os.path.basename(file_path)
        if file_key in registry:
            del registry[file_key]
            self.save_registry(registry)
            logger.info(f"模型记录已删除: {file_key}")


class ModelIntegrityChecker:
    """模型完整性校验器 - 确保模型文件的完整性和兼容性"""

    @staticmethod
    def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
        """计算文件的哈希值"""
        hash_func = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    @staticmethod
    def check_file_format(file_path: str) -> Tuple[bool, str]:
        """
        检查文件格式是否合法

        Returns:
            (is_valid, message)
        """
        if not os.path.exists(file_path):
            return False, f"文件不存在: {file_path}"

        if not file_path.endswith(".pth"):
            return False, f"文件格式不支持，仅支持.pth格式: {file_path}"

        # 检查文件大小（至少1KB）
        file_size = os.path.getsize(file_path)
        if file_size < 1024:
            return False, f"文件大小异常（{file_size}字节），可能已损坏"

        return True, ""

    @staticmethod
    def check_integrity(file_path: str, expected_hash: str = None) -> Tuple[bool, str]:
        """
        检查模型文件完整性

        Args:
            file_path: 模型文件路径
            expected_hash: 预期的哈希值（可选）

        Returns:
            (is_valid, message)
        """
        # 文件格式检查
        valid, msg = ModelIntegrityChecker.check_file_format(file_path)
        if not valid:
            return False, msg

        try:
            # 尝试用torch加载，验证是否为有效的模型文件
            checkpoint = torch.load(file_path, map_location="cpu")

            # 检查必要的键是否存在
            if isinstance(checkpoint, dict):
                # 检查是否为 FeatureBank 模式（特征库模型，无传统state_dict）
                is_fb = (
                    checkpoint.get("method") == "feature_bank"
                    or "reference_embedding" in checkpoint
                )

                if is_fb:
                    # FeatureBank 模型验证：只需确保特征库存在
                    ref = checkpoint.get("reference_embedding")
                    if ref is None:
                        return False, "FeatureBank模型缺少特征库(reference_embedding)"
                    # 检查特征库是有效的张量
                    if hasattr(ref, 'shape') and ref.numel() > 0:
                        return True, "FeatureBank特征库模型校验通过"
                    else:
                        return False, "FeatureBank模型的特征库为空"
                elif "model_state_dict" in checkpoint:
                    # 完整检查点
                    state_dict = checkpoint["model_state_dict"]
                elif "mapping_net" in checkpoint:
                    state_dict = checkpoint["mapping_net"]
                else:
                    # 可能直接是state_dict
                    state_dict = checkpoint

                # 验证state_dict是有效的模型参数
                if not isinstance(state_dict, dict):
                    return False, "模型文件格式异常：state_dict类型不匹配"

                # 检查参数是否为空
                if len(state_dict) == 0:
                    return False, "模型文件为空（无任何参数）"

            # 如果有预期哈希值，验证哈希
            if expected_hash:
                actual_hash = ModelIntegrityChecker.compute_file_hash(file_path)
                if actual_hash != expected_hash:
                    return False, f"文件哈希校验失败，文件可能被篡改"

            return True, "模型完整性校验通过"

        except (EOFError, RuntimeError, ValueError, KeyError) as e:
            return False, f"模型文件损坏或无法解析: {str(e)}"
        except Exception as e:
            return False, f"模型完整性检查异常: {str(e)}"

    @staticmethod
    def check_version_compatibility(checkpoint: dict) -> Tuple[bool, str]:
        """
        检查模型版本兼容性

        Args:
            checkpoint: 加载的检查点字典

        Returns:
            (compatible, message)
        """
        saved_version = checkpoint.get("system_version", "0.0.0")
        if saved_version.startswith(SUPPORTED_VERSION_PREFIX):
            return True, f"版本兼容 (模型版本: {saved_version}, 系统版本: {SYSTEM_VERSION})"
        else:
            return (
                False,
                f"版本不兼容！模型版本: {saved_version}, "
                f"系统版本: {SYSTEM_VERSION}，"
                f"支持版本前缀: {SUPPORTED_VERSION_PREFIX}x"
            )


class ModelManager:
    """
    模型管理器 - 统一接口

    整合名称验证、元数据管理、完整性校验功能
    提供完整的模型保存/加载/查询生命周期管理
    """

    def __init__(self, checkpoint_dir: str = None):
        if checkpoint_dir is None:
            checkpoint_dir = str(
                Path(__file__).resolve().parent.parent / "checkpoints"
            )
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.name_validator = ModelNameValidator()
        self.meta_manager = ModelMetaManager(checkpoint_dir)
        self.integrity_checker = ModelIntegrityChecker()

    # ==================== 命名管理 ====================

    def validate_name(self, name: str, exclude_file: str = None) -> Tuple[bool, str]:
        """
        完整名称验证：格式校验 + 唯一性校验

        Args:
            name: 模型名称
            exclude_file: 排除的文件（更新时）

        Returns:
            (is_valid, message)
        """
        # 1. 格式校验
        is_valid, error_msg = self.name_validator.validate(name)
        if not is_valid:
            return False, error_msg

        # 2. 唯一性校验
        clean_name = name.strip()
        if not self.meta_manager.is_name_unique(clean_name, exclude_file):
            return False, f"模型名称「{clean_name}」已存在，请使用其他名称"

        return True, "名称验证通过"

    # ==================== 保存管理 ====================

    def prepare_save(
        self, model_name: str, config: TrainingConfig,
        best_loss: float, current_epoch: int, total_epochs: int,
        train_losses: list, total_time: float,
        state_dict: dict, optimizer_state_dict: dict = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        准备保存模型：验证名称 → 构建元数据 → 生成保存路径

        Args:
            model_name: 自定义模型名称
            config: 训练配置
            best_loss: 最佳损失
            current_epoch: 当前轮次
            total_epochs: 总轮次
            train_losses: 训练损失记录
            total_time: 训练总耗时(秒)
            state_dict: 模型权重
            optimizer_state_dict: 优化器状态（可选）

        Returns:
            (success, message, save_path)
        """
        # 1. 验证名称
        is_valid, msg = self.validate_name(model_name)
        if not is_valid:
            return False, msg, None

        # 2. 生成安全的文件名
        safe_name = self.name_validator.sanitize_filename(model_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{safe_name}_{timestamp}.pth"
        save_path = os.path.join(self.checkpoint_dir, file_name)

        # 3. 构建元数据
        metadata = {
            "name": model_name.strip(),
            "file_name": file_name,
            "timestamp": datetime.now().isoformat(),
            "system_version": SYSTEM_VERSION,

            # 模型架构
            "backbone": config.backbone,
            "embedding_dim": config.embedding_dim,

            # 训练配置
            "image_size": list(config.image_size),
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "epochs": total_epochs,

            # 性能指标
            "best_loss": best_loss,
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "total_time_seconds": total_time,
            "train_losses": train_losses,

            # 推理配置
            "threshold": config.threshold,
        }

        # 4. 构建检查点
        checkpoint = {
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "config": config.to_dict(),
            "metadata": metadata,
            "system_version": SYSTEM_VERSION,
            "epoch": current_epoch,
            "best_loss": best_loss,
            "train_losses": train_losses,
        }

        # 5. 保存到文件
        try:
            torch.save(checkpoint, save_path)
            # 6. 注册到元数据
            self.meta_manager.register_model(save_path, metadata)
            logger.info(f"模型已保存: {model_name} -> {save_path}")
            return True, f"模型「{model_name}」保存成功", save_path
        except Exception as e:
            logger.error(f"模型保存失败: {str(e)}")
            return False, f"模型保存失败: {str(e)}", None

    # ==================== 加载管理 ====================

    def load_with_validation(
        self, file_path: str, map_location: str = "cpu"
    ) -> Tuple[bool, str, Optional[dict]]:
        """
        加载模型并执行完整的验证链

        验证流程:
          1. 文件格式检查
          2. 文件完整性检查（尝试torch.load）
          3. 版本兼容性检查
          4. 参数完整性检查

        Args:
            file_path: 模型文件路径
            map_location: 加载设备映射

        Returns:
            (success, message, checkpoint_dict)
        """
        # 1. 文件格式检查
        valid, msg = self.integrity_checker.check_file_format(file_path)
        if not valid:
            return False, msg, None

        # 2. 文件完整性检查
        valid, msg = self.integrity_checker.check_integrity(file_path)
        if not valid:
            return False, msg, None

        # 3. 加载模型
        try:
            checkpoint = torch.load(file_path, map_location=map_location)
        except Exception as e:
            return False, f"模型加载失败: {str(e)}", None

        if not isinstance(checkpoint, dict):
            return False, "模型文件格式异常：不是有效的字典格式", None

        # 4. 版本兼容性检查
        compatible, ver_msg = self.integrity_checker.check_version_compatibility(checkpoint)
        if not compatible:
            return False, ver_msg, None

        # 5. 验证模型参数键名
        # FeatureBank 模式检测：根据 method 或 reference_embedding 识别
        is_fb_mode = (
            checkpoint.get("method") == "feature_bank"
            or "reference_embedding" in checkpoint
        )

        if is_fb_mode:
            # FeatureBank 模型无需验证 state_dict，验证特征库即可
            ref = checkpoint.get("reference_embedding")
            if ref is None or (hasattr(ref, 'numel') and ref.numel() == 0):
                return False, "FeatureBank模型特征库为空或无效", None

            # 提取元数据展示
            metadata = {
                "name": checkpoint.get("name", "FeatureBank检测模型"),
                "system_version": checkpoint.get("system_version", "1.0.0"),
                "backbone": checkpoint.get("backbone", "resnet50"),
                "embedding_dim": checkpoint.get("embedding_dim", ref.shape[-1] if hasattr(ref, 'shape') else 0),
                "ok_sample_count": checkpoint.get("ok_sample_count", "?"),
                "ok_score_mean": checkpoint.get("ok_score_mean", 0.0),
                "ok_score_std": checkpoint.get("ok_score_std", 0.0),
                "suggested_threshold": checkpoint.get("suggested_threshold", 0.05),
                "method": "feature_bank",
                "knn_k": checkpoint.get("knn_k", 3),
            }
            return True, "FeatureBank特征库模型加载成功", checkpoint

        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            # 尝试直接作为state_dict
            if any(k.startswith("mapping_net") for k in checkpoint.keys()):
                state_dict = checkpoint
            else:
                # 查找任何包含卷积层参数的字典
                for key, val in checkpoint.items():
                    if isinstance(val, dict) and any(
                        k.endswith((".weight", ".bias")) for k in val.keys()
                    ):
                        state_dict = val
                        break

        if state_dict is None:
            return False, "模型文件中未找到有效的网络参数", None

        # 验证参数数量合理
        param_count = sum(v.numel() for v in state_dict.values())
        if param_count < 1000:
            return False, f"模型参数异常（仅{param_count}个参数），可能已损坏", None

        # 提取元数据展示
        metadata = checkpoint.get("metadata", {})
        if not metadata:
            # 从检查点顶层字段构建展示信息
            metadata = {
                "name": os.path.basename(file_path),
                "system_version": checkpoint.get("system_version", "未知"),
                "best_loss": checkpoint.get("best_loss", "未知"),
                "backbone": checkpoint.get("config", {}).get("backbone", "未知"),
                "current_epoch": checkpoint.get("epoch", 0),
            }

        # 返回成功，包含完整的checkpoint
        metadata["_param_count"] = param_count
        metadata["_file_path"] = file_path
        checkpoint["_loaded_metadata"] = metadata

        logger.info(
            f"模型加载验证通过: {metadata.get('name', 'unknown')} | "
            f"参数: {param_count:,} | "
            f"版本: {metadata.get('system_version', '?')}"
        )
        return True, f"模型「{metadata.get('name', '未知')}」加载成功", checkpoint

    # ==================== 查询管理 ====================

    def list_models(self) -> List[dict]:
        """列出所有已保存的模型信息"""
        return self.meta_manager.list_all_models()

    def get_model_summary(self, file_path: str) -> Optional[str]:
        """获取模型的摘要信息文本"""
        info = self.meta_manager.get_model_info(file_path)
        if not info:
            return None

        summary = (
            f"📋 模型名称: {info.get('name', '未知')}\n"
            f"─────────────────────────\n"
            f"🕐 创建时间: {info.get('timestamp', '未知')}\n"
            f"🏗  骨干网络: {info.get('backbone', '未知')}\n"
            f"📊 嵌入维度: {info.get('embedding_dim', '未知')}\n"
            f"🎯 最佳损失: {info.get('best_loss', '未知'):.6f}\n"
            f"📚 训练轮次: {info.get('current_epoch', 0)}/{info.get('total_epochs', 0)}\n"
            f"⏱  训练耗时: {info.get('total_time_seconds', 0):.1f}秒\n"
            f"📐 图像尺寸: {info.get('image_size', '未知')}\n"
            f"🎛  批处理大小: {info.get('batch_size', '未知')}\n"
            f"⚡ 学习率: {info.get('learning_rate', '未知')}\n"
            f"🔧 系统版本: {info.get('system_version', '未知')}"
        )
        return summary

    def get_model_display_info(self, file_path: str) -> Optional[Dict]:
        """获取模型显示信息（用于UI展示）"""
        info = self.meta_manager.get_model_info(file_path)
        if not info:
            return None
        return {
            "name": info.get("name", "未知"),
            "timestamp": info.get("timestamp", "未知"),
            "backbone": info.get("backbone", "未知"),
            "best_loss": info.get("best_loss", "N/A"),
            "epochs": f"{info.get('current_epoch', 0)}/{info.get('total_epochs', 0)}",
            "time": f"{info.get('total_time_seconds', 0):.1f}s",
            "image_size": info.get("image_size", "未知"),
            "batch_size": info.get("batch_size", "未知"),
            "param_count": info.get("_param_count", "未知"),
        }


class OperationHistory:
    """操作历史记录 - 记录所有模型相关操作的日志"""

    def __init__(self, history_dir: str = None):
        if history_dir is None:
            history_dir = str(
                Path(__file__).resolve().parent.parent / "logs"
            )
        self.history_file = os.path.join(history_dir, "operation_history.json")
        os.makedirs(history_dir, exist_ok=True)
        self._history: List[dict] = []
        self._load()

    def _load(self):
        """从文件加载历史记录"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self._history = json.load(f)
            except (json.JSONDecodeError, Exception):
                self._history = []

    def _save(self):
        """保存历史记录到文件"""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self._history[-200:], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"保存操作历史失败: {str(e)}")

    def add_record(self, operation: str, status: str, detail: str = ""):
        """
        添加一条操作记录

        Args:
            operation: 操作类型（如"模型保存"、"模型加载"、"模型训练"）
            status: 状态（"成功" / "失败"）
            detail: 详细信息
        """
        record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operation": operation,
            "status": status,
            "detail": detail,
        }
        self._history.append(record)
        self._save()
        logger.info(f"[操作记录] {operation} | {status} | {detail}")

    def get_history(self, limit: int = 50) -> List[dict]:
        """获取最近的操作历史"""
        return self._history[-limit:][::-1]  # 最新在前

    def get_formatted_history(self, limit: int = 20) -> str:
        """获取格式化的历史记录文本"""
        records = self.get_history(limit)
        if not records:
            return "暂无操作记录"

        lines = ["📋 操作历史记录", "=" * 50]
        for r in records:
            status_icon = "✅" if r["status"] == "成功" else "❌"
            lines.append(
                f"[{r['time']}] {status_icon} {r['operation']} | {r['detail']}"
            )
        return "\n".join(lines)
