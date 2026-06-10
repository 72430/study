"""
推理引擎模块 - 加载模型进行批量/单张图像异常检测
输出: OK/NG判定、异常置信度分数、推理时间、模型信息
"""
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from model.sup_simple_net import SuperSimpleNet
from core.dataset import InferenceDataset
from core.model_manager import ModelManager, OperationHistory
from config.config import TrainingConfig
from utils.logger import get_logger

logger = get_logger("inferencer")


class InferenceResult:
    """单张图像的推理结果"""

    def __init__(
        self,
        file_name: str,
        file_path: str,
        is_anomaly: bool,
        anomaly_score: float,
        inference_time_ms: float,
    ):
        self.file_name = file_name
        self.file_path = file_path
        self.is_anomaly = is_anomaly          # True=NG, False=OK
        self.anomaly_score = anomaly_score     # 异常得分 [0,1]
        self.inference_time_ms = inference_time_ms  # 推理耗时(ms)

    @property
    def label(self) -> str:
        """判定标签"""
        return "NG" if self.is_anomaly else "OK"

    @property
    def score_text(self) -> str:
        """格式化得分文本"""
        return f"{self.anomaly_score:.2%}"

    def __repr__(self) -> str:
        return (
            f"[{self.label}] {self.file_name} | "
            f"Score: {self.score_text} | "
            f"Time: {self.inference_time_ms:.1f}ms"
        )


class ModelLoadResult:
    """模型加载结果 - 包含加载状态、详细信息及模型元数据"""

    def __init__(self):
        self.success: bool = False
        self.message: str = ""
        self.model_info: Optional[Dict] = None  # 模型元数据（用于UI展示）
        self.summary_text: str = ""              # 模型摘要文本

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "message": self.message,
            "model_info": self.model_info,
        }


class Inferencer:
    """
    推理引擎

    功能:
      - 加载训练好的模型（含完整性校验、版本兼容性检查）
      - 显示模型基本信息和性能指标
      - 计算参考嵌入（训练集正常样本的特征中心）
      - 对单张/批量图像进行异常检测
    """

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.device = self._resolve_device(config.device)
        self.model_manager = ModelManager(config.checkpoint_dir)
        self.op_history = OperationHistory()

        self.model: Optional[SuperSimpleNet] = None
        self.reference_embedding: Optional[torch.Tensor] = None
        self._is_loaded = False

        # 当前加载的模型信息
        self._current_model_info: Optional[Dict] = None

        # 图像预处理参数（用于可视化）
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # FeatureBank 模式标志
        self._fb_mode = False
        self._fb_knn_k = 3
        self._fb_backbone = None

    def _resolve_device(self, device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def load_model(self, checkpoint_path: str) -> bool:
        """
        [保留] 原始加载方式 - 简单直接加载

        Args:
            checkpoint_path: 模型文件路径

        Returns:
            是否加载成功
        """
        result = self.load_model_with_validation(checkpoint_path)
        return result.success

    def load_model_with_validation(self, checkpoint_path: str) -> ModelLoadResult:
        """
        增强型模型加载 - 带完整的验证链和信息展示

        验证流程:
          1. 文件格式验证 (.pth)
          2. 文件存在性检查
          3. 完整性校验（尝试加载并验证参数结构）
          4. 版本兼容性验证
          5. 参数完整性检查

        Args:
            checkpoint_path: 模型文件路径

        Returns:
            ModelLoadResult 包含加载结果、消息、模型信息
        """
        result = ModelLoadResult()

        # ------ 步骤1: 使用ModelManager执行完整验证链 ------
        success, msg, checkpoint = self.model_manager.load_with_validation(
            checkpoint_path, map_location=str(self.device)
        )

        if not success:
            result.message = msg
            logger.error(f"模型加载验证失败: {msg}")
            self.op_history.add_record("模型加载", "失败", msg)
            return result

        # ------ 步骤2: 检测是否为FeatureBank模型 ------
        method = checkpoint.get("method", "")
        is_fb = method == "feature_bank" or "reference_embedding" in checkpoint and "model_state_dict" not in checkpoint

        if is_fb:
            # FeatureBank 模式：无需神经网络，直接使用特征库
            ref = checkpoint.get("reference_embedding")
            if ref is None:
                result.message = "模型格式异常: 缺少特征库 (reference_embedding)"
                logger.error(result.message)
                self.op_history.add_record("模型加载", "失败", result.message)
                return result

            self.reference_embedding = ref if ref.dim() > 1 else ref.unsqueeze(0)
            self._is_loaded = True
            self._fb_mode = True
            self._fb_knn_k = checkpoint.get("knn_k", 3)
            self._fb_backbone = checkpoint.get("backbone", "resnet50")

            # 构建FeatureBank骨干网络用于特征提取
            from model.backbones import get_backbone
            self.model = get_backbone(
                self._fb_backbone, pretrained=True
            ).to(self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

            # 提取并存储模型信息
            metadata = checkpoint.get("_loaded_metadata", {})
            metadata["_file_path"] = checkpoint_path
            metadata["_file_name"] = os.path.basename(checkpoint_path)
            metadata["_file_size"] = self._format_file_size(
                os.path.getsize(checkpoint_path)
            )
            metadata["backbone"] = self._fb_backbone
            metadata["embedding_dim"] = self.reference_embedding.shape[-1]
            metadata["image_size"] = [224, 224]
            metadata["device"] = str(self.device)
            metadata["method"] = "FeatureBank (kNN)"
            metadata["name"] = checkpoint.get("name", "FeatureBank模型")
            metadata["ok_sample_count"] = checkpoint.get("ok_sample_count", "?")
            metadata["suggested_threshold"] = checkpoint.get("suggested_threshold", 0.05)
            metadata["ok_score_mean"] = checkpoint.get("ok_score_mean", 0.0)
            metadata["ok_score_std"] = checkpoint.get("ok_score_std", 0.0)

            # 使用建议阈值
            suggested_th = metadata.get("suggested_threshold", 0.05)
            self.cfg.threshold = suggested_th

            self._current_model_info = metadata

            # 生成摘要文本
            summary = self._build_fb_summary(metadata)
            result.model_info = metadata
            result.summary_text = summary
            result.success = True
            result.message = f"特征库模型「{metadata.get('name', '未知')}」加载成功"

            logger.info(f"\n{summary}")
            self.op_history.add_record(
                "模型加载", "成功",
                f"名称: {metadata.get('name', '未知')} | "
                f"骨干: {self._fb_backbone} | "
                f"特征库: {self.reference_embedding.shape}"
            )
            return result

        # ------ 步骤3: 传统SuperSimpleNet模型加载（原逻辑） ------
        try:
            # 提取配置信息
            cfg_dict = checkpoint.get("config", {})
            backbone = cfg_dict.get("backbone", self.cfg.backbone)
            embedding_dim = cfg_dict.get("embedding_dim", self.cfg.embedding_dim)

            # 构建模型（使用保存时的配置）
            self.model = SuperSimpleNet(
                backbone=backbone,
                embedding_dim=embedding_dim,
                pretrained=False,
            ).to(self.device)
            self.model.set_eval_mode()

            # 加载状态字典
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            if isinstance(state_dict, dict) and "model_state_dict" not in checkpoint:
                # 尝试直接作为state_dict使用
                pass
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

            self.model.mapping_net.load_state_dict(state_dict)

        except Exception as e:
            result.message = f"模型重建失败（架构可能不兼容）: {str(e)}"
            logger.error(result.message)
            self.op_history.add_record("模型加载", "失败", result.message)
            return result

        # ------ 步骤3: 提取并存储模型信息 ------
        metadata = checkpoint.get("_loaded_metadata", {})

        # 加载参考嵌入（如果有）
        self.reference_embedding = checkpoint.get("reference_embedding", None)
        if self.reference_embedding is not None:
            logger.info(f"从模型文件加载参考嵌入 (shape={self.reference_embedding.shape})")
        else:
            logger.warning("模型文件中未包含参考嵌入，请在推理前调用 compute_reference_embedding()")
        metadata["_file_path"] = checkpoint_path
        metadata["_file_name"] = os.path.basename(checkpoint_path)
        metadata["_file_size"] = self._format_file_size(
            os.path.getsize(checkpoint_path)
        )
        metadata["backbone"] = cfg_dict.get("backbone", backbone)
        metadata["embedding_dim"] = cfg_dict.get("embedding_dim", embedding_dim)
        metadata["image_size"] = cfg_dict.get("image_size", [224, 224])
        metadata["device"] = str(self.device)

        self._current_model_info = metadata
        self._is_loaded = True

        # 生成摘要文本
        summary = self._build_model_summary(metadata)
        result.model_info = metadata
        result.summary_text = summary
        result.success = True
        result.message = f"模型「{metadata.get('name', '未知')}」加载成功"

        logger.info(f"\n{summary}")
        self.op_history.add_record(
            "模型加载", "成功",
            f"名称: {metadata.get('name', '未知')} | "
            f"骨干: {metadata.get('backbone', '?')} | "
            f"参数: {metadata.get('_param_count', '?'):,}"
        )
        return result

    def _build_model_summary(self, metadata: dict) -> str:
        """构建模型摘要文本（用于UI展示）"""
        name = metadata.get("name", "未知")
        backbone = metadata.get("backbone", "未知")
        embed_dim = metadata.get("embedding_dim", "未知")
        best_loss = metadata.get("best_loss", "N/A")
        if isinstance(best_loss, float):
            best_loss = f"{best_loss:.6f}"
        epochs = metadata.get("current_epoch", "?")
        total_epochs = metadata.get("total_epochs", "?")
        total_time = metadata.get("total_time_seconds", 0)
        if isinstance(total_time, (int, float)):
            total_time = f"{total_time:.1f}s"
        img_size = metadata.get("image_size", "未知")
        batch_size = metadata.get("batch_size", "未知")
        lr = metadata.get("learning_rate", "未知")
        param_count = metadata.get("_param_count", "未知")
        if isinstance(param_count, (int, float)):
            param_count = f"{param_count:,}"
        file_size = metadata.get("_file_size", "未知")
        timestamp = metadata.get("timestamp", "未知")

        summary = (
            "╔══════════════════════════════════════╗\n"
            "║       模型加载信息                   ║\n"
            "╠══════════════════════════════════════╣\n"
            f"║  📋 名称:       {str(name):<24} ║\n"
            f"║  🕐 创建时间:   {str(timestamp):<24} ║\n"
            f"║  🏗  骨干网络:  {str(backbone):<24} ║\n"
            f"║  📊 嵌入维度:   {str(embed_dim):<24} ║\n"
            f"║  🎯 最佳损失:   {str(best_loss):<24} ║\n"
            f"║  📚 训练轮次:   {str(epochs)}/{str(total_epochs):<19} ║\n"
            f"║  ⏱  训练耗时:   {str(total_time):<24} ║\n"
            f"║  📐 图像尺寸:   {str(img_size):<24} ║\n"
            f"║  🎛  批处理:    {str(batch_size):<24} ║\n"
            f"║  ⚡ 学习率:     {str(lr):<24} ║\n"
            f"║  🔢 参数量:     {str(param_count):<24} ║\n"
            f"║  💾 文件大小:   {str(file_size):<24} ║\n"
            f"║  💻 运行设备:   {str(self.device):<24} ║\n"
            "╚══════════════════════════════════════╝"
        )
        return summary

    def _build_fb_summary(self, metadata: dict) -> str:
        """构建 FeatureBank 模型摘要文本"""
        name = metadata.get("name", "FeatureBank模型")
        backbone = metadata.get("backbone", "resnet50")
        embed_dim = metadata.get("embedding_dim", "?")
        file_size = metadata.get("_file_size", "未知")
        ok_count = metadata.get("ok_sample_count", "?")
        ok_mean = metadata.get("ok_score_mean", 0.0)
        ok_std = metadata.get("ok_score_std", 0.0)
        suggested_th = metadata.get("suggested_threshold", 0.05)

        summary = (
            "╔══════════════════════════════════════╗\n"
            "║    FeatureBank 特征库模型             ║\n"
            "╠══════════════════════════════════════╣\n"
            f"║  📋 名称:       {str(name):<24} ║\n"
            f"║  🔬 方法:       FeatureBank (kNN)    ║\n"
            f"║  🏗  骨干网络:  {str(backbone):<24} ║\n"
            f"║  📊 嵌入维度:   {str(embed_dim):<24} ║\n"
            f"║  📚 OK样本数:   {str(ok_count):<24} ║\n"
            f"║  📈 OK得分均值: {float(ok_mean):<10.4f}          ║\n"
            f"║  📉 OK得分标准差:{float(ok_std):<10.4f}          ║\n"
            f"║  🎯 建议阈值:   {float(suggested_th):<10.4f}     ║\n"
            f"║  💾 文件大小:   {str(file_size):<24} ║\n"
            f"║  💻 运行设备:   {str(self.device):<24} ║\n"
            "╚══════════════════════════════════════╝"
        )
        return summary

    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"

    def get_model_info(self) -> Optional[Dict]:
        """获取当前加载模型的详细信息"""
        return self._current_model_info

    def get_model_summary_text(self) -> str:
        """获取当前加载模型的摘要文本"""
        if self._current_model_info:
            return self._build_model_summary(self._current_model_info)
        return "未加载模型"

    def compute_reference_embedding(self, dataloader) -> bool:
        """
        计算训练集正常样本的参考嵌入（特征中心）

        Args:
            dataloader: 训练数据加载器

        Returns:
            是否计算成功
        """
        if not self._is_loaded or self.model is None:
            logger.error("模型未加载，无法计算参考嵌入")
            return False

        self.model.set_eval_mode()
        all_embeddings = []

        logger.info("开始计算参考嵌入...")
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                embeddings = self.model(images)
                all_embeddings.append(embeddings.cpu())

        if len(all_embeddings) == 0:
            logger.error("无有效数据计算参考嵌入")
            return False

        # 对所有正常样本嵌入取平均作为参考中心
        all_embeddings = torch.cat(all_embeddings, dim=0)
        self.reference_embedding = all_embeddings.mean(dim=0, keepdim=True)

        # 计算正常样本的得分分布用于动态阈值
        scores = []
        for emb in all_embeddings:
            sim = F.cosine_similarity(
                emb.unsqueeze(0), self.reference_embedding, dim=1
            )
            scores.append((1.0 - sim.item()) / 2.0)

        scores = np.array(scores)
        logger.info(
            f"参考嵌入计算完成 | "
            f"正常得分范围: [{scores.min():.4f}, {scores.max():.4f}] | "
            f"均值: {scores.mean():.4f} | "
            f"标准差: {scores.std():.4f}"
        )

        # 根据正常样本得分分布自动调整阈值
        auto_threshold = scores.mean() + 3 * scores.std()
        logger.info(f"建议阈值: {auto_threshold:.4f} (均值+3σ)")
        return True

    def infer_single(self, image_tensor: torch.Tensor) -> Tuple[bool, float, float]:
        """
        单张图像推理
        支持 FeatureBank (kNN) 和 SuperSimpleNet 两种模式

        Args:
            image_tensor: 预处理后的图像张量 [1, C, H, W]

        Returns:
            (is_anomaly, anomaly_score, inference_time_ms)
        """
        if not self._is_loaded or self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        start_time = time.time()

        if self._fb_mode:
            # ===== FeatureBank 模式：kNN 余弦距离 =====
            self.model.eval()
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                features = self.model(image_tensor)  # [1, 2048]
                features = torch.nn.functional.normalize(features, p=2, dim=1)

                bank = self.reference_embedding.to(self.device)  # [N, D]
                # 余弦相似度 → 余弦距离
                sim = torch.mm(features, bank.t())  # [1, N]
                distances = 1.0 - sim  # 余弦距离 [0, 2]
                # k 个最近邻平均
                k = min(self._fb_knn_k, distances.size(1))
                topk_dist, _ = torch.topk(distances, k=k, dim=1, largest=False)
                anomaly_score = float(topk_dist.mean().item())
        else:
            # ===== SuperSimpleNet 模式：余弦距离到参考嵌入 =====
            self.model.set_eval_mode()
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                embedding = self.model(image_tensor)

                if self.reference_embedding is not None:
                    ref = self.reference_embedding.to(self.device)
                    similarity = F.cosine_similarity(embedding, ref, dim=1)
                    anomaly_score = (1.0 - similarity.item()) / 2.0
                else:
                    anomaly_score = 0.5

        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        is_anomaly = anomaly_score >= self.cfg.threshold
        return is_anomaly, anomaly_score, inference_time

    def infer_batch(self, data_path: str) -> List[InferenceResult]:
        """
        对文件夹内所有图像进行批量推理（包含子文件夹）

        Args:
            data_path: 图像文件夹路径

        Returns:
            推理结果列表
        """
        from torch.utils.data import DataLoader

        dataset = InferenceDataset(data_path, self.cfg.image_size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        results = []
        logger.info(f"批量推理开始，共 {len(dataset)} 张图像")

        for image_tensor, file_name, file_path in dataloader:
            try:
                is_anomaly, score, time_ms = self.infer_single(image_tensor)
                result = InferenceResult(
                    file_name=file_name[0],
                    file_path=file_path[0],
                    is_anomaly=is_anomaly,
                    anomaly_score=score,
                    inference_time_ms=time_ms,
                )
                results.append(result)
                logger.debug(str(result))
            except Exception as e:
                logger.error(f"推理失败 [{file_name[0]}]: {str(e)}")

        # 统计
        ng_count = sum(1 for r in results if r.is_anomaly)
        ok_count = len(results) - ng_count
        logger.info(
            f"批量推理完成 | 总计: {len(results)} | "
            f"OK: {ok_count} | NG: {ng_count}"
        )
        self.op_history.add_record(
            "批量推理", "完成",
            f"总计: {len(results)}, OK: {ok_count}, NG: {ng_count}"
        )
        return results

    def set_threshold(self, threshold: float):
        """设置异常判定阈值"""
        self.cfg.threshold = threshold
        logger.info(f"阈值已更新: {threshold:.4f}")

    def get_operation_history(self, limit: int = 20) -> str:
        """获取操作历史文本"""
        return self.op_history.get_formatted_history(limit)
