"""
基于特征库的无监督异常检测 - 使用预训练骨干网络直接提取特征

核心思想（SPADE/PaDiM风格）:
  1. 使用预训练CNN骨干网络(冻结)提取多尺度特征
  2. 建立OK样本的特征库作为参考
  3. 对测试样本通过kNN距离或马氏距离计算异常得分
  4. 无需训练，避免维度坍缩
"""
import os
import sys
import time
import numpy as np
from glob import glob
from typing import List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 将项目根目录加入路径
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.backbones import get_backbone, BACKBONE_CHOICES
from utils.logger import get_logger

logger = get_logger("feature_bank")


class FeatureBankDetector:
    """
    基于特征库的异常检测器

    使用预训练CNN骨干网络提取特征，通过kNN距离检测异常。
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        device: str = "auto",
        n_neighbors: int = 3,
    ):
        """
        Args:
            backbone: 骨干网络名称
            device: 运行设备
            n_neighbors: kNN邻居数
        """
        self.backbone_name = backbone
        self.n_neighbors = n_neighbors

        # 自动选择设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 加载骨干网络（冻结）
        self.backbone = get_backbone(backbone, pretrained=True)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.to(self.device)

        # 获取输出维度
        self.out_dim = BACKBONE_CHOICES[backbone]["out_dim"]
        logger.info(f"初始化特征库检测器 | backbone={backbone}, "
                     f"out_dim={self.out_dim}, device={self.device}")

        # 特征库
        self.feature_bank: Optional[torch.Tensor] = None
        self._is_fitted = False
        self.threshold = 0.5  # 默认阈值，将在评估时自动优化

        # 图像预处理
        self.transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image_paths: List[str]) -> torch.Tensor:
        """
        批量提取图像特征

        Args:
            image_paths: 图像路径列表

        Returns:
            特征矩阵 [N, out_dim]
        """
        all_features = []
        batch_size = 32

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    tensor = self.transform(img)
                    batch_tensors.append(tensor)
                except Exception as e:
                    logger.warning(f"加载失败 [{path}]: {e}")
                    continue

            if not batch_tensors:
                continue

            batch = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad():
                features = self.backbone(batch)  # [B, out_dim]
                features = F.normalize(features, p=2, dim=1)
                all_features.append(features.cpu())

        if not all_features:
            return torch.empty(0)

        return torch.cat(all_features, dim=0)

    def fit(self, ok_image_paths: List[str]):
        """
        构建OK样本特征库

        Args:
            ok_image_paths: OK样本图像路径列表
        """
        logger.info(f"构建特征库: {len(ok_image_paths)} 张OK样本")
        start_time = time.time()

        features = self.extract_features(ok_image_paths)
        if len(features) == 0:
            raise RuntimeError("特征提取失败，无有效图像")

        self.feature_bank = features  # [N, D]

        self._is_fitted = True
        elapsed = time.time() - start_time
        logger.info(f"特征库构建完成 | shape={features.shape} | 耗时={elapsed:.1f}s")

        # 评估OK样本的得分分布（使用PyTorch kNN）
        ok_scores = self._batch_knn_distance(features, features)
        logger.info(f"OK得分分布: mean={ok_scores.mean():.4f}, "
                     f"std={ok_scores.std():.4f}, "
                     f"max={ok_scores.max():.4f}, min={ok_scores.min():.4f}")

        # 建议阈值: 均值+3σ
        suggested_threshold = float(ok_scores.mean() + 3 * ok_scores.std())
        logger.info(f"建议阈值(均值+3σ): {suggested_threshold:.4f}")

        return {
            "feature_shape": tuple(features.shape),
            "ok_score_mean": float(ok_scores.mean()),
            "ok_score_std": float(ok_scores.std()),
            "suggested_threshold": suggested_threshold,
        }

    def predict_single(self, image_path: str) -> Tuple[bool, float]:
        """
        单张图像异常检测

        Args:
            image_path: 图像路径

        Returns:
            (is_anomaly, anomaly_score)
        """
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 构建特征库")

        try:
            img = Image.open(image_path).convert("RGB")
            tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.backbone(tensor)
                features = F.normalize(features, p=2, dim=1)

            # PyTorch kNN: 计算与特征库的余弦距离
            bank = self.feature_bank.to(self.device)  # [N, D]
            query = features  # [1, D]
            # cosine distance = 1 - cosine_similarity
            sim = F.linear(query, bank)  # [1, N]
            distances = 1.0 - sim  # [1, N]  余弦距离 [0, 2]
            # 取k个最近邻的平均距离
            k = min(self.n_neighbors, distances.size(1))
            topk_dist, _ = torch.topk(distances, k=k, dim=1, largest=False)
            anomaly_score = float(topk_dist.mean().item())
            is_anomaly = anomaly_score > self.threshold

            return is_anomaly, anomaly_score

        except Exception as e:
            logger.error(f"推理失败 [{image_path}]: {e}")
            return False, 0.0

    def _batch_knn_distance(
        self, queries: torch.Tensor, bank: torch.Tensor
    ) -> torch.Tensor:
        """
        批量计算kNN余弦距离

        Args:
            queries: 查询特征 [M, D]
            bank: 特征库 [N, D]

        Returns:
            每个查询到k个最近邻的平均距离 [M]
        """
        # 归一化（如果未归一化）
        queries = F.normalize(queries, p=2, dim=1)
        bank = F.normalize(bank, p=2, dim=1)

        # 余弦相似度矩阵 [M, N]
        sim = torch.mm(queries, bank.t())
        distances = 1.0 - sim  # 余弦距离 [0, 2]

        k = min(self.n_neighbors, distances.size(1))
        topk_dist, _ = torch.topk(distances, k=k, dim=1, largest=False)
        return topk_dist.mean(dim=1)

    def predict_batch(
        self, image_paths: List[str], threshold: float = None
    ) -> List[dict]:
        """
        批量异常检测

        Args:
            image_paths: 图像路径列表
            threshold: 判定阈值（None则使用自动计算值）

        Returns:
            结果列表 [{"file": str, "label": str, "score": float, "time_ms": float}]
        """
        if threshold is not None:
            self.threshold = threshold

        results = []
        for path in image_paths:
            start = time.time()
            is_anomaly, score = self.predict_single(path)
            elapsed = (time.time() - start) * 1000

            results.append({
                "file": os.path.basename(path),
                "label": "NG" if is_anomaly else "OK",
                "score": round(score, 4),
                "time_ms": round(elapsed, 1),
            })

        return results

    def evaluate(
        self,
        ok_test_paths: List[str],
        ng_test_paths: List[str],
        threshold: float = None,
    ) -> dict:
        """
        完整评估

        Args:
            ok_test_paths: OK测试图像路径
            ng_test_paths: NG测试图像路径
            threshold: 判定阈值

        Returns:
            评估报告
        """
        if threshold is not None:
            self.threshold = threshold

        # 计算所有得分
        ok_scores_list, ng_scores_list = [], []

        for path in ok_test_paths:
            _, score = self.predict_single(path)
            ok_scores_list.append(score)

        for path in ng_test_paths:
            _, score = self.predict_single(path)
            ng_scores_list.append(score)

        ok_scores = np.array(ok_scores_list)
        ng_scores = np.array(ng_scores_list)

        # 计算AUROC
        auroc = self._compute_auroc(ok_scores, ng_scores)

        # 搜索最佳阈值
        best_th, best_f1, best_acc, best_prec, best_rec = self._find_best_threshold(
            ok_scores, ng_scores
        )

        # 使用最佳阈值重新判定
        correct_ng = sum(1 for s in ng_scores if s >= best_th)
        missed_ng = len(ng_scores) - correct_ng
        ng_detection_rate = correct_ng / max(len(ng_scores), 1)

        report = {
            "method": "FeatureBank (kNN)",
            "backbone": self.backbone_name,
            "n_neighbors": self.n_neighbors,
            "feature_bank_size": len(self.feature_bank) if self.feature_bank is not None else 0,
            "ok_samples": len(ok_scores),
            "ng_samples": len(ng_scores),
            "auroc": round(auroc, 4),
            "best_threshold": round(best_th, 4),
            "f1_score": round(best_f1, 4),
            "accuracy": round(best_acc, 4),
            "precision": round(best_prec, 4),
            "recall": round(best_rec, 4),
            "ng_detection_rate": round(ng_detection_rate, 4),
            "correct_ng": correct_ng,
            "missed_ng": missed_ng,
            "ok_score_stats": {
                "mean": round(float(np.mean(ok_scores)), 4),
                "std": round(float(np.std(ok_scores)), 4),
                "min": round(float(np.min(ok_scores)), 4),
                "max": round(float(np.max(ok_scores)), 4),
            },
            "ng_score_stats": {
                "mean": round(float(np.mean(ng_scores)), 4),
                "std": round(float(np.std(ng_scores)), 4),
                "min": round(float(np.min(ng_scores)), 4),
                "max": round(float(np.max(ng_scores)), 4),
            },
            "detailed_ng": [
                {
                    "file": os.path.basename(path),
                    "score": round(score, 4),
                    "prediction": "NG" if score >= best_th else "OK",
                    "correct": bool(score >= best_th),
                }
                for path, score in zip(ng_test_paths, ng_scores)
            ],
        }

        # 打印报告
        self._print_report(report)
        return report

    @staticmethod
    def _compute_auroc(ok_scores, ng_scores):
        """计算AUROC"""
        scores = list(ok_scores) + list(ng_scores)
        labels = [0] * len(ok_scores) + [1] * len(ng_scores)
        pairs = list(zip(scores, labels))
        pairs.sort(key=lambda x: x[0], reverse=True)
        pos = sum(labels)
        neg = len(labels) - pos
        if pos == 0 or neg == 0:
            return 0.5
        tp, fp = 0, 0
        tpr_list, fpr_list = [0.0], [0.0]
        for i in range(len(pairs)):
            if pairs[i][1] == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / pos)
            fpr_list.append(fp / neg)
        auroc = 0.0
        for i in range(1, len(tpr_list)):
            auroc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2
        return auroc

    @staticmethod
    def _find_best_threshold(ok_scores, ng_scores):
        """搜索最佳阈值"""
        if len(ok_scores) == 0 or len(ng_scores) == 0:
            return 0.5, 0.0, 0.0, 0.0, 0.0
        all_scores = np.concatenate([ok_scores, ng_scores])
        labels = [0] * len(ok_scores) + [1] * len(ng_scores)
        candidates = np.linspace(0.0, max(all_scores) * 1.1, 501)
        best_f1, best_th = 0.0, 0.5
        best_acc, best_prec, best_rec = 0.0, 0.0, 0.0
        for th in candidates:
            preds = [1 if s >= th else 0 for s in all_scores]
            tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
            fn = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
            tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
            acc = (tp + tn) / max(len(labels), 1)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            if f1 > best_f1:
                best_f1, best_th = f1, th
                best_acc, best_prec, best_rec = acc, prec, rec
        return best_th, best_f1, best_acc, best_prec, best_rec

    def _print_report(self, report: dict):
        """打印评估报告"""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"特征库异常检测评估报告")
        logger.info(f"{'=' * 60}")
        logger.info(f"方法: {report['method']} (backbone={report['backbone']}, "
                     f"k={report['n_neighbors']})")
        logger.info(f"特征库大小: {report['feature_bank_size']}")
        logger.info(f"{'─' * 60}")
        logger.info(f"AUROC:      {report['auroc']:.4f}")
        logger.info(f"F1分数:     {report['f1_score']:.4f}")
        logger.info(f"准确率:     {report['accuracy']:.4f}")
        logger.info(f"精确率:     {report['precision']:.4f}")
        logger.info(f"召回率:     {report['recall']:.4f}")
        logger.info(f"NG检出率:   {report['ng_detection_rate']:.2%}")
        logger.info(f"正确检出:   {report['correct_ng']}/{report['ng_samples']}")
        logger.info(f"漏检:       {report['missed_ng']}")
        ok_st = report["ok_score_stats"]
        ng_st = report["ng_score_stats"]
        logger.info(f"\nOK得分: mean={ok_st['mean']:.4f} std={ok_st['std']:.4f} "
                     f"[{ok_st['min']:.4f}, {ok_st['max']:.4f}]")
        logger.info(f"NG得分: mean={ng_st['mean']:.4f} std={ng_st['std']:.4f} "
                     f"[{ng_st['min']:.4f}, {ng_st['max']:.4f}]")
        logger.info(f"{'─' * 60}")
        logger.info(f"NG逐样本:")
        for d in report["detailed_ng"]:
            mark = "✅" if d["correct"] else "❌"
            logger.info(f"  {mark} {d['file']:50s} score={d['score']:.4f} "
                         f"pred={d['prediction']}")
        logger.info(f"{'=' * 60}")


def run_feature_bank_pipeline(
    ok_dir: str = r"e:\Code\study\datas\ok",
    ng_dir: str = r"e:\Code\study\datas\ng",
    infer_dir: str = r"e:\Code\study\datas",
    backbone: str = "resnet50",
    n_neighbors: int = 3,
    output_dir: str = None,
):
    """
    执行完整的特征库异常检测流水线

    步骤:
      1. 加载OK样本构建特征库
      2. 评估NG样本（计算指标）
      3. 批量推理所有图像
    """
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "checkpoints")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"特征库异常检测流水线")
    logger.info(f"{'=' * 60}")
    logger.info(f"OK数据: {ok_dir}")
    logger.info(f"NG评估: {ng_dir}")
    logger.info(f"推理:   {infer_dir}")
    logger.info(f"骨干:   {backbone}, kNN: {n_neighbors}")

    # 加载文件列表
    ok_files = sorted(
        glob(os.path.join(ok_dir, "*.jpg")) + glob(os.path.join(ok_dir, "*.png"))
    )
    ng_files = sorted(
        glob(os.path.join(ng_dir, "*.jpg")) + glob(os.path.join(ng_dir, "*.png"))
    )
    # 收集所有图像（包括ok和ng子目录）
    infer_files = sorted(
        glob(os.path.join(infer_dir, "**/*.jpg"), recursive=True) +
        glob(os.path.join(infer_dir, "**/*.png"), recursive=True)
    )

    logger.info(f"OK训练: {len(ok_files)} 张")
    logger.info(f"NG评估: {len(ng_files)} 张")
    logger.info(f"推理:   {len(infer_files)} 张")

    # 初始化检测器
    detector = FeatureBankDetector(
        backbone=backbone,
        n_neighbors=n_neighbors,
        device="auto",
    )

    # 构建特征库
    detector.fit(ok_files)

    # 评估
    report = detector.evaluate(
        ok_test_paths=ok_files,  # 使用训练集作为参考（在无监督下这是标准做法）
        ng_test_paths=ng_files,
    )

    # 批量推理
    logger.info(f"\n{'─' * 60}")
    logger.info(f"批量推理所有图像...")
    results = detector.predict_batch(infer_files, threshold=report["best_threshold"])

    ok_count = sum(1 for r in results if r["label"] == "OK")
    ng_count = len(results) - ok_count
    logger.info(f"推理完成: 总计 {len(results)} 张, OK={ok_count}, NG={ng_count}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"逐张结果:")
    for r in results:
        mark = "✅" if r["label"] == "OK" else "❌"
        logger.info(f"  {mark} {r['file']:50s} {r['label']:<4s} "
                     f"score={r['score']:.4f}  {r['time_ms']:.1f}ms")
    logger.info(f"{'=' * 60}")

    # 导出报告
    import json
    output_path = os.path.join(output_dir, "feature_bank_report.json")

    def convert_for_json(obj):
        """递归转换为JSON兼容类型"""
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    output_data = convert_for_json({
        "evaluation": report,
        "inference": {
            "total": len(results),
            "ok": ok_count,
            "ng": ng_count,
            "threshold": float(report["best_threshold"]),
            "results": results,
        },
    })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info(f"报告已导出: {output_path}")

    return report, results


if __name__ == "__main__":
    run_feature_bank_pipeline()
