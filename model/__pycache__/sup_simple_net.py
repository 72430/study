"""
SuperSimpleNet 无监督缺陷检测模型实现

核心思想:
  1. 使用预训练CNN骨干网络提取图像的多尺度特征
  2. 通过一个轻量级映射网络将特征投影到低维嵌入空间
  3. 训练时仅使用OK样本，学习紧凑的正常特征分布
  4. 推理时计算样本与正常分布的偏离程度作为异常分数

Reference:
  SuperSimpleNet: Simple and Effective Unsupervised Anomaly Detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbones import get_backbone, BACKBONE_CHOICES


class FeatureMappingNet(nn.Module):
    """特征映射网络 - 将骨干网络提取的特征映射到低维嵌入空间"""

    def __init__(self, in_dim: int, embedding_dim: int = 128):
        """
        Args:
            in_dim: 骨干网络输出特征维度
            embedding_dim: 嵌入向量维度
        """
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将特征映射到嵌入空间并L2归一化"""
        x = self.mapping(x)
        # L2 归一化，将嵌入向量约束在单位超球面上
        x = F.normalize(x, p=2, dim=1)
        return x


class SuperSimpleNet(nn.Module):
    """
    SuperSimpleNet 主模型

    包含:
      - 骨干网络（冻结）：特征提取
      - 特征映射网络（训练）：嵌入学习
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 128,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim

        # 1. 骨干网络 - 冻结BN和参数
        self.backbone = get_backbone(backbone, pretrained)
        self._freeze_backbone()

        # 2. 获取骨干网络输出维度
        out_dim = BACKBONE_CHOICES[backbone]["out_dim"]

        # 3. 特征映射网络 - 可训练
        self.mapping_net = FeatureMappingNet(out_dim, embedding_dim)

    def _freeze_backbone(self):
        """冻结骨干网络的所有参数（仅用于特征提取）"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # 将骨干网络设为eval模式以固定BN统计量
        self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 [B, C, H, W]

        Returns:
            归一化的嵌入向量 [B, embedding_dim]
        """
        # 提取特征
        with torch.no_grad():
            features = self.backbone(x)
        # 映射到嵌入空间
        embeddings = self.mapping_net(features)
        return embeddings

    def get_anomaly_score(
        self, x: torch.Tensor, reference_embeddings: torch.Tensor = None
    ) -> torch.Tensor:
        """
        计算异常得分

        Args:
            x: 输入图像 [B, C, H, W]
            reference_embeddings: 参考正常样本的嵌入中心 [1, embedding_dim]
                                 如果为None，使用batch内的平均嵌入

        Returns:
            anomaly_scores: 异常得分 [B]，值越大表示越异常
        """
        embeddings = self.forward(x)

        if reference_embeddings is None:
            # 使用当前batch的平均嵌入作为参考
            reference_embeddings = embeddings.mean(dim=0, keepdim=True)

        # 计算余弦距离作为异常得分
        # 距离越大越异常（因为正常样本应紧凑分布在参考点周围）
        similarity = F.cosine_similarity(
            embeddings.unsqueeze(1),  # [B, 1, D]
            reference_embeddings.unsqueeze(0),  # [1, 1, D]
            dim=2,
        )  # [B, 1]
        # 余弦相似度[-1,1] -> 异常得分[0,1]
        anomaly_scores = (1.0 - similarity.squeeze(1)) / 2.0
        return anomaly_scores

    def set_train_mode(self):
        """设置模型为训练模式（仅解冻映射网络）"""
        self.backbone.eval()  # 骨干网络始终保持eval
        self.mapping_net.train()

    def set_eval_mode(self):
        """设置模型为推理模式"""
        self.backbone.eval()
        self.mapping_net.eval()
