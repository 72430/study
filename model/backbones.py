"""
骨干网络模块 - 提供多种预训练特征提取器
支持: ResNet18/34/50, WideResNet50_2
"""
import torch
import torch.nn as nn
import torchvision.models as models


# 可用的骨干网络列表
BACKBONE_CHOICES = {
    "resnet18": {"out_dim": 512, "channels": 512},
    "resnet34": {"out_dim": 512, "channels": 512},
    "resnet50": {"out_dim": 2048, "channels": 2048},
    "wide_resnet50_2": {"out_dim": 2048, "channels": 2048},
}


def get_backbone(name: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """
    获取指定名称的骨干网络（去掉最后的全连接层）

    Args:
        name: 骨干网络名称
        pretrained: 是否使用ImageNet预训练权重

    Returns:
        去掉分类头的特征提取网络
    """
    name = name.lower()
    if name not in BACKBONE_CHOICES:
        raise ValueError(
            f"不支持的骨干网络: {name}，可选: {list(BACKBONE_CHOICES.keys())}"
        )

    # 加载预训练模型
    model_fn = getattr(models, name, None)
    if model_fn is None:
        raise ValueError(f"torchvision中找不到模型: {name}")

    backbone = model_fn(weights="DEFAULT" if pretrained else None)

    # 去掉全连接层，只保留特征提取部分
    if hasattr(backbone, "fc"):
        backbone.fc = nn.Identity()
    elif hasattr(backbone, "classifier"):
        backbone.classifier = nn.Identity()

    return backbone


class MultiScaleFeatureExtractor(nn.Module):
    """
    多尺度特征提取器
    从骨干网络的不同层次提取特征，用于更精细的异常检测
    """

    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = get_backbone(backbone_name, pretrained)

        # 注册钩子以获取中间层特征
        self.features = {}
        self._register_hooks()

    def _register_hooks(self):
        """注册前向钩子以捕获中间层特征"""
        # ResNet系列的层次结构: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4
        layer_names = ["layer1", "layer2", "layer3", "layer4"]
        for name in layer_names:
            if hasattr(self.backbone, name):
                layer = getattr(self.backbone, name)
                # 为每一层注册钩子
                layer.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        """创建前向钩子"""

        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，提取多尺度特征

        Args:
            x: 输入图像张量 [B, C, H, W]

        Returns:
            全局特征向量 [B, D]
        """
        self.features = {}
        # 完整前向传播（钩子会自动捕获中间层特征）
        out = self.backbone(x)
        return out

    def get_multi_scale_features(self, x: torch.Tensor) -> dict:
        """
        获取多尺度特征

        Returns:
            包含各层特征的字典: {layer_name: feature_tensor}
        """
        self.forward(x)
        return self.features
