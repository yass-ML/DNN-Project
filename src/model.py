"""
FaceNet Model Architecture: Inception-ResNet-v1

Implementation based on "FaceNet: A Unified Embedding for Face Recognition and Clustering"
(Schroff et al., 2015) - https://arxiv.org/abs/1503.03832

The model outputs 128-dimensional embeddings that are L2-normalized to lie on a unit hypersphere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    """Basic convolution block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Stem(nn.Module):
    """
    Stem block for Inception-ResNet-v1.
    Input: 3 x 160 x 160
    Output: 256 x 35 x 35
    """
    
    def __init__(self):
        super().__init__()
        self.conv1 = BasicConv2d(3, 32, kernel_size=3, stride=2)  # 79x79
        self.conv2 = BasicConv2d(32, 32, kernel_size=3)  # 77x77
        self.conv3 = BasicConv2d(32, 64, kernel_size=3, padding=1)  # 77x77
        self.pool1 = nn.MaxPool2d(3, stride=2)  # 38x38
        self.conv4 = BasicConv2d(64, 80, kernel_size=1)  # 38x38
        self.conv5 = BasicConv2d(80, 192, kernel_size=3)  # 36x36
        self.conv6 = BasicConv2d(192, 256, kernel_size=3, stride=2)  # 17x17
        # Additional layer to match expected dimensions
        self.conv7 = BasicConv2d(256, 256, kernel_size=3, padding=1)  # 17x17
        # Upsample to 35x35 for Inception-ResNet-A compatibility
        self.upsample = nn.Upsample(size=(35, 35), mode='bilinear', align_corners=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.upsample(x)
        return x


class InceptionResNetA(nn.Module):
    """
    Inception-ResNet-A block.
    Input/Output: 256 x 35 x 35
    """
    
    def __init__(self, scale: float = 0.17):
        super().__init__()
        self.scale = scale
        
        # Branch 1: 1x1 conv
        self.branch1 = BasicConv2d(256, 32, kernel_size=1)
        
        # Branch 2: 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1)
        )
        
        # Branch 3: 1x1 -> 3x3 -> 3x3
        self.branch3 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1)
        )
        
        # 1x1 conv to match residual dimensions (no ReLU)
        self.conv = nn.Conv2d(96, 256, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(256, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        
        out = identity + self.scale * out
        out = self.relu(out)
        return out


class ReductionA(nn.Module):
    """
    Reduction-A block.
    Input: 256 x 35 x 35
    Output: 896 x 17 x 17
    """
    
    def __init__(self):
        super().__init__()
        
        # Branch 1: 3x3 conv stride 2 (valid)
        self.branch1 = BasicConv2d(256, 384, kernel_size=3, stride=2)
        
        # Branch 2: 1x1 -> 3x3 -> 3x3 stride 2
        self.branch2 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=3, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )
        
        # Branch 3: max pool stride 2
        self.branch3 = nn.MaxPool2d(3, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1)


class InceptionResNetB(nn.Module):
    """
    Inception-ResNet-B block.
    Input/Output: 896 x 17 x 17
    """
    
    def __init__(self, scale: float = 0.10):
        super().__init__()
        self.scale = scale
        
        # Branch 1: 1x1 conv
        self.branch1 = BasicConv2d(896, 128, kernel_size=1)
        
        # Branch 2: 1x1 -> 1x7 -> 7x1
        self.branch2 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1),
            BasicConv2d(128, 128, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(128, 128, kernel_size=(7, 1), padding=(3, 0))
        )
        
        # 1x1 conv to match residual dimensions
        self.conv = nn.Conv2d(256, 896, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(896, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        
        out = torch.cat([b1, b2], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        
        out = identity + self.scale * out
        out = self.relu(out)
        return out


class ReductionB(nn.Module):
    """
    Reduction-B block.
    Input: 896 x 17 x 17
    Output: 1792 x 8 x 8
    """
    
    def __init__(self):
        super().__init__()
        
        # Branch 1: 1x1 -> 3x3 stride 2
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )
        
        # Branch 2: 1x1 -> 3x3 stride 2
        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )
        
        # Branch 3: 1x1 -> 3x3 -> 3x3 stride 2
        self.branch3 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=3, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )
        
        # Branch 4: max pool stride 2
        self.branch4 = nn.MaxPool2d(3, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class InceptionResNetC(nn.Module):
    """
    Inception-ResNet-C block.
    Input/Output: 1792 x 8 x 8
    """
    
    def __init__(self, scale: float = 0.20):
        super().__init__()
        self.scale = scale
        
        # Branch 1: 1x1 conv
        self.branch1 = BasicConv2d(1792, 192, kernel_size=1)
        
        # Branch 2: 1x1 -> 1x3 -> 3x1
        self.branch2 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(192, 192, kernel_size=(3, 1), padding=(1, 0))
        )
        
        # 1x1 conv to match residual dimensions
        self.conv = nn.Conv2d(384, 1792, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1792, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        
        out = torch.cat([b1, b2], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        
        out = identity + self.scale * out
        out = self.relu(out)
        return out


class InceptionResNetV1(nn.Module):
    """
    Inception-ResNet-v1 for FaceNet.
    
    Produces 128-dimensional face embeddings that are L2-normalized
    to lie on a unit hypersphere (||f(x)||_2 = 1).
    
    Args:
        embedding_dim: Dimension of output embedding (default: 128)
        dropout_prob: Dropout probability (default: 0.2)
        pretrained: Path to pretrained weights (default: None)
    
    Input: (N, 3, 160, 160) - RGB face images
    Output: (N, 128) - L2-normalized embeddings
    """
    
    def __init__(self, embedding_dim: int = 128, dropout_prob: float = 0.2,
                 pretrained: str = None):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Stem
        self.stem = Stem()
        
        # 5x Inception-ResNet-A
        self.inception_a = nn.Sequential(*[InceptionResNetA() for _ in range(5)])
        
        # Reduction-A
        self.reduction_a = ReductionA()
        
        # 10x Inception-ResNet-B
        self.inception_b = nn.Sequential(*[InceptionResNetB() for _ in range(10)])
        
        # Reduction-B
        self.reduction_b = ReductionB()
        
        # 5x Inception-ResNet-C
        self.inception_c = nn.Sequential(*[InceptionResNetC() for _ in range(5)])
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Embedding layer (bottleneck)
        self.embedding = nn.Linear(1792, embedding_dim, bias=False)
        
        # Batch normalization for embedding
        self.bn = nn.BatchNorm1d(embedding_dim, eps=0.001, momentum=0.1)
        
        # Initialize weights
        self._initialize_weights()
        
        # Load pretrained weights if provided
        if pretrained is not None:
            self.load_pretrained(pretrained)
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
    
    def load_pretrained(self, path: str):
        """Load pretrained weights from file."""
        state_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {path}")
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (N, 3, 160, 160)
            return_features: If True, return pre-normalized features
        
        Returns:
            L2-normalized embeddings of shape (N, embedding_dim)
        """
        # Backbone
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Dropout
        x = self.dropout(x)
        
        # Embedding
        x = self.embedding(x)
        x = self.bn(x)
        
        if return_features:
            return x
        
        # L2 normalization - ensures ||f(x)||_2 = 1
        # This constrains embeddings to lie on a unit hypersphere
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward pass - generates L2-normalized embedding."""
        return self.forward(x)


def get_model(embedding_dim: int = 128, pretrained: str = None) -> InceptionResNetV1:
    """
    Factory function to create InceptionResNetV1 model.
    
    Args:
        embedding_dim: Dimension of output embedding
        pretrained: Path to pretrained weights
    
    Returns:
        InceptionResNetV1 model instance
    """
    return InceptionResNetV1(embedding_dim=embedding_dim, pretrained=pretrained)


if __name__ == "__main__":
    # Test the model
    model = InceptionResNetV1()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    x = torch.randn(2, 3, 160, 160)
    embeddings = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {embeddings.shape}")
    
    # Verify L2 normalization
    norms = torch.norm(embeddings, p=2, dim=1)
    print(f"Embedding L2 norms: {norms}")
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), "Embeddings should be L2 normalized!"
    print("✓ L2 normalization verified!")
