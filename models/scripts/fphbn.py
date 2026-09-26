"""FPHBN의 핵심을 유지한 간단한 구현. 원 논문의 학습 조건까지 재현하지 않는다."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG16_Weights, vgg16


def boosting_loss(sides, target, pos_weight):
    """얕은→깊은 예측 목록. 바로 위(깊은) 단계가 틀린 픽셀에 더 집중한다."""
    losses = []
    for level, logits in enumerate(sides):
        weight = torch.ones_like(target)
        if level + 1 < len(sides):
            # 가중치를 줄이는 방향으로 상위 예측이 학습되지 않도록 detach한다.
            weight = (sides[level + 1].detach().sigmoid() - target).abs()
        pixel_loss = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight, reduction='none',
        )
        losses.append((weight * pixel_loss).mean())
    return torch.stack(losses).mean()


class FPHBN(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        features = vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None).features
        # VGG16의 conv1~conv5. pooling은 forward에서 단계 사이에 네 번 수행한다.
        self.encoders = nn.ModuleList([
            features[0:4],    # 3 → 64, 64
            features[5:9],    # 64 → 128, 128
            features[10:16],  # 128 → 256, 256, 256
            features[17:23],  # 256 → 512, 512, 512
            features[24:30],  # 512 → 512, 512, 512
        ])
        # 얕은 특징과 확대된 상위 특징을 concat한 뒤 1×1 Conv로 채널을 줄인다.
        self.merges = nn.ModuleList([
            nn.Conv2d(64 + 128, 64, 1),
            nn.Conv2d(128 + 256, 128, 1),
            nn.Conv2d(256 + 512, 256, 1),
            nn.Conv2d(512 + 512, 512, 1),
        ])
        self.side_heads = nn.ModuleList(nn.Conv2d(c, 1, 1) for c in (64, 128, 256, 512, 512))
        # 얕은 단계까지 반복 결합하므로 새 projection/head를 작게 초기화한다.
        # 큰 초기 logits로 시작하는 현상을 줄이며, 체크포인트 로딩값은 바꾸지 않는다.
        for layer in list(self.merges) + list(self.side_heads):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.zeros_(layer.bias)
        self.fuse = nn.Conv2d(5, 1, 1)
        nn.init.constant_(self.fuse.weight, 0.2)  # 처음에는 다섯 예측의 평균
        nn.init.zeros_(self.fuse.bias)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, return_sides=False):
        size = x.shape[-2:]
        x = (x * 0.5 + 0.5 - self.mean) / self.std  # [-1,1] → ImageNet 정규화
        features = []
        for level, encoder in enumerate(self.encoders):
            if level:
                x = F.max_pool2d(x, 2)
            x = encoder(x)
            features.append(x)

        # 5→4→3→2→1 순서로 의미 정보를 전달한다.
        for level in range(3, -1, -1):
            upper = F.interpolate(features[level + 1], size=features[level].shape[-2:],
                                  mode='bilinear', align_corners=False)
            features[level] = F.relu(self.merges[level](torch.cat([features[level], upper], dim=1)))
        sides = [F.interpolate(head(feature), size=size, mode='bilinear', align_corners=False)
                 for head, feature in zip(self.side_heads, features)]
        fused = self.fuse(torch.cat(sides, dim=1))
        return (fused, sides) if return_sides else fused

    def training_loss(self, images, target, criterion):
        fused, sides = self(images, return_sides=True)
        # 기존 BCE+Dice와 pos_weight를 유지하고, 단계별 weighted BCE 평균을 추가한다.
        return criterion(fused, target) + boosting_loss(sides, target, criterion.pos_weight)
