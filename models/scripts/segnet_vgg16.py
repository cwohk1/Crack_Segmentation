"""VGG16 기반 SegNet: 위치를 저장하는 encoder와 역순으로 복원하는 decoder."""
import torch
from torch import nn


def conv_block(in_channels, *out_channels):
    """적힌 채널 순서대로 3×3 Conv → BatchNorm → ReLU를 반복한다."""
    layers = []
    for channels in out_channels:
        layers.extend([
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ])
        in_channels = channels
    return nn.Sequential(*layers)


class SegNetVGG16(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super().__init__()
        # VGG16의 합성곱 13개(2+2+3+3+3). 완전연결층은 사용하지 않는다.
        self.encoders = nn.ModuleList([
            conv_block(3, 64, 64),
            conv_block(64, 128, 128),
            conv_block(128, 256, 256, 256),
            conv_block(256, 512, 512, 512),
            conv_block(512, 512, 512, 512),
        ])
        # 깊은 단계부터 실행한다. 마지막 채널 수를 다음 unpool에 맞춘다.
        self.decoders = nn.ModuleList([
            conv_block(512, 512, 512, 512),
            conv_block(512, 512, 512, 256),
            conv_block(256, 256, 256, 128),
            conv_block(128, 128, 64),
            conv_block(64, 64, 64),
        ])
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

        # 데이터로더의 [-1,1] 입력을 ImageNet 정규화로 바꾼다.
        # 사전학습 여부와 무관하게 같으므로 체크포인트 재로딩도 동일하다.
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if pretrained:
            self.load_pretrained_encoder()

    def load_pretrained_encoder(self):
        """torchvision VGG16-BN의 Conv/BN만 복사한다. decoder는 새로 학습한다."""
        from torchvision.models import VGG16_BN_Weights, vgg16_bn
        features = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT).features
        source_layers = [layer for layer in features if not isinstance(layer, nn.MaxPool2d)]
        target_layers = [layer for block in self.encoders for layer in block]
        for target, source in zip(target_layers, source_layers, strict=True):
            target.load_state_dict(source.state_dict())

    def forward(self, x):
        x = (x * 0.5 + 0.5 - self.mean) / self.std
        indices, sizes = [], []
        for encoder in self.encoders:
            x = encoder(x)
            sizes.append(x.size())  # 홀수 해상도도 정확히 복원하기 위한 크기
            x, index = self.pool(x)
            indices.append(index)

        for decoder, index, size in zip(self.decoders, reversed(indices), reversed(sizes)):
            x = self.unpool(x, index, output_size=size)
            x = decoder(x)  # 저장된 위치에 배치한 뒤 합성곱으로 주변 특징을 채운다.
        # BCEWithLogitsLoss에서 sigmoid를 적용하므로 여기서는 logits를 반환한다.
        return self.classifier(x)
