"""Model factory. All binary models return N x 1 x H x W logits."""

MODEL_NAMES = ('segnet', 'segnet_vgg16', 'FPHBN', 'UNet18', 'UNetResNet50', 'UNetVGGNet16')

MODEL_DIRS = {'segnet': 'segnet', 'segnet_vgg16': 'segnet_vgg16', 'FPHBN': 'fphbn',
              'UNet18': 'unet_18', 'UNetResNet50': 'unet_resnet50', 'UNetVGGNet16': 'unet_vgg16'}

def build_model(name, pretrained=False):
    if name == 'FPHBN':
        from .fphbn import FPHBN
        return FPHBN(pretrained=pretrained)
    if name == 'segnet_vgg16':
        from .segnet_vgg16 import SegNetVGG16
        return SegNetVGG16(pretrained=pretrained)
    if name == 'segnet':
        from .segnet import SegNet
        return SegNet()
    if name == 'UNet18':
        from .unet18 import UNet18
        return UNet18(pretrained=pretrained)
    if name == 'UNetResNet50':
        from .unet_resnet50 import UNetResNet50
        return UNetResNet50(pretrained=pretrained)
    if name == 'UNetVGGNet16':
        from .unet_vgg16 import UNet16
        return UNet16(pretrained=pretrained)
    raise ValueError(f'Unknown model: {name}')
