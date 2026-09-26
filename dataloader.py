from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def image_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'Image directory not found: {directory}')
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError(f'No images in {directory}')
    return files

class TrainImageTransforms:
    def __call__(self, image):
        array = np.array(image, dtype=np.float32, copy=True) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).sub(0.5).div(0.5)

TestImageTransforms = TrainImageTransforms

class MaskTransforms:
    def __call__(self, image):
        array = np.asarray(image)
        if array.ndim == 3:
            array = array[:, :, 0]
        threshold = 0 if array.max() <= 1 else 127
        return torch.from_numpy((array > threshold).astype(np.float32)).unsqueeze(0)

class CrackDataSet(Dataset):
    def __init__(self, image_dir, mask_dir, image_transforms=None, mask_transforms=None,
                 size=256, augment=False):
        self.images = image_files(image_dir)
        masks = image_files(mask_dir)
        by_stem = {}
        for path in masks:
            if path.stem in by_stem:
                raise ValueError(f'Duplicate mask stem: {path.stem}')
            by_stem[path.stem] = path
        if len({p.stem for p in self.images}) != len(self.images):
            raise ValueError('Duplicate image stems')
        missing = [p.name for p in self.images if p.stem not in by_stem]
        if missing:
            raise ValueError(f'Missing masks: {missing[:10]}')
        self.masks = [by_stem[p.stem] for p in self.images]
        self.fnames = [p.name for p in self.images]
        self.img_transforms = image_transforms or TrainImageTransforms()
        self.mask_transforms = mask_transforms or MaskTransforms()
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with Image.open(self.images[index]) as source:
            image = source.convert('RGB')
        with Image.open(self.masks[index]) as source:
            mask = source.convert('L')
        if image.size != mask.size:
            raise ValueError(f'Image/mask size mismatch: {self.images[index]}')
        if self.size:
            image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
            mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        if self.augment:
            for operation in (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM):
                if torch.rand(()) < 0.5:
                    image, mask = image.transpose(operation), mask.transpose(operation)
        return self.img_transforms(image), self.mask_transforms(mask)
