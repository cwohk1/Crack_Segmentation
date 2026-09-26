"""Evaluate labeled data (--data) or save image predictions (--input)."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from dataloader import CrackDataSet, TestImageTransforms, image_files
from models.scripts import build_model
from trainer import BCE_DiceLoss, evaluate, get_device


def predict(model, source, output, size, device, threshold):
    paths = image_files(source) if source.is_dir() else [source]
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as original:
                image = original.convert('RGB')
            resized = image.resize((size, size), Image.Resampling.BILINEAR)
            tensor = TestImageTransforms()(resized).unsqueeze(0).to(device)
            probabilities = torch.nn.functional.interpolate(
                model(tensor).sigmoid(), size=(image.height, image.width),
                mode='bilinear', align_corners=False,
            )[0, 0].cpu().numpy()
            mask = probabilities >= threshold
            Image.fromarray(mask.astype(np.uint8) * 255).save(output / (path.name + '_mask.png'))
            overlay = np.array(image)
            overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            Image.fromarray(overlay).save(output / (path.name + '_overlay.png'))
    print(f'Saved {len(paths)} predictions to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weight', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data', type=Path, help='Dataset root with test/ or val/ images and masks')
    source.add_argument('--input', type=Path, help='Image or directory to predict without labels')
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output', type=Path, help='Evaluation JSON file or prediction directory')
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error('threshold must be between 0 and 1')
    device = get_device(args.device)
    checkpoint = torch.load(args.weight, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    model = build_model(config['model']).to(device)
    model.load_state_dict(checkpoint['model'])
    run_dir = args.weight.parent
    if args.input:
        predict(model, args.input, args.output or run_dir/'predictions',
                config['size'], device, args.threshold)
        return
    dataset = CrackDataSet(args.data/args.split/'images', args.data/args.split/'masks', size=config['size'])
    result = evaluate(
        model, DataLoader(dataset, batch_size=config['batch_size']),
        BCE_DiceLoss(config['pos_weight']).to(device), device, args.threshold,
    )
    result.update({'images': len(dataset), 'threshold': args.threshold, 'size': config['size'],
                   'checkpoint': str(args.weight), 'split': args.split})
    print(json.dumps(result, indent=2))
    output = args.output or run_dir/(args.split+'.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
