"""Train on train/, select checkpoints on val/ or a held-out train split."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from dataloader import CrackDataSet
from torch import nn
from models.scripts import MODEL_NAMES, build_model


class BCE_DiceLoss(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.register_buffer('pos_weight', torch.tensor(float(pos_weight)))

    def forward(self, logits, target):
        if logits.shape != target.shape:
            raise ValueError(f'Shape mismatch: {logits.shape} vs {target.shape}')
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        probability = logits.sigmoid()
        dims = (1, 2, 3)
        dice = (2 * (probability * target).sum(dims) + 1) / (probability.sum(dims) + target.sum(dims) + 1)
        return bce + 1 - dice.mean()

class Metrics:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def update(self, logits, targets, threshold=0.5):
        prediction, truth = logits.sigmoid() >= threshold, targets >= 0.5
        self.tp += int((prediction & truth).sum())
        self.fp += int((prediction & ~truth).sum())
        self.fn += int((~prediction & truth).sum())

    def compute(self):
        def ratio(n, d):
            return n / d if d else 1.0
        return {'iou': ratio(self.tp, self.tp + self.fp + self.fn),
                'dice': ratio(2*self.tp, 2*self.tp + self.fp + self.fn),
                'precision': ratio(self.tp, self.tp+self.fp),
                'recall': ratio(self.tp, self.tp+self.fn)}

def get_device(value='auto'):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if value == 'auto' else torch.device(value)


def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total, count, metrics = 0.0, 0, Metrics()
    with torch.inference_mode():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            total += criterion(logits, masks).item() * len(images)
            count += len(images)
            metrics.update(logits, masks, threshold)
    if not count:
        raise ValueError('Evaluation set is empty')
    return {'loss': total/count, **metrics.compute()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('datasets/deepcrack/prepared'))
    parser.add_argument('--output', type=Path, help='Run logs directory (default: Codex/runs/<run name>)')
    parser.add_argument('--weights-dir', type=Path, help='Checkpoint directory (default: models/weights/<run name>)')
    parser.add_argument('--model', choices=MODEL_NAMES, default='segnet')
    parser.add_argument('--epochs', type=int, default=30, help='Total target epochs including resumed epochs')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=None, help='AdamW learning rate. Default 1e-3; resume keeps the checkpoint rate unless this is set')
    parser.add_argument('--pos-weight', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.output is None:
        name = args.resume.parent.name if args.resume else 'baseline'
        args.output = Path('Codex/runs') / name
    if args.weights_dir is None:
        args.weights_dir = args.resume.parent if args.resume else Path('models/weights') / args.output.name
    if (args.weights_dir/'last.pt').exists() and not args.resume:
        parser.error('Checkpoints already exist; use --resume or a new output name')
    if args.resume and args.weights_dir.resolve() != args.resume.parent.resolve():
        parser.error('Resume into the original weights directory to preserve best.pt')
    if args.size < 64 or args.size % 32 or args.epochs < 1 or args.batch_size < 1 or args.pos_weight <= 0:
        parser.error('size must be a multiple of 32 and at least 64; epochs, batch-size and pos-weight must be positive')
    if args.lr is not None and args.lr <= 0:
        parser.error('lr must be positive')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = get_device(args.device)
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    if checkpoint:
        for key in ('model', 'size', 'seed', 'augment', 'pos_weight', 'pretrained', 'batch_size'):
            setattr(args, key, checkpoint['config'][key])
    base = CrackDataSet(args.data/'train/images', args.data/'train/masks', size=args.size, augment=args.augment)
    clean = CrackDataSet(args.data/'train/images', args.data/'train/masks', size=args.size)
    if (args.data/'val').is_dir():
        train_set = base
        val_set = CrackDataSet(args.data/'val/images', args.data/'val/masks', size=args.size)
        split = {'train': list(range(len(base))), 'val': None}
    else:
        if len(base) < 2:
            raise ValueError('At least two training images required for a held-out validation split')
        order = torch.randperm(len(base), generator=torch.Generator().manual_seed(args.seed)).tolist()
        nval = max(1, round(len(base)*0.2))
        split = {'train': order[nval:], 'val': order[:nval]}
        train_set, val_set = Subset(base, split['train']), Subset(clean, split['val'])
    manifest = {'train': [p.name for p in base.images], 'val': [p.name for p in val_set.images] if split['val'] is None else None}
    if checkpoint and (checkpoint['manifest'] != manifest or checkpoint['split'] != split):
        raise ValueError('Dataset manifest or validation split differs from checkpoint')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)
    if args.lr is None:
        args.lr = float(checkpoint['optimizer']['param_groups'][0]['lr']) if checkpoint else 1e-3
    model = build_model(args.model, args.pretrained and not checkpoint).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = BCE_DiceLoss(args.pos_weight).to(device)
    history, start, best = [], 1, -1.0
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = args.lr
        history, start, best = checkpoint['history'], checkpoint['epoch']+1, checkpoint['best_iou']
        torch.set_rng_state(checkpoint['rng'].cpu())
        if device.type == 'cuda' and checkpoint.get('cuda_rng'):
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    if checkpoint and args.epochs < start:
        raise ValueError('Target epochs must exceed the resumed epoch')
    args.output.mkdir(parents=True, exist_ok=True)
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    print(f'device={device} train={len(train_set)} val={len(val_set)} model={args.model}', flush=True)
    for epoch in range(start, args.epochs+1):
        model.train()
        total = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            total += loss.item()*len(images)
        validation = evaluate(model, val_loader, criterion, device)
        row = {'epoch': epoch, 'train_loss': total/len(train_set), **{'val_'+k:v for k,v in validation.items()}}
        history.append(row)
        improved = validation['iou'] > best
        best = max(best, validation['iou'])
        state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'history': history, 'best_iou': best, 'config': config, 'manifest': manifest,
                 'split': split, 'rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
        torch.save(state, args.weights_dir/'last.pt')
        if improved:
            torch.save(state, args.weights_dir/'best.pt')
        (args.output/'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)

if __name__ == '__main__':
    main()
