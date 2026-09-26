"""Train on train/, select checkpoints on val/ or a held-out train split."""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from dataloader import CrackDataSet
from torch import nn
from models.scripts import MODEL_NAMES, MODEL_DIRS, build_model


class BCE_DiceLoss(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.register_buffer('pos_weight', torch.tensor(float(pos_weight)))

    def forward(self, logits, target):
        if logits.shape != target.shape:
            raise ValueError(f'Shape mismatch: {logits.shape} vs {target.shape}')
        logits = logits.float()  # BF16 학습에서도 손실과 Dice 합산은 FP32
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


def save_checkpoint(state, path):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='Default: datasets/deepcrack/prepared; resume restores saved data path')
    parser.add_argument('--output', type=Path, help='Directory for weights and results (default: models/weights/<model>/<dataset>)')
    parser.add_argument('--weights-dir', type=Path, help='Alias for --output; weights and results share one directory')
    parser.add_argument('--model', choices=MODEL_NAMES, default='segnet')
    parser.add_argument('--epochs', type=int, default=30, help='Total target epochs including resumed epochs')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=None, help='AdamW learning rate. Default 1e-3; resume keeps the checkpoint rate unless this is set')
    parser.add_argument('--pos-weight', type=float, default=1.0)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=0, help='Early stop after this many epochs without validation IoU improvement; 0 disables')
    parser.add_argument('--lr-patience', type=int, default=0, help='Halve LR on a validation plateau; 0 disables')
    parser.add_argument('--amp', action='store_true', help='CUDA BF16 mixed precision (requires BF16 support)')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-seed', type=int, help='Data split seed; defaults to --seed')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    if checkpoint:
        for key in ('model', 'size', 'seed', 'augment', 'pos_weight', 'pretrained', 'batch_size'):
            setattr(args, key, checkpoint['config'][key])
        for key in ('amp', 'weight_decay'):
            setattr(args, key, checkpoint['config'].get(key, False if key == 'amp' else 0.01))
        args.split_seed = checkpoint['config'].get('split_seed', checkpoint['config']['seed'])
    if args.split_seed is None:
        args.split_seed = args.seed
    if args.data is None:
        args.data = Path(checkpoint['config']['data']) if checkpoint else Path('datasets/deepcrack/prepared')
    if args.output and args.weights_dir and args.output.resolve() != args.weights_dir.resolve():
        parser.error('--output and --weights-dir must refer to the same directory')
    dataset = args.data.parent.name if args.data.name == 'prepared' else args.data.name
    args.output = args.output or args.weights_dir or (args.resume.parent if args.resume else
                  Path('models/weights') / MODEL_DIRS[args.model] / dataset)
    args.weights_dir = args.output
    if (args.weights_dir/'last.pt').exists() and not args.resume:
        parser.error('Checkpoints already exist; use --resume or a new output name')
    if args.resume and args.weights_dir.resolve() != args.resume.parent.resolve():
        parser.error('Resume into the original weights directory to preserve best.pt')
    if args.size < 64 or args.size % 32 or args.epochs < 1 or args.batch_size < 1 or args.pos_weight <= 0:
        parser.error('size must be a multiple of 32 and at least 64; epochs, batch-size and pos-weight must be positive')
    if args.lr is not None and args.lr <= 0:
        parser.error('lr must be positive')
    if min(args.patience, args.lr_patience, args.weight_decay) < 0:
        parser.error('patience, lr-patience and weight-decay must be non-negative')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = get_device(args.device)
    if args.amp and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        parser.error('--amp requires a CUDA GPU with BF16 support')
    base = CrackDataSet(args.data/'train/images', args.data/'train/masks', size=args.size, augment=args.augment)
    clean = CrackDataSet(args.data/'train/images', args.data/'train/masks', size=args.size)
    if (args.data/'val').is_dir():
        train_set = base
        val_set = CrackDataSet(args.data/'val/images', args.data/'val/masks', size=args.size)
        split = {'train': list(range(len(base))), 'val': None}
    else:
        if len(base) < 2:
            raise ValueError('At least two training images required for a held-out validation split')
        order = torch.randperm(len(base), generator=torch.Generator().manual_seed(args.split_seed)).tolist()
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                patience=args.lr_patience, threshold=0, min_lr=1e-6) if args.lr_patience else None
    criterion = BCE_DiceLoss(args.pos_weight).to(device)
    history, start, best = [], 1, -1.0
    bad_epochs = 0
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler and checkpoint.get('scheduler'):
            scheduler.load_state_dict(checkpoint['scheduler'])
            scheduler.patience = args.lr_patience
        bad_epochs = checkpoint.get('bad_epochs', 0)
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
        epoch_start = time.monotonic()
        model.train()
        total = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
                if hasattr(model, 'training_loss'):
                    loss = model.training_loss(images, masks, criterion)
                else:
                    loss = criterion(model(images), masks)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()*len(images)
        validation = evaluate(model, val_loader, criterion, device)
        row = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 'seconds': time.monotonic()-epoch_start, 'train_loss': total/len(train_set), **{'val_'+k:v for k,v in validation.items()}}
        history.append(row)
        improved = validation['iou'] > best
        best = max(best, validation['iou'])
        bad_epochs = 0 if improved else bad_epochs + 1
        if scheduler:
            scheduler.step(validation['iou'])
        state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'history': history, 'best_iou': best, 'bad_epochs': bad_epochs,
                 'scheduler': scheduler.state_dict() if scheduler else None, 'config': config, 'manifest': manifest,
                 'split': split, 'rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
        save_checkpoint(state, args.weights_dir/'last.pt')
        if improved:
            save_checkpoint(state, args.weights_dir/'best.pt')
        (args.output/'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
        stopped = bool(args.patience and bad_epochs >= args.patience)
        (args.output/'training.json').write_text(json.dumps({'epoch': epoch, 'target_epochs': args.epochs,
            'early_stopped': stopped, 'best_val_iou': best, 'finished': stopped or epoch == args.epochs}, indent=2))
        if stopped:
            print(f'Early stopping at epoch {epoch}', flush=True)
            break


def tune():
    """저장된 계획대로 비교 → 상위 조건 연장 → 다른 seed 확인을 순차 실행한다."""
    import hashlib
    import subprocess
    import sys
    parser = argparse.ArgumentParser(description='Repeatable DeepCrack tuning for all registered models')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--plan-only', action='store_true', help='Write model plans without training')
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument('--upload', action='store_true', help='Upload completed trials and plans to the model repo')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data = root/'datasets/deepcrack/prepared'
    sources = [root/'trainer.py', root/'dataloader.py', *sorted((root/'models/scripts').glob('*.py'))]
    code_hash = hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()
    env = {**__import__('os').environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'HF_HUB_DISABLE_PROGRESS_BARS': '1'}

    def write(path, value):
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

    def command(arguments, log):
        with log.open('a', encoding='utf-8') as output:
            output.write('\nCOMMAND '+ ' '.join(map(str, arguments))+'\n'); output.flush()
            subprocess.run([sys.executable, *map(str, arguments)], cwd=root, env=env,
                           stdout=output, stderr=subprocess.STDOUT, check=True)

    for model in args.models:
        folder = root/'models/weights'/MODEL_DIRS[model]
        folder.mkdir(parents=True, exist_ok=True)
        plan_path = folder/'training-plan.json'
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if plan['code_sha256'] != code_hash:
                raise ValueError(f'Source changed since plan creation: {plan_path}; review before mixing runs')
        else:
            rates = [0.0001, 0.0003] if model == 'FPHBN' else [0.0003, 0.001]
            trials = [{'name': f'tune_v1_lr{lr:g}_pw{pw}_s42'.replace('.', 'p'),
                       'lr': lr, 'pos_weight': pw, 'seed': 42} for lr in rates for pw in (1, 3)]
            plan = {'model': model, 'dataset': 'deepcrack', 'code_sha256': code_hash,
                    'screen_epochs': 8, 'promote_epochs': 30, 'final_epochs': 60, 'repeat_epochs': 40,
                    'converge_chunk_epochs': 40,
                    'patience': 12, 'lr_patience': 4, 'batch_size': 4, 'size': 256,
                    'amp': True, 'pretrained': False, 'weight_decay': 0.01, 'trials': trials,
                    'selection': 'best validation IoU; test only after final selection',
                    'split': 'seed 42: train 240 / val 60; official test 237'}
            write(plan_path, plan)
        (folder/'training-plan.md').write_text(
            f"# {model}: DeepCrack 학습계획\n\n"
            "- 원본 train 300장 중 seed 42로 train 240 / val 60을 고정. 공식 test 237장은 최종 평가만 수행.\n"
            "- 256×256, batch 4, 좌우·상하 반전, AdamW, weight decay 0.01, CUDA BF16.\n"
            "- 모든 모델은 사전학습 없이 시작. 원본 성능 재현이나 사전학습 모델 비교가 아님.\n"
            "- 4조건(학습률 2개 × pos_weight 1/3)을 8 epoch 비교 → 상위 2개 최대 30 → 최상위 최대 60.\n"
            "- 검증 IoU 4 epoch 정체 후 학습률 절반, 12 epoch 개선 없으면 조기 종료. gradient norm 최대 5.\n"
            "- 우수 조건을 seed 1337로 최대 40 epoch 재학습. 데이터 분할 seed는 42 유지.\n"
            "- 우수 조건의 두 seed 실행은 40 epoch 단위로 계속 재개하고, 12 epoch 연속 검증 개선이 없을 때 종료. 시간/총 epoch 제한 없음.\n"
            "- 최종 조건/seed도 검증 IoU로 선택한 뒤 best.pt로 test.json 생성. threshold 0.5 고정.\n"
            "- 조건별 best/last, config, history, train.log, training.json, summary.json, plan.json 보존.\n"
            "- 실패는 failure.json에 기록하고 다른 모델은 계속 실행. 기존 baseline은 덮어쓰지 않음.\n"
            "- 재실행: python trainer.py --tune --upload (완료 단계 건너뛰기).\n"
            "- 학습률 후보와 정확한 값은 training-plan.json 참조. 모든 조건 소진은 전역 최적 보장이 아님.\n",
            encoding='utf-8')
        if args.plan_only:
            print(f'PLAN {plan_path}', flush=True)
            continue
        base = folder/'deepcrack'; base.mkdir(exist_ok=True)
        summary_path = base/'tuning-summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {'model': model, 'stages': {}, 'trials': {}}

        summary['status'] = 'running'; summary['uploaded'] = False
        summary.pop('error', None)
        write(summary_path, summary)

        def train(trial, epochs):
            name = trial['name']; out = base/name; out.mkdir(exist_ok=True)
            write(out/'plan.json', plan)
            progress = json.loads((out/'training.json').read_text()) if (out/'training.json').exists() else {}
            if not (progress.get('early_stopped') or progress.get('epoch', 0) >= epochs):
                print(f"TRAIN {model} {name} target={epochs}", flush=True)
                cmd = ['trainer.py', '--data', 'datasets/deepcrack/prepared', '--output', out,
                       '--epochs', epochs, '--patience', plan['patience'], '--lr-patience', plan['lr_patience'],
                       '--device', 'cuda', '--workers', 2]
                if (out/'last.pt').exists():
                    cmd += ['--resume', out/'last.pt']
                else:
                    cmd += ['--model', model, '--lr', trial['lr'], '--pos-weight', trial['pos_weight'],
                            '--seed', trial['seed'], '--split-seed', 42,
                            '--batch-size', plan['batch_size'], '--size', plan['size'], '--augment', '--amp']
                command(cmd, out/'train.log')
            history = json.loads((out/'history.json').read_text())
            best = max(history, key=lambda row: row['val_iou'])
            result = {**trial, 'epochs': history[-1]['epoch'], 'best_epoch': best['epoch'],
                      'best_val_iou': best['val_iou'], 'best_val_dice': best['val_dice'],
                      'training_seconds': sum(row.get('seconds', 0) for row in history)}
            summary['trials'][name] = result
            write(out/'summary.json', result); write(summary_path, summary)
            return result

        try:
            for key, count, epochs in [('screen', 4, plan['screen_epochs']),
                                        ('promote', 2, plan['promote_epochs']), ('final', 1, plan['final_epochs'])]:
                if key in summary['stages']:
                    continue
                candidates = plan['trials'] if key == 'screen' else sorted(
                    [summary['trials'][t['name']] for t in plan['trials']], key=lambda t: t['best_val_iou'], reverse=True)[:count]
                for trial in candidates:
                    train(trial, epochs)
                summary['stages'][key] = [t['name'] for t in candidates]; write(summary_path, summary)
            if 'repeat' not in summary['stages']:
                winner = max(summary['trials'].values(), key=lambda t:t['best_val_iou'])
                repeat = {**winner, 'name': winner['name'].replace('_s42', '_s1337'), 'seed': 1337}
                train(repeat, plan['repeat_epochs'])
                summary['stages']['repeat'] = [repeat['name']]; write(summary_path, summary)
            if 'converge' not in summary['stages']:
                # 시간 제한 대신 검증 정체로 종료한다. 중단 후에는 last.pt에서 이어간다.
                names = list(dict.fromkeys(summary['stages']['final'] + summary['stages']['repeat']))
                for name in names:
                    progress_path = base/name/'training.json'
                    while True:
                        progress = json.loads(progress_path.read_text())
                        if progress.get('early_stopped'):
                            break
                        train(summary['trials'][name], progress['epoch'] + plan['converge_chunk_epochs'])
                summary['stages']['converge'] = names; write(summary_path, summary)
            winner = max(summary['trials'].values(), key=lambda t:t['best_val_iou'])
            out = base/winner['name']
            with (out/'best.pt').open('rb') as stream:
                checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
            evaluation = json.loads((out/'test.json').read_text()) if (out/'test.json').exists() else {}
            if evaluation.get('checkpoint_sha256') != checkpoint_hash:
                command(['test.py', '--weight', out/'best.pt', '--data', 'datasets/deepcrack/prepared', '--device', 'cuda'], out/'evaluation.log')
                evaluation = json.loads((out/'test.json').read_text())
                evaluation['checkpoint_sha256'] = checkpoint_hash
                write(out/'test.json', evaluation)
            summary['selected'] = winner['name']; summary['test'] = evaluation
            summary['status'] = 'completed'; write(summary_path, summary)
            if args.upload:
                for name in summary['trials']:
                    command(['hub.py', 'upload', 'weights', f'{MODEL_DIRS[model]}/deepcrack/{name}'], base/name/'upload.log')
                from huggingface_hub import HfApi, CommitOperationAdd
                from hub import default_repo
                HfApi().create_commit(default_repo('weights'), repo_type='model', commit_message=f'Tuning plan and results: {model}',
                    operations=[CommitOperationAdd(path_in_repo=f'weights/{MODEL_DIRS[model]}/{p.relative_to(folder).as_posix()}', path_or_fileobj=p)
                                for p in (plan_path, folder/'training-plan.md', summary_path)])
                summary['uploaded'] = True; write(summary_path, summary)
            print('COMPLETED '+json.dumps({'model': model, 'winner': winner, 'test': summary['test']}), flush=True)
        except Exception as error:
            summary['status'] = 'failed'; summary['error'] = str(error); write(summary_path, summary)
            write(base/'failure.json', {'error': str(error), 'time': time.time()})
            print(f'FAILED {model}: {error}', flush=True)
    print('TUNING_BATCH_FINISHED', flush=True)


if __name__ == '__main__':
    import sys
    tune() if '--tune' in sys.argv else main()
