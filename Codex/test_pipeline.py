from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import torch
from dataloader import CrackDataSet, MaskTransforms
from trainer import BCE_DiceLoss, Metrics
from models.scripts import MODEL_NAMES, MODEL_DIRS, build_model

def test_mask_not_divided_twice():
    result = MaskTransforms()(np.array([[0,255],[255,0]],dtype=np.uint8))
    assert result.tolist() == [[[0.,1.],[1.,0.]]]
    assert MaskTransforms()(np.array([[0,1]],dtype=np.uint8)).tolist() == [[[0.,1.]]]

def test_pairing_and_nearest_resize(tmp_path):
    (tmp_path/'images').mkdir(); (tmp_path/'masks').mkdir()
    Image.new('RGB',(9,9),'white').save(tmp_path/'images/a.jpg')
    Image.fromarray(np.eye(9,dtype=np.uint8)*255).save(tmp_path/'masks/a.png')
    data=CrackDataSet(tmp_path/'images',tmp_path/'masks',size=32)
    x,y=data[0]
    assert x.shape==(3,32,32) and y.shape==(1,32,32)
    assert set(y.unique().tolist())=={0.,1.}
    (tmp_path/'masks/a.png').rename(tmp_path/'masks/b.png')
    with pytest.raises(ValueError,match='Missing masks'):
        CrackDataSet(tmp_path/'images',tmp_path/'masks')

def test_metrics_known_confusion():
    m=Metrics()
    m.update(torch.tensor([10.,10.,-10.,-10.]),torch.tensor([1.,0.,1.,0.]))
    assert m.compute()['iou']==pytest.approx(1/3)
    assert m.compute()['dice']==0.5

def test_extreme_logits_loss():
    x=torch.tensor([[[[-1000.,1000.]]]],requires_grad=True)
    y=torch.tensor([[[[1.,0.]]]])
    loss=BCE_DiceLoss()(x,y); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()

@pytest.mark.parametrize('name',MODEL_NAMES)
def test_models_batch_one_logits(name):
    torch.set_num_threads(2)
    model=build_model(name).eval()
    with torch.no_grad():
        logits=model(torch.randn(1,3,64,64))
    assert logits.shape==(1,1,64,64)
    assert torch.isfinite(logits).all()
    assert logits.std()>0

@pytest.mark.parametrize('model_name', ['segnet', 'FPHBN'])
def test_cli_training_resume_and_prediction(tmp_path, model_name):
    import json
    import os
    import subprocess
    import sys
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    root=Path(__file__).resolve().parents[1]
    for split in ('train','test'):
        for kind in ('images','masks'):
            (tmp_path/split/kind).mkdir(parents=True)
        for i in range(3):
            a=np.zeros((41,53),dtype=np.uint8); a[:,20:24]=255
            Image.fromarray(np.repeat(a[:,:,None],3,axis=2)).save(tmp_path/split/'images'/f'{i}.png')
            Image.fromarray(a).save(tmp_path/split/'masks'/f'{i}.png')
    def run(*args):
        return subprocess.run([sys.executable,str(root/args[0]),*map(str,args[1:])],cwd=tmp_path,env=env,capture_output=True,text=True,check=True)
    output=tmp_path/'models/weights'/MODEL_DIRS[model_name]/tmp_path.name
    weights=output
    run('trainer.py','--model',model_name,'--data',tmp_path,'--epochs',1,'--size',64,'--lr','0.01','--device','cpu')
    run('trainer.py','--epochs',2,'--resume',weights/'last.pt','--device','cpu')
    assert (output/'history.json').is_file()
    assert not (tmp_path/'Codex/runs').exists()
    state=torch.load(weights/'last.pt',weights_only=True)
    assert state['epoch']==2 and len(state['history'])==2
    assert state['optimizer']['param_groups'][0]['lr']==pytest.approx(0.01)
    assert state['config']['lr']==pytest.approx(0.01)
    assert json.loads((output/'config.json').read_text())['lr']==pytest.approx(0.01)
    run('trainer.py','--data',tmp_path,'--output',output,'--weights-dir',weights,'--epochs',3,'--resume',weights/'last.pt','--lr','0.002','--device','cpu')
    state=torch.load(weights/'last.pt',weights_only=True)
    assert state['epoch']==3
    assert state['optimizer']['param_groups'][0]['lr']==pytest.approx(0.002)
    assert state['config']['lr']==pytest.approx(0.002)
    assert json.loads((output/'config.json').read_text())['lr']==pytest.approx(0.002)
    run('test.py','--data',tmp_path,'--weight',weights/'best.pt','--device','cpu')
    run('test.py','--input',tmp_path/'test/images/0.png','--weight',weights/'best.pt','--device','cpu')
    with Image.open(output/'predictions/0.png_mask.png') as mask:
        assert mask.size==(53,41)
        assert set(np.unique(mask).tolist())<={0,255}

@pytest.mark.parametrize('name, stages', [('segnet', 4), ('segnet_vgg16', 5)])
@pytest.mark.parametrize('shape', [(1, 3, 64, 64), (2, 3, 65, 97)])
def test_segnet_unpool_restores_saved_positions_and_size(shape, name, stages):
    from models.scripts.segnet import SegNet
    torch.manual_seed(7)
    model = SegNet(init_f=2) if name == 'segnet' else build_model(name)
    pools, restored = [], []

    def capture_pool(module, inputs, output):
        pools.append((output[1], inputs[0].shape))

    def verify_unpool(module, inputs, output):
        values, indices = inputs
        expected_indices, expected_shape = pools[-1-len(restored)]
        assert torch.equal(indices, expected_indices)
        assert output.shape == expected_shape
        # Values land at the stored maxima; other positions remain zero.
        flat = output.flatten(2)
        positions = indices.flatten(2)
        assert torch.equal(flat.gather(2, positions), values.flatten(2))
        occupied = torch.zeros_like(flat, dtype=torch.bool).scatter_(2, positions, True)
        assert torch.count_nonzero(flat.masked_select(~occupied)) == 0
        restored.append(output.shape)

    handles = [model.pool.register_forward_hook(capture_pool),
               model.unpool.register_forward_hook(verify_unpool)]
    x = torch.randn(shape, requires_grad=True)
    logits = model(x)
    for handle in handles:
        handle.remove()
    assert len(restored) == stages
    assert logits.shape == (shape[0], 1, shape[2], shape[3])
    BCE_DiceLoss()(logits, torch.ones_like(logits)).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_segnet_checkpoint_versions_preserve_legacy_behavior():
    from models.scripts.segnet import SegNet
    from torch.nn import functional as F
    torch.manual_seed(9)
    model = SegNet(init_f=2)
    new_state = model.state_dict()
    clone = SegNet(init_f=2)
    clone.load_state_dict(new_state, strict=True)
    assert clone.indexed_unpool
    x = torch.randn(1, 3, 64, 64)
    assert torch.equal(model(x), clone(x))

    # Reproduce the old forward independently to protect existing baseline output.
    legacy_state = {k: v for k, v in new_state.items() if k != '_unpool_version'}
    clone.load_state_dict(legacy_state, strict=True)
    assert not clone.indexed_unpool
    expected = x
    for conv in (clone.conv1, clone.conv2, clone.conv3, clone.conv4):
        expected = F.max_pool2d(F.relu(conv(expected)), 2, 2)
    expected = F.relu(clone.conv5(expected))
    for conv in (clone.conv_up1, clone.conv_up2, clone.conv_up3, clone.conv_up4):
        expected = F.relu(conv(F.interpolate(expected, scale_factor=2, mode='bilinear', align_corners=True)))
    expected = clone.conv_out(expected)
    assert torch.equal(clone(x), expected)
    reloaded = SegNet(init_f=2)
    reloaded.load_state_dict(clone.state_dict(), strict=True)
    assert not reloaded.indexed_unpool
    assert torch.equal(reloaded(x), expected)


def test_segnet_vgg16_pretrained_copy_and_checkpoint(monkeypatch):
    import torchvision.models as models
    from torch import nn
    # 다운로드 없이 실제 torchvision 구조로 사전학습 가중치 복사를 검증한다.
    reference = models.vgg16_bn(weights=None).features.eval()
    requested = []

    def local_vgg16_bn(*, weights):
        from types import SimpleNamespace
        requested.append(weights)
        return SimpleNamespace(features=reference)

    monkeypatch.setattr(models, 'vgg16_bn', local_vgg16_bn)
    model = build_model('segnet_vgg16', pretrained=True).eval()
    assert requested == [models.VGG16_BN_Weights.DEFAULT]
    for blocks, counts in ((model.encoders, [2, 2, 3, 3, 3]),
                           (model.decoders, [3, 3, 3, 2, 2])):
        assert [sum(isinstance(layer, nn.Conv2d) for layer in block) for block in blocks] == counts
    source = [layer for layer in reference if not isinstance(layer, nn.MaxPool2d)]
    target = [layer for block in model.encoders for layer in block]
    for original, copied in zip(source, target, strict=True):
        for key, value in original.state_dict().items():
            assert torch.equal(value, copied.state_dict()[key])

    x = torch.rand(1, 3, 65, 97) * 2 - 1
    with torch.no_grad():
        encoded = (x * 0.5 + 0.5 - model.mean) / model.std
        expected = reference(encoded)
        for block in model.encoders:
            encoded, _ = model.pool(block(encoded))
        torch.testing.assert_close(encoded, expected)
        logits = model(x)
        clone = build_model('segnet_vgg16').eval()
        clone.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(clone(x), logits, rtol=0, atol=0)
    assert len(requested) == 1  # 체크포인트 복원은 사전학습 다운로드가 필요 없다.


@pytest.mark.parametrize('kind', ['dataset', 'weights'])
def test_hub_roundtrip_offline(tmp_path, monkeypatch, kind):
    import json
    from types import SimpleNamespace
    import shutil
    import huggingface_hub
    import hub
    source, remote, target = [tmp_path / name for name in ('source', 'remote', 'target')]
    remote.mkdir()
    name = 'sample' if kind == 'dataset' else 'unet_18/sample'
    if kind == 'dataset':
        base = source / 'datasets' / name
        raw = base / 'raw'
        raw.mkdir(parents=True)
        records = []
        row = {}
        for key, folder in [('image', 'images'), ('mask', 'masks')]:
            original = raw / (key + '.png')
            Image.new('RGB' if key == 'image' else 'L', (8, 8), 255).save(original)
            relative = f'prepared/train/{folder}/a.png'
            local = base / relative
            local.parent.mkdir(parents=True)
            local.symlink_to(original)
            row[key], row[key + '_sha256'] = relative, hub.digest(original)
        (base / 'manifest.json').write_text(json.dumps([row]))
        (base / 'inventory.json').write_text('{}')
        (base / 'README.md').write_text('Source attribution')
    else:
        base = source / 'models/weights' / name
        base.mkdir(parents=True)
        (base / 'best.pt').write_bytes(b'best checkpoint')
        (base / 'last.pt').write_bytes(b'resume checkpoint')
        logs = base
        (logs / 'config.json').write_text('{}')
        (logs / 'history.json').write_text('[]')
        (logs / 'test.json').write_text('{"iou": 0.5}')
        (logs / 'predictions').mkdir()
        Image.new('L', (8, 8)).save(logs / 'predictions/a.png')

    class FakeApi:
        def create_repo(self, repo, **kwargs):
            assert kwargs['private'] is True
            assert kwargs['repo_type'] == ('dataset' if kind == 'dataset' else 'model')
        def create_commit(self, repo, *, operations, **kwargs):
            for operation in operations:
                path = remote / operation.path_in_repo
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(operation.path_or_fileobj, path)
            return SimpleNamespace(oid='fixed-commit')
        def repo_info(self, repo, **kwargs):
            return SimpleNamespace(sha='fixed-commit')

    def fetch(repo, filename, *, repo_type, revision):
        assert repo_type == ('dataset' if kind == 'dataset' else 'model')
        assert revision == 'fixed-commit'
        return str(remote / filename)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fetch)
    args = SimpleNamespace(root=source, name=name, kind=kind, repo='user/test',
                           revision='main', checkpoint='all', overwrite=False, public=False, predictions=False)
    hub.upload(args, FakeApi())
    args.root = target
    if kind == 'weights':
        args.checkpoint = 'best.pt'
    hub.download(args, FakeApi())
    hub.download(args, FakeApi())  # identical files are idempotent
    if kind == 'dataset':
        restored = target / 'datasets/sample/prepared/train/images/a.png'
        assert restored.is_file() and not restored.is_symlink()
        assert restored.read_bytes() == (base / row['image']).read_bytes()
        assert not (target / 'datasets/sample/raw').exists()
    else:
        restored = target / 'models/weights/unet_18/sample/best.pt'
        assert restored.read_bytes() == b'best checkpoint'
        assert not (target / 'models/weights/unet_18/sample/last.pt').exists()
        assert (target / 'models/weights/unet_18/sample/history.json').is_file()
        assert (target / 'models/weights/unet_18/sample/test.json').is_file()
        assert not (target / 'models/weights/unet_18/sample/predictions/a.png').exists()
        args.root, args.predictions = source, True
        hub.upload(args, FakeApi())
        args.root = target
        args.checkpoint = 'all'
        hub.download(args, FakeApi())
        assert (target / 'models/weights/unet_18/sample/last.pt').read_bytes() == b'resume checkpoint'
        assert (target / 'models/weights/unet_18/sample/predictions/a.png').is_file()
    restored.write_bytes(b'local edit')
    with pytest.raises(FileExistsError):
        hub.download(args, FakeApi())
    assert restored.read_bytes() == b'local edit'
    args.overwrite = True
    hub.download(args, FakeApi())
    assert restored.read_bytes() != b'local edit'


def test_hub_rejects_corrupt_and_unsafe_archives(tmp_path):
    from zipfile import ZipFile
    import hub
    archive = tmp_path / 'bad.zip'
    with ZipFile(archive, 'w') as zipped:
        zipped.writestr('../escape.txt', b'bad')
    metadata = {'sha256': 'bad', 'files': {'../escape.txt': 'bad'}}
    with pytest.raises(ValueError, match='checksum'):
        hub.unpack_dataset(archive, metadata, tmp_path / 'out')
    metadata['sha256'] = hub.digest(archive)
    with pytest.raises(ValueError, match='Unsafe path'):
        hub.unpack_dataset(archive, metadata, tmp_path / 'out')
    assert not (tmp_path / 'escape.txt').exists()
    with ZipFile(archive, 'w') as zipped:
        zipped.writestr('prepared/a.png', b'bad')
    metadata = {'sha256': hub.digest(archive), 'files': {'prepared/a.png': 'bad'}}
    with pytest.raises(ValueError, match='File checksum'):
        hub.unpack_dataset(archive, metadata, tmp_path / 'out')


def test_hub_separate_default_repositories(monkeypatch):
    import hub
    monkeypatch.delenv('HF_DATASET_REPO', raising=False)
    monkeypatch.delenv('HF_MODEL_REPO', raising=False)
    monkeypatch.setenv('HF_REPO', 'obsolete/shared')
    assert hub.default_repo('dataset') == 'cwohk/crack-segmentation'
    assert hub.default_repo('weights') == 'cwohk/crack-segmentation-weights'
    monkeypatch.setenv('HF_MODEL_REPO', 'user/models')
    assert hub.default_repo('weights') == 'user/models'
    assert hub.default_repo('dataset') == 'cwohk/crack-segmentation'


def test_fphbn_boosting_focus_and_detached_weights():
    from models.scripts.fphbn import boosting_loss
    from torch.nn import functional as F
    # 같은 배경 픽셀인데 상위 예측이 오른쪽에서 더 틀린다.
    target = torch.zeros(1, 1, 1, 2)
    shallow = torch.zeros_like(target, requires_grad=True)
    deep = torch.tensor([[[[-2.1972246, 2.1972246]]]], requires_grad=True)
    pos_weight = torch.tensor(3.)
    loss = boosting_loss([shallow, deep], target, pos_weight)
    loss.backward()
    assert shallow.grad[0, 0, 0, 1] / shallow.grad[0, 0, 0, 0] == pytest.approx(9.)
    # 상위 단계에는 자신의 BCE gradient만 전달된다 (하위 가중치 경로는 차단).
    torch.testing.assert_close(deep.grad, deep.detach().sigmoid() / 4)
    for foreground in (0., 1.):
        logits = torch.full_like(target, -1000., requires_grad=True)
        value = boosting_loss([logits] * 5, torch.full_like(target, foreground), pos_weight)
        value.backward()
        assert torch.isfinite(value) and torch.isfinite(logits.grad).all()


def test_fphbn_odd_size_gradients_and_checkpoint():
    torch.set_num_threads(2)
    model = build_model('FPHBN')
    images = torch.rand(1, 3, 65, 97) * 2 - 1
    masks = torch.randint(0, 2, (1, 1, 65, 97)).float()
    fused, sides = model(images, return_sides=True)
    assert fused.shape == masks.shape
    assert len(sides) == 5 and all(side.shape == masks.shape for side in sides)
    torch.testing.assert_close(fused, torch.stack(sides).mean(0))
    loss = model.training_loss(images, masks, BCE_DiceLoss(3))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    clone = build_model('FPHBN')
    clone.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(clone.eval()(images), model.eval()(images), rtol=0, atol=0)


def test_training_plateau_early_stop_resume_and_split_seed(tmp_path, monkeypatch):
    import sys
    import json
    import trainer
    for kind in ('images', 'masks'):
        folder = tmp_path/'data/train'/kind
        folder.mkdir(parents=True)
        for i in range(3):
            Image.new('RGB' if kind == 'images' else 'L', (64, 64), 255).save(folder/f'{i}.png')
    monkeypatch.setattr(trainer, 'evaluate', lambda *a, **kw: {'loss': 1., 'iou': .2, 'dice': .3})
    out = tmp_path/'result'
    monkeypatch.setattr(sys, 'argv', ['trainer.py', '--data', str(tmp_path/'data'), '--output', str(out),
        '--epochs', '10', '--size', '64', '--seed', '1337', '--split-seed', '42',
        '--lr', '.001', '--patience', '2', '--lr-patience', '1', '--device', 'cpu'])
    trainer.main()
    state = torch.load(out/'last.pt', weights_only=True)
    assert state['epoch'] == 3 and state['bad_epochs'] == 2
    assert state['optimizer']['param_groups'][0]['lr'] == pytest.approx(.0005)
    assert state['config']['seed'] == 1337 and state['config']['split_seed'] == 42
    expected = torch.randperm(3, generator=torch.Generator().manual_seed(42)).tolist()
    assert state['split'] == {'train': expected[1:], 'val': expected[:1]}
    assert json.loads((out/'training.json').read_text())['early_stopped']
    monkeypatch.setattr(sys, 'argv', ['trainer.py', '--resume', str(out/'last.pt'), '--epochs', '4',
        '--patience', '9', '--lr-patience', '1', '--device', 'cpu'])
    trainer.main()
    resumed = torch.load(out/'last.pt', weights_only=True)
    assert resumed['epoch'] == 4 and resumed['bad_epochs'] == 3
    assert resumed['split'] == state['split']
    assert resumed['scheduler']['best'] == .2
    assert resumed['optimizer']['param_groups'][0]['lr'] == pytest.approx(.0005)


def test_tuning_convergence_resume_and_evaluation_cache(tmp_path, monkeypatch):
    import hashlib
    import json
    import subprocess
    import sys
    import trainer
    (tmp_path/'trainer.py').write_text('# fixture')
    (tmp_path/'dataloader.py').write_text('# fixture')
    (tmp_path/'models/scripts').mkdir(parents=True)
    monkeypatch.setattr(trainer, '__file__', str(tmp_path/'trainer.py'))
    monkeypatch.setattr(sys, 'argv', ['trainer.py', '--tune', '--models', 'segnet'])
    calls = []

    def run(arguments, **kwargs):
        calls.append(arguments)
        if arguments[1] == 'trainer.py':
            out = Path(arguments[arguments.index('--output')+1])
            epoch = int(arguments[arguments.index('--epochs')+1])
            score = .3 + epoch/1000 + (.1 if '_pw1_' in out.name else 0)
            history = [{'epoch': epoch, 'val_iou': score, 'val_dice': score, 'seconds': 1}]
            (out/'history.json').write_text(json.dumps(history))
            stopped = epoch > (40 if '_s1337' in out.name else 60)
            (out/'training.json').write_text(json.dumps({'epoch': epoch, 'early_stopped': stopped}))
            for name in ('best.pt', 'last.pt'):
                (out/name).write_bytes(str(epoch).encode())
        else:
            assert arguments[1] == 'test.py'
            out = Path(arguments[arguments.index('--weight')+1]).parent
            (out/'test.json').write_text(json.dumps({'iou': .5, 'dice': .6}))
    monkeypatch.setattr(subprocess, 'run', run)
    trainer.tune()
    base = tmp_path/'models/weights/segnet/deepcrack'
    summary = json.loads((base/'tuning-summary.json').read_text())
    assert summary['status'] == 'completed' and len(summary['trials']) == 5
    assert len(summary['stages']['converge']) == 2
    for name in summary['stages']['converge']:
        assert json.loads((base/name/'training.json').read_text())['early_stopped']
    count = len(calls)
    trainer.tune()
    assert len(calls) == count  # no duplicate training or evaluation after completion
    best = base/summary['selected']/'best.pt'
    best.write_bytes(b'new checkpoint')
    trainer.tune()
    assert len(calls) == count + 1 and calls[-1][1] == 'test.py'
    result = json.loads((best.parent/'test.json').read_text())
    assert result['checkpoint_sha256'] == hashlib.sha256(best.read_bytes()).hexdigest()
    (tmp_path/'trainer.py').write_text('# changed source')
    with pytest.raises(ValueError, match='Source changed'):
        trainer.tune()
