from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import torch
from dataloader import CrackDataSet, MaskTransforms
from trainer import BCE_DiceLoss, Metrics
from models.scripts import MODEL_NAMES, build_model

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

def test_cli_training_resume_and_prediction(tmp_path):
    import json
    import os
    import subprocess
    import sys
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    root=Path(__file__).resolve().parents[2]
    for split in ('train','test'):
        for kind in ('images','masks'):
            (tmp_path/split/kind).mkdir(parents=True)
        for i in range(3):
            a=np.zeros((41,53),dtype=np.uint8); a[:,20:24]=255
            Image.fromarray(np.repeat(a[:,:,None],3,axis=2)).save(tmp_path/split/'images'/f'{i}.png')
            Image.fromarray(a).save(tmp_path/split/'masks'/f'{i}.png')
    def run(*args):
        return subprocess.run([sys.executable,*map(str,args)],cwd=root,env=env,capture_output=True,text=True,check=True)
    output=tmp_path/'run'
    weights=tmp_path/'weights'
    run('trainer.py','--data',tmp_path,'--output',output,'--weights-dir',weights,'--epochs',1,'--size',64,'--lr','0.01','--device','cpu')
    run('trainer.py','--data',tmp_path,'--output',output,'--weights-dir',weights,'--epochs',2,'--resume',weights/'last.pt','--device','cpu')
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
    run('test.py','--data',tmp_path,'--weight',weights/'best.pt','--device','cpu','--output',output/'test.json')
    run('test.py','--input',tmp_path/'test/images/0.png','--weight',weights/'best.pt','--output',output/'pred','--device','cpu')
    with Image.open(output/'pred/0.png_mask.png') as mask:
        assert mask.size==(53,41)
        assert set(np.unique(mask).tolist())<={0,255}
