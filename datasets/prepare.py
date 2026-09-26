"""Rebuild dataset views; optionally download and extract verified source archives."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import ZipFile
import numpy as np
from PIL import Image
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__(); self.action=None; self.params={}
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='form': self.action=attrs.get('action')
        if tag=='input' and attrs.get('name'): self.params[attrs['name']]=attrs.get('value','')

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

def extract(archive,destination):
    destination.mkdir(parents=True,exist_ok=True)
    with ZipFile(archive) as z:
        for info in z.infolist():
            if not (destination/info.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError(f'Unsafe archive member: {info.filename}')
        z.extractall(destination)

def download_archives():
    sources=json.loads((ROOT/'sources.json').read_text())
    for source in sources:
        path=PROJECT_ROOT/source['archive'];path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            url=source['url']
            if source['dataset']=='fphbn':
                url='https://drive.google.com/uc?export=download&id=13_vDYl54Mrd34dddX9w4ppAEiuWv4MlD'
                with urlopen(url,timeout=60) as response:
                    if 'text/html' in response.headers.get('Content-Type',''):
                        form=DownloadForm();form.feed(response.read().decode())
                        if form.action!='https://drive.usercontent.google.com/download':
                            raise RuntimeError('Public Drive download unavailable or confirmation page changed')
                        url=form.action+'?'+urlencode(form.params)
            temporary=path.with_suffix('.zip.part')
            with urlopen(url,timeout=120) as response,temporary.open('wb') as output:
                if 'text/html' in response.headers.get('Content-Type',''):
                    raise RuntimeError('Download returned HTML instead of an archive')
                shutil.copyfileobj(response,output,8*1024*1024)
            if sha256(temporary)!=source['sha256']:
                raise ValueError(f'Download changed: {temporary}. Inspect source before updating sources.json.')
            temporary.rename(path)
        if sha256(path)!=source['sha256']:
            raise ValueError(f'Archive checksum mismatch: {path}')
        with ZipFile(path) as z:
            if bad:=z.testzip():raise ValueError(f'Bad CRC: {bad}')
        print(f'Verified {path.relative_to(PROJECT_ROOT)}',flush=True)
        if source['dataset'] in {'cfd','deepcrack'}:
            extract(path,path.parent/'raw')
        if source['dataset']=='deepcrack':
            extract(path.parent/'raw/DeepCrack-master/dataset/DeepCrack.zip',path.parent/'raw')

def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(source.resolve())

def prepare_deepcrack():
    root=ROOT/'deepcrack'
    for split in ('train','test'):
        for kind, suffix in [('images','img'),('masks','lab')]:
            for path in sorted((root/'raw'/f'{split}_{suffix}').iterdir()):
                if path.suffix.lower() in {'.jpg','.png','.bmp','.jpeg'}:
                    link(path,root/'prepared'/split/kind/path.name)

def prepare_cfd():
    root=ROOT/'cfd'; raw=root/'raw/CrackForest-dataset-master'
    for image in sorted((raw/'image').glob('*.jpg')):
        if not (raw/'groundTruth'/f'{image.stem}.mat').exists():
            continue
        link(image,root/'prepared/all/images'/image.name)
        annotation=loadmat(raw/'groundTruth'/f'{image.stem}.mat',simplify_cells=True)['groundTruth']
        labels=annotation['Segmentation']
        values=np.unique(labels)
        if values.min() != 1 or values.max() > 255:
            raise ValueError(f'Unexpected CFD labels {values}')
        destination=root/'prepared/all/masks'/f'{image.stem}.png'
        destination.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray((labels>1).astype(np.uint8)*255).save(destination)

def inspect(name):
    root=ROOT/name
    records=[]; summary={}
    for split in sorted((root/'prepared').iterdir()):
        image_dir,mask_dir=split/'images',split/'masks'
        if not image_dir.is_dir(): continue
        sizes=Counter(); fg=total=0
        masks={p.stem:p for p in mask_dir.iterdir()}
        images=sorted(image_dir.iterdir())
        assert len(images)==len(masks), (name,split.name,'count mismatch')
        for path in images:
            mask_path=masks[path.stem]
            with Image.open(path) as image:
                image.load(); size=image.size
            with Image.open(mask_path) as mask:
                mask.load(); assert mask.size==size,(path,mask_path)
                array=np.array(mask.convert('L'))
            sizes[f'{size[0]}x{size[1]}']+=1
            fg+=int((array>127).sum()); total+=array.size
            records.append({'split':split.name,'image':str(path.relative_to(root)),
                            'mask':str(mask_path.relative_to(root)),'width':size[0],'height':size[1],
                            'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'mask_sha256':hashlib.sha256(mask_path.read_bytes()).hexdigest()})
        summary[split.name]={'pairs':len(images),'sizes_width_x_height':dict(sizes),'foreground_fraction':fg/total}
    (root/'inventory.json').write_text(json.dumps(summary,indent=2))
    (root/'manifest.json').write_text(json.dumps(records,indent=2))
    print(name,json.dumps(summary),flush=True)

def extract_fphbn():
    root = ROOT
    mapping={'CRACK500':'crack500','GAPS384':'gaps384','cracktree200':'cracktree200','AEL':'ael','CFD':'cfd_fphbn'}
    with ZipFile(root/'fphbn/source.zip') as z:
        for info in z.infolist():
            parts=info.filename.split('/')
            if len(parts)!=3 or parts[1] not in mapping or 'result' in parts[2]: continue
            target=root/mapping[parts[1]]/'raw'/parts[2]
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as source, target.open('wb') as dest: shutil.copyfileobj(source,dest)
    for name in mapping.values():
        for path in (root/name/'raw').glob('*.zip'):
            with ZipFile(path) as z:
                print(name,path.name,len(z.infolist()),z.namelist()[:4],flush=True)
                target=path.parent/path.stem
                target.mkdir(exist_ok=True)
                for info in z.infolist():
                    output=(target/info.filename).resolve()
                    if not output.is_relative_to(target.resolve()): raise ValueError(info.filename)
                z.extractall(target)

def pair(name, split, image, mask):
    if not mask.is_file(): raise FileNotFoundError(mask)
    link(image, ROOT/name/'prepared'/split/'images'/image.name)
    link(mask, ROOT/name/'prepared'/split/'masks'/(image.stem+'.png'))

def prepare_fphbn():
    for split in ('train','val','test'):
        folder=ROOT/'crack500/raw'/f'{split}crop'/f'{split}crop'
        for image in folder.glob('*.jpg'):
            pair('crack500',split,image,image.with_suffix('.png'))

    for name, images, masks in [
        ('gaps384','croppedimg/croppedimg','croppedgt/croppedgt'),
        ('cracktree200','cracktree200rgb/cracktree200rgb','cracktree200_gt/cracktree200_gt'),
        ('cfd_fphbn','cfd_image/cfd_image','cfd_gt/seg_gt')]:
        root=ROOT/name/'raw'
        for image in (root/images).glob('*.jpg'):
            mask=root/masks/(image.stem+'.png')
            if name=='cfd_fphbn' and not mask.exists(): continue
            pair(name,'test',image,mask)

    root=ROOT/'ael/raw'
    for image in (root/'img/img').rglob('*.jpg'):
        stem=image.stem.removeprefix('Im_').removesuffix('or')
        mask=root/'gt/gt'/image.parent.name/(stem+'.png')
        link(image, ROOT/'ael/prepared/test/images'/image.name)
        target=ROOT/'ael/prepared/test/masks'/(image.stem+'.png')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_symlink(): target.unlink()
        with Image.open(mask) as source:
            binary=(np.asarray(source.convert('L')) < 128).astype(np.uint8)*255
        Image.fromarray(binary).save(target)

    for name in ('crack500','gaps384','cracktree200','ael','cfd_fphbn'):
        inspect(name)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Download/reuse verified archives and extract them')
    parser.add_argument('--extract', action='store_true', help='Extract the existing FPHBN archive')
    args = parser.parse_args()
    if args.download:
        download_archives()
    if args.download or args.extract:
        extract_fphbn()
    prepare_deepcrack()
    prepare_cfd()
    inspect('deepcrack')
    inspect('cfd')
    prepare_fphbn()

if __name__ == '__main__':
    main()
