"""Hugging Face 데이터셋 ZIP / 학습 체크포인트 전송. 인증: hf auth login 또는 HF_TOKEN."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_path(base, relative):
    """원격 파일명이 로컬 폴더 밖으로 나가거나 다른 OS에서 경로가 되지 않게 검사."""
    path = PurePosixPath(relative)
    if (not relative or '\\' in relative or ':' in relative or path.is_absolute()
            or any(part in ('', '.', '..') for part in relative.split('/'))):
        raise ValueError(f'Unsafe path: {relative}')
    target = base.joinpath(*path.parts)
    if not target.resolve().is_relative_to(base.resolve()):
        raise ValueError(f'Path escapes destination: {relative}')
    return target


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def pack_dataset(root, name, destination):
    """prepared 링크를 따라 실제 바이트를 ZIP에 저장한다. raw 원본은 포함하지 않는다."""
    source = root / 'datasets' / name
    prepared = source / 'prepared'
    files = sorted(p for p in prepared.rglob('*') if p.is_file())
    if not files:
        raise ValueError(f'No prepared files: {prepared}')
    records = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    expected = {row[kind]: row[f'{kind}_sha256'] for row in records for kind in ('image', 'mask')}
    if set(expected) != {p.relative_to(source).as_posix() for p in files}:
        raise ValueError('Prepared files differ from manifest; run datasets/prepare.py first')
    for relative, checksum in expected.items():
        # 기존 prepared의 절대 symlink는 허용하되 manifest 경로 자체는 검사한다.
        safe_path(Path('/tmp/hub-path-validation'), relative)
        if digest(source / relative) != checksum:
            raise ValueError(f'Prepared checksum mismatch: {relative}')
    for required in ('README.md', 'manifest.json', 'inventory.json'):
        path = source / required
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(path)
    archive = destination / f'{name}.zip'
    checksums = {}
    with ZipFile(archive, 'w', ZIP_DEFLATED, compresslevel=1) as zipped:
        for path in files:
            relative = path.relative_to(source).as_posix()
            safe_path(Path('.'), relative)
            checksums[relative] = digest(path)
            zipped.write(path, relative)  # symlink 자체가 아닌 대상 파일 내용을 저장
        sources = root / 'datasets' / 'sources.json'
        if sources.is_file():
            checksums['sources.json'] = digest(sources)
            zipped.write(sources, 'sources.json')
    metadata = {'version': 1, 'kind': 'dataset', 'name': name,
                'sha256': digest(archive), 'files': checksums}
    write_json(destination / f'{name}.json', metadata)
    return metadata


def unpack_dataset(archive, metadata, destination):
    if digest(archive) != metadata['sha256']:
        raise ValueError('Dataset ZIP checksum mismatch')
    with ZipFile(archive) as zipped:
        members = zipped.infolist()
        names = [item.filename for item in members]
        if len(names) != len(set(names)) or set(names) != set(metadata['files']):
            raise ValueError('ZIP file list differs from manifest')
        for item in members:
            target = safe_path(destination, item.filename)
            if item.is_dir() or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Unsupported ZIP entry: {item.filename}')
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(item) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)
            if digest(target) != metadata['files'][item.filename]:
                raise ValueError(f'File checksum mismatch: {item.filename}')


def install_files(pairs, overwrite=False):
    """모든 파일을 먼저 비교한다. 기존 다른 파일은 --overwrite 없이는 보존한다."""
    pending = []
    for source, target in pairs:
        if target.is_symlink():
            raise ValueError(f'Destination is a symlink; use a clean --root: {target}')
        if target.exists():
            if target.is_file() and digest(source) == digest(target):
                continue
            if not overwrite or not target.is_file():
                raise FileExistsError(f'Different file exists: {target}; use --overwrite or --root')
        pending.append((source, target))
    for source, target in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        # 같은 파일시스템의 임시 파일을 완성한 뒤 교체하므로 중단 시 반쪽 파일을 남기지 않는다.
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def upload(args, api):
    from huggingface_hub import CommitOperationAdd
    repo_type = 'dataset' if args.kind == 'dataset' else 'model'
    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary)
        if args.kind == 'dataset':
            pack_dataset(args.root, args.name, stage)
            files = {f'datasets/{p.name}': p for p in stage.iterdir()}
        else:
            files = {}
            for filename in ('best.pt', 'last.pt'):
                source = args.root / 'models/weights' / args.name / filename
                if source.is_file():
                    # 학습 중 파일 교체의 영향을 피하도록 업로드할 스냅샷을 만든다.
                    shutil.copyfile(source, stage / filename)
                    files[f'weights/{args.name}/{filename}'] = stage / filename
            if not files:
                raise ValueError('No best.pt or last.pt to upload')
            for filename in ('config.json', 'history.json', 'test.json', 'val.json', 'train.log', 'training.json', 'summary.json', 'plan.json'):
                source = args.root / 'models/weights' / args.name / filename
                if source.is_file():
                    shutil.copyfile(source, stage / filename)
                    files[f'weights/{args.name}/{filename}'] = stage / filename
            if args.predictions:
                base = args.root / 'models/weights' / args.name
                for source in sorted((base / 'predictions').glob('*.png')):
                    relative = source.relative_to(base).as_posix()
                    target = stage / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    files[f'weights/{args.name}/{relative}'] = target
            metadata = {'version': 1, 'kind': 'weights', 'name': args.name,
                        'files': {key.removeprefix(f'weights/{args.name}/'): digest(path) for key, path in files.items()}}
            write_json(stage / 'transfer.json', metadata)
            files[f'weights/{args.name}/transfer.json'] = stage / 'transfer.json'
        # --public은 신규 저장소에 적용한다. 기존 저장소의 공개 범위는 유지한다.
        api.create_repo(args.repo, repo_type=repo_type, private=not args.public, exist_ok=True)
        result = api.create_commit(
            args.repo, repo_type=repo_type, commit_message=f'Upload {args.kind}: {args.name}',
            operations=[CommitOperationAdd(path_in_repo=key, path_or_fileobj=path)
                        for key, path in files.items()],
        )
        print(f'Uploaded {args.name}: {result.oid}')


def download(args, api):
    from huggingface_hub import hf_hub_download
    repo_type = 'dataset' if args.kind == 'dataset' else 'model'
    # main이 전송 도중 바뀌어도 모든 파일을 하나의 commit에서 받는다.
    revision = api.repo_info(args.repo, repo_type=repo_type, revision=args.revision).sha

    def fetch(filename):
        return Path(hf_hub_download(args.repo, filename, repo_type=repo_type, revision=revision))

    metadata_path = f'datasets/{args.name}.json' if args.kind == 'dataset' else f'weights/{args.name}/transfer.json'
    metadata = json.loads(fetch(metadata_path).read_text(encoding='utf-8'))
    if (metadata.get('version'), metadata.get('kind'), metadata.get('name')) != (1, args.kind, args.name):
        raise ValueError('Unsupported transfer metadata')
    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary)
        if args.kind == 'dataset':
            unpack_dataset(fetch(f'datasets/{args.name}.zip'), metadata, stage)
            target_root = safe_path(args.root, f'datasets/{args.name}')
            pairs = [(safe_path(stage, key), safe_path(target_root, key)) for key in metadata['files']]
        else:
            allowed = {'best.pt', 'last.pt', 'config.json', 'history.json', 'test.json', 'val.json', 'train.log', 'training.json', 'summary.json', 'plan.json'}
            if any(key not in allowed and not re.fullmatch(r'predictions/[A-Za-z0-9_.-]+\.png', key)
                   for key in metadata['files']):
                raise ValueError('Unexpected weight bundle files')
            selected = {key for key in metadata['files'] if args.predictions or not key.startswith('predictions/')}
            if args.checkpoint != 'all':
                if args.checkpoint not in selected:
                    raise ValueError(f'{args.checkpoint} is not in this bundle')
                selected -= {'best.pt', 'last.pt'} - {args.checkpoint}
            pairs = []
            for filename in sorted(selected):
                source = fetch(f'weights/{args.name}/{filename}')
                if digest(source) != metadata['files'][filename]:
                    raise ValueError(f'Checksum mismatch: {filename}')
                folder = 'models/weights'
                pairs.append((source, safe_path(args.root, f'{folder}/{args.name}/{filename}')))
        install_files(pairs, args.overwrite)
    print(f'Downloaded {args.name} from {args.repo}@{revision} into {args.root}')


def default_repo(kind):
    if kind == 'dataset':
        return os.environ.get('HF_DATASET_REPO', 'cwohk/crack-segmentation')
    return os.environ.get('HF_MODEL_REPO', 'cwohk/crack-segmentation-weights')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('upload', 'download'))
    parser.add_argument('kind', choices=('dataset', 'weights'))
    parser.add_argument('name', help='Dataset name or model/dataset[/experiment], e.g. unet_18/deepcrack')
    parser.add_argument('--predictions', action='store_true', help='Include prediction PNGs in upload/download')
    parser.add_argument('--repo', help='Override dataset/model repository (HF_DATASET_REPO / HF_MODEL_REPO)')
    parser.add_argument('--public', action='store_true', help='Create a public repository on upload')
    parser.add_argument('--root', type=Path, default=ROOT, help='Local project root')
    parser.add_argument('--revision', default='main', help='Download commit/tag/branch')
    parser.add_argument('--checkpoint', choices=('all', 'best.pt', 'last.pt'), default='all')
    parser.add_argument('--overwrite', action='store_true', help='Replace differing downloaded files')
    args = parser.parse_args()
    parts = args.name.split('/')
    if (len(parts) not in ((1,) if args.kind == 'dataset' else (2, 3))
            or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', part) for part in parts)):
        parser.error('Use dataset name or model/dataset[/experiment]; only letters, digits, _ and -')
    args.repo = args.repo or default_repo(args.kind)
    if not args.repo:
        parser.error('Set --repo owner/repository or the matching HF_*_REPO environment variable')
    args.root = args.root.resolve()
    from huggingface_hub import HfApi
    action = upload if args.action == 'upload' else download
    action(args, HfApi())


if __name__ == '__main__':
    main()
