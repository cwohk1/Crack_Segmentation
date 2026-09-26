# Crack Segmentation

WSL Ubuntu / Python 3.12 / PyTorch 기반 균열 분할 프로젝트입니다.

## 구조

```text
README.md
trainer.py                  # 학습·재개, 손실·평가 지표
test.py                    # 정답이 있는 데이터 평가 / 이미지 추론
dataloader.py              # 이미지·마스크 로딩과 변환
models/
  scripts/                  # 모델 구현과 build_model()
  weights/<run>/            # best.pt, last.pt
datasets/                  # 원본·준비된 데이터와 출처별 README
  prepare.py                # 다운로드·전처리 재현 도구
Codex/
  AGENTS.md                 # 프로젝트 작업 규칙
  runs/<run>/               # 설정·학습 이력·평가 결과·예측 이미지
  runs/test_pipeline.py     # 회귀 테스트
requirements.txt
requirements-lock.txt
.gitignore
```

## 환경

```bash
source .venv312/bin/activate
```

새 Python 3.12 환경은 `pip install -r requirements.txt`로 설치합니다.
전체 설치 버전은 `requirements-lock.txt`에 고정되어 있습니다.

## 학습과 재개

```bash
python trainer.py --model UNet18 --augment --pos-weight 3 --epochs 30 --output Codex/runs/experiment
python trainer.py --resume models/weights/experiment/last.pt --epochs 60
```

기본 데이터는 `datasets/deepcrack/prepared`입니다. 다른 데이터는 `--data <경로>`로 지정합니다.
`--output`에는 실행 기록이, 같은 이름의 `models/weights/<run>/`에는 가중치가 저장됩니다.
필요하면 `--weights-dir`로 가중치 경로를 직접 지정할 수 있습니다.
`--epochs`는 재개 전을 포함한 총 목표 epoch입니다. 재개 시 optimizer와 학습률을 복원하며 `--lr`를 지정하면 변경합니다.

모델: `segnet`, `UNet18`, `UNetResNet50`, `UNetVGGNet16`.
입력: 기본 256×256, RGB [-1,1]. `--size`는 64 이상의 32의 배수입니다.
마스크는 nearest 보간으로 0/1 변환하며, 모델의 logits에 BCE + Dice 손실을 적용합니다.
공식 val이 없으면 train의 20%를 seed 42로 검증용 분리합니다. 테스트 세트로 모델을 선택하지 않습니다.
`best.pt`는 검증 IoU 기준, `last.pt`는 재개용입니다. `--device cpu`도 지원합니다.

## 평가와 이미지 추론

```bash
python test.py --weight models/weights/deepcrack_unet18/best.pt --data datasets/deepcrack/prepared
python test.py --weight models/weights/deepcrack_unet18/best.pt --input datasets/deepcrack/prepared/test/images
```

평가 결과는 `Codex/runs/<run>/test.json`, 추론 마스크·overlay는 `Codex/runs/<run>/predictions/`에 저장됩니다.
`--output`으로 평가 JSON 파일 또는 추론 출력 폴더를 지정할 수 있습니다.
IoU·Dice는 학습 입력 크기에서 sigmoid 임계값 0.5로 계산하며, 추론 PNG는 원본 해상도로 복원합니다.
논문의 tolerance 기반 ODS/OIS와 직접 비교할 수 없습니다.

## 데이터와 기존 결과

[데이터셋 출처·해상도·특징](datasets/README.md). 원본을 유지한 채 학습용 경로를 재생성하려면:

```bash
python datasets/prepare.py
# 새로 다운로드해야 하는 경우:
python datasets/prepare.py --download
```

기존 체크포인트는 `models/weights/deepcrack_baseline/`, `models/weights/deepcrack_unet18/`에 있습니다.
실행 이력은 같은 이름의 `Codex/runs/` 폴더에 보존했습니다. 이동 전 기록 속 예전 경로는 당시 실행 기록입니다.

## 검증

```bash
python -m pytest -q Codex/runs/test_pipeline.py -p no:cacheprovider
```
