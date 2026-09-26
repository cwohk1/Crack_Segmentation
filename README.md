# Crack Segmentation

WSL Ubuntu / Python 3.12 / PyTorch 기반 균열 분할 프로젝트입니다.

## 구조

```text
README.md
trainer.py                  # 학습·재개, 손실·평가 지표
test.py                    # 정답이 있는 데이터 평가 / 이미지 추론
dataloader.py              # 이미지·마스크 로딩과 변환
hub.py                     # Hugging Face 데이터셋·가중치 전송
models/
  scripts/                  # 모델 구현과 build_model()
  weights/<model>/<dataset>/ # 가중치·설정·이력·평가·예측 이미지
datasets/                  # 원본·준비된 데이터와 출처별 README
  prepare.py                # 다운로드·전처리 재현 도구
Codex/
  AGENTS.md                 # 프로젝트 작업 규칙
  test_pipeline.py          # 회귀 테스트
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
`requirements-lock.txt`는 현재 WSL/CUDA 환경 기록입니다. macOS에 그대로 설치하지 말고 `requirements.txt`를 사용하세요.

## Hugging Face 보관과 서버 간 이동

코드는 GitHub에, 데이터셋은 비공개 **dataset 저장소** `cwohk/crack-segmentation`의 `datasets/`에, 가중치는 공개 **model 저장소** `cwohk/crack-segmentation-weights`의 `weights/`에 보관합니다.
`hub.py`는 설치나 학습과 별도로 실행합니다. 기존 `dataloader.py`와 `datasets/prepare.py`는 유지합니다.

```bash
pip install -r requirements.txt
hf auth login
export HF_DATASET_REPO="cwohk/crack-segmentation"
export HF_MODEL_REPO="cwohk/crack-segmentation-weights"
# 위 두 값은 기본값이므로 생략 가능
```

기본 저장소는 [데이터셋](https://huggingface.co/datasets/cwohk/crack-segmentation)과 [가중치](https://huggingface.co/cwohk/crack-segmentation-weights)로 분리되어 있습니다. 이전 공통 환경변수 `HF_REPO`는 더 이상 사용하지 않습니다. 다른 저장소를 쓰려면 위 값을 바꿉니다. 환경변수 대신 각 명령에 `--repo 내계정/저장소`를 붙여도 됩니다.
인증은 `hf auth login` 또는 `HF_TOKEN`으로 제공합니다. 업로드에는 쓰기 권한, 비공개 다운로드에는 읽기 권한이 필요합니다.
토큰은 코드나 Git에 저장하지 않습니다. 새 저장소는 기본 비공개이며 `upload ... --public`으로 공개 생성할 수 있습니다. 기존 저장소의 공개 범위는 변경하지 않습니다.
데이터셋 출처별 이용 조건은 `datasets/README.md`와 각 출처 README를 따릅니다.

현재 컴퓨터에서 데이터셋별로 업로드합니다:

```bash
python hub.py upload dataset deepcrack
python hub.py upload dataset crack500
python hub.py upload weights unet_18/deepcrack
```

저장소의 `datasets/`에는 `<이름>.zip`과 `<이름>.json`(SHA-256 목록)이 저장됩니다.
ZIP은 `prepared/`, 출처 README, `manifest.json`, `inventory.json`, 원본 출처 목록을 포함합니다.
`raw/`는 제외하며 prepared 심볼릭 링크를 실제 파일 내용으로 담습니다.
압축 전에 기존 manifest와 이미지·마스크 체크섬이 일치하는지 검사합니다.
저장소의 `weights/`에는 `<모델>/<데이터셋>/best.pt`, `last.pt`, 설정·학습 이력·평가 JSON·train.log와 검증용 `transfer.json`을 저장합니다.
가중치는 새 model 저장소로 업로드됩니다. 기존 비공개 dataset 저장소의 가중치 사본은 보존합니다.
학습을 마친 뒤 업로드하세요. 학습이 동시에 진행되면 가중치와 로그의 시점이 다를 수 있습니다.

새 서버에서 GitHub 코드를 받은 뒤 Python 가상환경을 만들고 의존성과 인증을 준비합니다. 이어서:

```bash
python hub.py download dataset deepcrack
python hub.py download weights unet_18/deepcrack --checkpoint best.pt
python test.py --weight models/weights/unet_18/deepcrack/best.pt --data datasets/deepcrack/prepared

# 학습 재개: best.pt와 last.pt, 설정·이력을 함께 받기
python hub.py download weights unet_18/deepcrack
python trainer.py --resume models/weights/unet_18/deepcrack/last.pt --epochs 60

# 서버 반납 전에 가중치와 이력을 다시 보관
python hub.py upload weights unet_18/deepcrack
```

`--checkpoint`를 생략하면 업로드된 best/last를 모두 받습니다.
예측 PNG는 로컬에 보존하며 기본 전송 대상에서 제외합니다. 필요한 경우 업로드·다운로드에 `--predictions`를 추가합니다. 공개 업로드 시 이미지의 이용·재배포 조건도 확인해야 합니다.
데이터는 `datasets/<이름>/prepared/`, 가중치와 기록은 모두 `models/weights/<모델>/<데이터셋>/`에 복원됩니다. 선택적으로 그 아래 `<실험명>/`도 사용할 수 있습니다.
별도 전처리 없이 기존 학습·평가 코드를 사용합니다. `--root /다른/프로젝트경로`로 복원 위치를 바꿀 수 있습니다.
기존 파일과 같으면 건너뛰고 다르면 중단합니다. 의도적으로 갱신할 때만 `--overwrite`를 붙입니다.
기존 prepared symlink가 있는 컴퓨터에는 다시 다운로드할 필요가 없습니다. 새 위치에 복원하려면 `--root`를 사용하세요.
로컬에만 있는 추가 파일은 삭제하지 않으므로 정확한 데이터 버전별 실험은 깨끗한 경로에 다운로드하세요.

다운로드 시 `--revision <커밋 또는 태그>`로 버전을 선택할 수 있습니다.
지정하지 않으면 main의 현재 커밋을 고정하여 모든 파일을 같은 버전에서 받고, 완료 시 커밋을 출력합니다.
Hugging Face 다운로드 캐시를 재사용합니다. ZIP 캐시와 임시 압축 해제본, 최종 데이터가 함께 존재할 공간이 필요합니다.
SHA-256 검사가 끝난 뒤 로컬에 설치하며 파일 단위로 교체합니다. 전체 디렉터리 교체는 원자적이지 않으므로 설치 도중 학습을 실행하지 마세요.
macOS에서도 전송 기능은 같으며, CUDA 학습은 GPU 서버에서 실행합니다.

같은 모델/데이터셋 경로로 업로드하면 최신 파일이 갱신되고 이전 커밋은 남습니다. 새 튜닝 실험은 `unet_18/deepcrack/v2`처럼 하위 폴더로 구분하세요. 저장소 분리는 이력 삭제나 저장 용량 회수를 수행하지 않습니다.

API 참고: [다운로드](https://huggingface.co/docs/huggingface_hub/guides/download),
[업로드](https://huggingface.co/docs/huggingface_hub/guides/upload).

## 학습과 재개

```bash
python trainer.py --model UNet18 --augment --pos-weight 3 --epochs 30
python trainer.py --resume models/weights/unet_18/deepcrack/last.pt --epochs 60
```

기본 데이터는 `datasets/deepcrack/prepared`입니다. 다른 데이터는 `--data <경로>`로 지정합니다.
기본 저장 경로는 `models/weights/<모델>/<데이터셋>/`이며 가중치와 기록을 함께 저장합니다.
모델 폴더명은 `segnet`, `segnet_vgg16`, `fphbn`, `unet_18`, `unet_resnet50`, `unet_vgg16`입니다.
데이터 경로가 `datasets/deepcrack/prepared`이면 데이터셋 이름은 `deepcrack`입니다.
여러 튜닝 실험은 `--output models/weights/unet_18/deepcrack/v2`처럼 구분하세요.
`--weights-dir`는 `--output`의 호환 별칭이며 둘을 지정하면 같은 경로여야 합니다.
재개 시 저장 위치는 체크포인트의 실제 부모 폴더이고, `--data` 생략 시 체크포인트에 기록된 데이터 경로를 사용합니다.
`--epochs`는 재개 전을 포함한 총 목표 epoch입니다. 재개 시 optimizer와 학습률을 복원하며 `--lr`를 지정하면 변경합니다.

모델: `segnet`, `segnet_vgg16`, `FPHBN`, `UNet18`, `UNetResNet50`, `UNetVGGNet16`.
입력: 기본 256×256, RGB [-1,1]. `--size`는 64 이상의 32의 배수입니다.
마스크는 nearest 보간으로 0/1 변환하며, 모델의 logits에 BCE + Dice 손실을 적용합니다.
공식 val이 없으면 train의 20%를 seed 42로 검증용 분리합니다. 테스트 세트로 모델을 선택하지 않습니다.
`best.pt`는 검증 IoU 기준, `last.pt`는 재개용입니다. `--device cpu`도 지원합니다.

## 평가와 이미지 추론

```bash
python test.py --weight models/weights/unet_18/deepcrack/best.pt --data datasets/deepcrack/prepared
python test.py --weight models/weights/unet_18/deepcrack/best.pt --input datasets/deepcrack/prepared/test/images
```

평가 결과는 `models/weights/<모델>/<데이터셋>/test.json`, 추론 마스크·overlay는 `models/weights/<모델>/<데이터셋>/predictions/`에 저장됩니다.
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

기존 체크포인트는 `models/weights/segnet/deepcrack/`, `models/weights/unet_18/deepcrack/`에 있습니다.
설정·이력·평가·예측도 각 가중치 폴더로 옮겼습니다. 기존 체크포인트 바이트는 유지하며, 내부 설정과 과거 기록의 예전 경로는 당시 실행 기록입니다.

## 검증

```bash
python -m pytest -q Codex/test_pipeline.py -p no:cacheprovider
```

## SegNet의 pooling 복원

`segnet` 인코더는 사전학습 백본 없이 3×3 합성곱 5개(기본 채널 16/32/64/128/256)를 사용하는 자체 CNN입니다.
원 논문의 VGG16 전체 구조를 재현한 모델은 아닙니다.
새 학습은 `MaxPool2d(return_indices=True)`로 네 단계의 최대값 위치와 pooling 전 크기를 저장하고,
decoder에서 채널 수를 맞춘 뒤 `MaxUnpool2d(output_size=...)`로 역순 복원합니다.
최대값 위치에 decoder 특징을 배치하며, pooling에서 버린 나머지 원래 값을 완전히 복원하는 것은 아닙니다.

기존 `deepcrack_baseline` 가중치는 버전 표식이 없어 자동으로 이전 bilinear 방식으로 로딩됩니다.
재개 학습도 해당 방식을 유지합니다. 새 unpool 구조를 학습하려면 기존 baseline을 재개하지 않고 새 실행을 시작하세요:

```bash
python trainer.py --model segnet --augment --pos-weight 3 --epochs 15 --output models/weights/segnet/deepcrack/unpool
```

새 체크포인트에는 `_unpool_version=1`이 저장되어 평가·재개 시 같은 구조를 복원합니다.

## VGG16 기반 SegNet

`models/scripts/segnet_vgg16.py`의 `SegNetVGG16`은
[SegNet 원 논문 §III](https://arxiv.org/html/1511.00561v3#S3)의 주요 구조를 구현한 별도 모델입니다.
`conv_block(3, 64, 64)`는 채널이 `3 → 64 → 64`로 바뀌는 두 합성곱을 뜻합니다.
각 합성곱은 `3×3 Conv → BatchNorm → ReLU` 순서입니다.

| 단계 | 인코더 채널 변화 (이후 pool) | 대응 디코더 채널 변화 (먼저 unpool) |
|---|---|---|
| 1 | 3 → 64 → 64 | 64 → 64 → 64 |
| 2 | 64 → 128 → 128 | 128 → 128 → 64 |
| 3 | 128 → 256 → 256 → 256 | 256 → 256 → 256 → 128 |
| 4 | 256 → 512 → 512 → 512 | 512 → 512 → 512 → 256 |
| 5 | 512 → 512 → 512 → 512 | 512 → 512 → 512 → 512 |

인코더는 1→5, 디코더는 5→1 순서로 실행됩니다. 각각 합성곱 13개이며 VGG16의 완전연결층은 제외합니다.
다섯 번의 2×2 max-pooling에서 위치와 입력 크기를 저장합니다.
디코더는 그 위치로 unpool한 뒤 합성곱으로 특징을 채웁니다. 인코더 특징 전체를 이어 붙이는 skip connection은 없습니다.
마지막에는 64채널 특징을 1×1 합성곱으로 균열 logits 1채널로 바꿉니다.

원 논문의 다중 클래스 softmax 대신 현재 프로젝트의 이진 분할용 BCE + Dice를 사용합니다.
학습기도 기존 AdamW를 유지하므로 논문의 학습 조건까지 그대로 재현하는 구현은 아닙니다.
`--pretrained`는 torchvision **VGG16-BN ImageNet**의 인코더 Conv/BN 가중치를 가져오는 선택 옵션입니다.
이 옵션이 없으면 처음부터 학습하며, 디코더는 항상 새로 초기화합니다.
새 모델은 사전학습 여부에 관계없이 데이터로더의 RGB [-1,1]을 내부에서 ImageNet mean/std로 정규화합니다.
체크포인트를 평가하거나 재개할 때는 사전학습 파일을 다시 다운로드하지 않습니다.

```bash
python trainer.py --model segnet_vgg16 --augment --epochs 30 --output models/weights/segnet_vgg16/deepcrack
# ImageNet 인코더로 시작하려면 위 명령에 --pretrained 추가 (첫 사용 시 다운로드)
python test.py --weight models/weights/segnet_vgg16/deepcrack/best.pt --data datasets/deepcrack/prepared
```


## FPHBN (단순화 버전)

`models/scripts/fphbn.py`는 [FPHBN 논문](https://arxiv.org/html/1901.06340)의 핵심을 현재 학습 흐름에 맞게 구현합니다.
VGG16의 다섯 단계 특징(64/128/256/512/512)을 뽑고, 깊은 층의 특징을 확대하여 얕은 층에 concat합니다.
1×1 합성곱으로 채널을 줄인 뒤 각 단계에서 균열 logits를 예측합니다.
다섯 예측을 원래 입력 크기로 맞추고 1×1 합성곱으로 최종 logits 하나를 만듭니다.

학습할 때는 깊은 단계가 틀린 픽셀에 얕은 단계도 집중하도록,
바로 위 단계의 `abs(sigmoid(logits) - 정답)`을 현재 단계 BCE의 픽셀별 가중치로 사용합니다.
가장 깊은 단계의 가중치는 1이고, 오차 가중치는 detach하여 역전파하지 않습니다.
`training_loss()`가 이 보조 손실을 계산하며, 일반 `forward()`와 평가·추론은 기존 모델처럼 N×1×H×W logits를 반환합니다.

단순화/변경 사항:

- VGG16과 top-down concat, 다섯 side output, 학습 가능한 fusion, 상위 오차 기반 reweighting을 유지합니다.
- 고정 deconvolution 대신 크기를 명시한 bilinear 보간을 사용합니다. 홀수 크기도 처리합니다.
- feature 결합 뒤 ReLU를 사용하며, merge와 side head는 표준편차 0.01로 초기화하고, fusion은 평균(0.2)으로 초기화합니다.
- 논문의 이미지별 클래스 균형 계수 대신 기존 `--pos-weight`를 사용합니다.
- 최종 손실은 기존 BCE+Dice에 다섯 weighted BCE의 **평균**을 더합니다. 논문의 손실 합산 방식과 다릅니다.
- 기존 AdamW·학습률·검증 IoU를 유지합니다. 논문의 SGD 일정, AIU/ODS/OIS 평가와 성능을 재현했다는 의미가 아닙니다.
- 검증 loss는 다른 모델과 동일한 최종 출력 BCE+Dice입니다. 보조 손실이 포함된 train loss와 직접 비교하지 않습니다.
- `--pretrained`는 torchvision VGG16 ImageNet 인코더만 초기화합니다. 생략하면 처음부터 학습합니다.
  두 경우 모두 모델 내부에서 [-1,1] 입력을 ImageNet 정규화로 변환합니다.

```bash
python trainer.py --model FPHBN --augment --pos-weight 3 --epochs 30 --output models/weights/fphbn/deepcrack/v1
python test.py --weight models/weights/fphbn/deepcrack/v1/best.pt --data datasets/deepcrack/prepared
python hub.py upload weights fphbn/deepcrack/v1
```

위 명령은 사용 예시이며, 구현 검증만으로 학습 완료 가중치가 생성되지는 않습니다.


## 모든 모델의 DeepCrack 반복 튜닝

```bash
python trainer.py --tune --plan-only       # 모델별 계획만 작성
python trainer.py --tune --upload          # 계획 실행·중단 재개·결과 업로드
python trainer.py --tune --models FPHBN    # 지정 모델만 실행
```

각 `models/weights/<모델>/training-plan.md`와 `.json`에 계획을 저장합니다.
4조건(학습률 2개 × pos_weight 1/3)을 8 epoch 비교하고 검증 IoU 상위 2개를 최대 30,
최상위 조건을 최대 60 epoch까지 이어 학습합니다. 우수 조건은 seed 1337로 최대 40 epoch 다시 학습합니다.
이어 우수 조건의 두 seed 실행을 각각 검증 IoU가 12 epoch 연속 개선되지 않을 때까지 연장합니다.
40 epoch 단위로 재개하며 총 학습 시간·epoch 상한은 두지 않습니다.
모두 사전학습 없이 시작하며 데이터 분할 seed는 42로 고정합니다. 정체 시 LR 감소와 조기 종료가 적용됩니다.
공식 test는 검증 점수로 조건을 선택한 뒤 평가하며, 튜닝 판단에는 사용하지 않습니다.
추가 학습으로 최고 가중치가 바뀌면 체크섬을 비교해 test 결과를 다시 계산합니다. 조건별 결과는 `<모델>/deepcrack/<학습조건>/`,
모델별 비교 요약은 `<모델>/deepcrack/tuning-summary.json`에 남깁니다.
`--upload`는 조건별 가중치·기록과 모델별 계획·요약을 공개 model 저장소에 올리고 원본/예측 이미지는 제외합니다.

단일 학습에도 `--amp`(CUDA BF16), `--patience 12`, `--lr-patience 4`, `--weight-decay 0.01`,
`--split-seed 42`를 사용할 수 있습니다. 손실 합산은 FP32, gradient clipping은 norm 5를 사용합니다.
체크포인트에는 scheduler와 정체 epoch 수가 저장되며, `--resume`은 AMP/weight decay/분할 seed도 복원합니다.
조기 종료와 LR 감소를 재개할 때 같은 `--patience`/`--lr-patience` 값을 지정하세요.
계획 작성 후 모델·학습 코드가 바뀌면 자동 재실행을 중단하므로, 결과를 섞지 않도록 계획과 새 실험 이름을 검토해야 합니다.
FPHBN은 초기 logits 폭증을 줄이기 위해 새 merge/side head를 std 0.01로 초기화합니다.
이 변경은 기존 체크포인트를 로드할 때 저장된 가중치를 바꾸지 않습니다.

실제 DeepCrack 전체 모델 비교 결과와 가중치 경로는 [Hugging Face 결과표](https://huggingface.co/cwohk/crack-segmentation-weights/blob/main/weights/README.md)에 정리했습니다.

## 완료된 DeepCrack 실험 결과

6개 모델, 30개 실행, 합계 1,496 epoch의 비교 결과입니다. train 240 / val 60 / test 237, 입력 256×256, threshold 0.5, 사전학습 없이 학습했습니다.

| 모델 | 1차 테스트 IoU | 추가 학습 후 테스트 IoU | 추가 학습 후 Dice |
|---|---:|---:|---:|
| segnet | 0.6819 | 0.6808 | 0.8101 |
| segnet_vgg16 | 0.7058 | 0.6854 | 0.8133 |
| FPHBN | 0.7190 | 0.6968 | 0.8213 |
| UNet18 | 0.7114 | 0.7042 | 0.8264 |
| UNetResNet50 | 0.7095 | 0.6955 | 0.8204 |
| UNetVGGNet16 | 0.7063 | 0.6913 | 0.8175 |

추가 학습으로 검증 IoU는 개선됐지만 테스트 IoU는 모두 낮아졌습니다. 테스트 성능 개선으로 해석하지 않습니다. SegNet과 UNet18은 최종 선택된 seed도 바뀌어 차이를 학습 시간만의 효과로 볼 수 없습니다.
이전 가중치를 복원할 revision과 최종 선택 조건은 Hugging Face의 모델별 `deepcrack/initial-comparison.json` 및 `tuning-summary.json`에 있습니다. 가중치·원본 데이터·임시 학습 파일은 GitHub에 포함하지 않습니다.

## 장기 학습 작업 방식

5분 이상 예상되는 학습은 필요한 조건을 `tmp/train.sh`에 모아 사용자가 프로젝트 루트에서 한 번 실행합니다.

```bash
./tmp/train.sh
```

전체 출력은 `tmp/train.log`, 조건별 가중치·이력은 기존 결과 폴더에 저장합니다. Codex는 학습을 직접 시작하거나 기다리거나 상태를 polling하지 않습니다.
사용자가 완료를 알리면 로그와 history를 분석하고 필요한 기록을 영구 결과 폴더에 보존한 뒤 임시 파일을 정리합니다. `tmp/`는 Git에서 제외하므로 실행 스크립트는 로컬에서 준비합니다.
후속 대배치 실험 계획은 모델별 `training-plan.json`의 `followup_large_batch`에 기록돼 있으며, 위 완료 결과에는 포함되지 않습니다.
