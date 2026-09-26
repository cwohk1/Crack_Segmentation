# 공개 균열 분할 데이터셋

다운로드·파일 검사일: 2026-09-26. 크기는 **가로×세로(W×H)** 픽셀입니다.
논문에 적힌 대표 크기와 다운로드한 파일의 실제 크기가 다르면 실제 값을 우선 표시했습니다.

| 디렉터리 | 출처/대상 | 검증된 이미지–마스크 쌍 | 실제 학습용 이미지 크기 | 특징 |
|---|---|---:|---|---|
| [crack500](crack500/README.md) | FPHBN 저자 배포 / 도로 포장 | train 1,896 · val 348 · test 1,124 | 640×360, 360×640, 648×484 | 넓은 균열 영역, 배경·촬영 조건 다양 |
| [deepcrack](deepcrack/README.md) | Liu 등의 공식 저장소 / 콘크리트·아스팔트 | train 300 · test 237 | 544×384, 384×544 | 거칠기·오염·균열 폭이 다양한 혼합 장면 |
| [cfd](cfd/README.md) | CrackForest 공식 저장소 / 도로, 아스팔트 중심 | 118 | 480×320 | 작은 데이터셋, MAT 영역 주석 |
| [gaps384](gaps384/README.md) | FPHBN 배포 / 독일 아스팔트 도로 | test 509 패치 | 540×640, 540×440 | 도로 촬영 영상, 얇은 균열 |
| [cracktree200](cracktree200/README.md) | FPHBN 배포 / 도로 포장 | test 206 | 800×600 | 그림자·가림·저대비, 가는 주석 |
| [ael](ael/README.md) | FPHBN 배포 / 도로 포장 | test 58 | 991×462, 311×462, 768×512, 700×1000 | Aigle-RN·ESAR·LCMS를 합친 평가 모음 |
| [cfd_fphbn](cfd_fphbn/README.md) | CFD의 FPHBN 평가용 배포본 | test 118 | 480×320 | 위 CFD와 같은 출처의 별도 주석 버전, 독립 데이터셋으로 세면 안 됨 |

재료는 데이터셋 전체 설명 또는 확인한 샘플 기준입니다. 개별 이미지마다 콘크리트/아스팔트 정답 라벨이 제공되는 것은 아닙니다. 특히 '도로 포장'을 전부 아스팔트라고 단정하지 않았습니다.

## 출처

- [FPHBN 공식 저장소](https://github.com/fyangneil/pavement-crack-detection) → [저자가 연결한 공개 배포 ZIP](https://drive.google.com/file/d/13_vDYl54Mrd34dddX9w4ppAEiuWv4MlD/view)
- [FPHBN 논문, 데이터셋 설명 IV-B](https://arxiv.org/html/1901.06340)
- [DeepCrack 공식 저장소와 논문](https://github.com/yhlleo/DeepCrack)
- [CrackForest 공식 저장소](https://github.com/cuilimeng/CrackForest-dataset)
- [OmniCrack30k 논문](https://openaccess.thecvf.com/content/CVPR2024W/VAND/papers/Benz_OmniCrack30k_A_Benchmark_for_Crack_Segmentation_and_the_Reasonable_Effectiveness_CVPRW_2024_paper.pdf): CFD를 아스팔트 중심으로 설명합니다.

## 저장 방식과 검증

- `raw/`: 다운로드한 원본과 원래 주석을 보존합니다.
- `prepared/<train|val|test|all>/{images,masks}/`: 프로젝트용 경로입니다. 대부분 원본을 가리키는 WSL 심볼릭 링크입니다. CFD MAT 변환과 AEL 흑백 반전 결과만 새 PNG입니다.
- `inventory.json`: 실제 쌍 개수, 해상도별 개수, 균열 픽셀 비율.
- `manifest.json`: 모든 쌍의 경로·해상도·이미지 및 마스크 SHA-256.
- [sources.json](sources.json): 세 다운로드 원본의 URL, 크기, SHA-256, Git 커밋, ZIP CRC 검사 결과.
- 원본 ZIP CRC와 prepared의 모든 이미지·마스크 디코딩, 크기 일치, 파일 대응을 검사했습니다.
- train/val/test 사이 바이트가 동일한 이미지는 발견되지 않았습니다. 이것은 유사 장면이나 같은 촬영 대상의 중복까지 배제한다는 뜻은 아닙니다.
- Crack500에는 split 간 동일한 원본 파일명 접두어가 6개 있습니다. 정확한 중복 이미지 해시는 없지만 원본 장면 단위 누수 가능성을 추가로 검토해야 합니다. 공식 분할은 변경하지 않았습니다.

## 배포본에서 발견한 차이

- DeepCrack 테스트 47장은 세로형입니다.
- CFD 저장소는 JPG 155장을 포함하지만 MAT 정답은 118개입니다. 추가 37장은 raw에 남기고 supervised prepared에서 제외했습니다.
- CFD 공식 `Segmentation`과 FPHBN PNG 주석은 동일하지 않습니다. 균열 픽셀 비율도 각각 약 2.287%, 1.616%이므로 실험 시 버전을 반드시 구분하세요.
- AEL 원본 마스크는 검은색이 균열입니다. prepared에서는 흰색=균열로 반전했습니다.
- GAPs384 배포의 전체 이미지 404장 중 PNG 정답은 384개입니다. 평가용으로 제공된 509개 패치를 별도로 사용합니다.
- Crack500 ZIP에는 원본 val 50장/test 200장과 각 패치가 있으나 원본 train 250장용 `traindata.zip`은 없습니다. 학습 패치 1,896쌍은 포함되어 있습니다.

## 재준비

WSL 프로젝트 루트에서:

```bash
python datasets/prepare.py --download
```

`datasets/prepare.py --download`는 기존 ZIP의 체크섬을 검사해 재사용하고 누락한 파일만 다운로드합니다.
원본 보관 용량과 중첩 ZIP/추출본 용량이 함께 필요합니다. 데이터 파일은 Git에서 제외합니다.
공식 test 전용 배포본은 임의로 학습 세트에 섞지 않았습니다. CFD `all/`도 임의의 공식 분할로 부르지 않습니다.

## 사용 조건

DeepCrack는 비상업 연구·교육 용도, CFD는 비상업 연구 용도로 제한됩니다(각 공식 README).
FPHBN 묶음의 명시적 통합 사용 허가 조건은 확인되지 않았으며 원저자는 각 데이터셋 논문 인용을 요청합니다.
각 하위 README에 논문과 출처를 적었습니다.
