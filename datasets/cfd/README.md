# CFD / CrackForest Dataset (공식 MAT 주석)

- 출처: https://github.com/cuilimeng/CrackForest-dataset
- 논문: Shi et al., **Automatic road crack detection using random structured forests**, IEEE T-ITS 17(12), 3434–3445, 2016. 공식 README에 서지정보 수록.
- 대상: 도시 도로 포장, 아스팔트 중심. [OmniCrack30k](https://openaccess.thecvf.com/content/CVPR2024W/VAND/papers/Benz_OmniCrack30k_A_Benchmark_for_Crack_Segmentation_and_the_Reasonable_Effectiveness_CVPRW_2024_paper.pdf)에서도 아스팔트 중심으로 설명합니다.
- 특징: 480×320의 소규모 도로 균열 데이터, 얇고 분기하는 균열. 촬영 장비는 FPHBN 논문 §IV-B4에서 iPhone5로 설명합니다.
- 실제 다운로드: JPG 155장, MAT 정답 118개, SEG 정답 118개. 대응하는 118쌍만 `prepared/all/`에 연결합니다.
- 추가 무주석 이미지 37장은 `raw/CrackForest-dataset-master/image/`에 그대로 보존.
- 변환: `groundTruth.Segmentation`의 1=배경, >1=균열로 이진화하여 0/255 PNG 저장. 일부 파일의 라벨 3도 포함합니다. `Boundaries`와는 다른 영역 주석입니다.
- 크기: 준비된 118쌍 모두 480×320. 흰색=균열. 균열 픽셀 비율 2.287%.
- 분할: 원본에 공식 train/val/test가 없어 `all/`로 보관했습니다.
- 사용 조건: 비상업 연구용(공식 README).
- `cfd_fphbn`은 같은 CFD의 다른 평가 주석 버전입니다. 둘을 독립 샘플처럼 합치면 중복됩니다.

`inventory.json`과 `manifest.json`에 실측 및 파일 해시가 있습니다.
