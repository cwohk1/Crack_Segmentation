# DeepCrack (Liu et al., Neurocomputing 2019)

- 출처: https://github.com/yhlleo/DeepCrack
- 논문: **DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation**, https://doi.org/10.1016/j.neucom.2019.01.036
- 대상: 콘크리트와 아스팔트. 논문 §3.1(pp.145–146)에서 명시합니다. 같은 이름의 Zou 등의 DeepCrack 데이터와 구분합니다.
- 특징: 매끈한/거친/오염된 배경, 다양한 장면과 균열 폭(논문상 1–180px), 픽셀 단위 영역 마스크.
- 실제 파일: train 300쌍(544×384), test 237쌍(544×384 190장, 384×544 47장). 총 537쌍.
- 원본: `raw/{train_img,train_lab,test_img,test_lab}`. 저장소 문서·논문도 `raw/DeepCrack-master/`에 보존.
- 학습 경로: `prepared/train`, `prepared/test`. 공식 val은 없어 trainer가 train 20%를 seed 42로 분리합니다.
- 사용 조건: 비상업 연구 및 교육 목적으로 제한(공식 README).
- 실제 균열 픽셀 비율: train 2.913%, test 4.327%. 원본 파일 기준이며 학습 resize 후 비율은 달라질 수 있습니다.

세부 개수와 모든 파일 해시는 `inventory.json`, `manifest.json`을 참고하세요.
