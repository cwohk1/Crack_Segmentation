# CFD: FPHBN 평가용 배포본

- 원 데이터 출처: https://github.com/cuilimeng/CrackForest-dataset
- 실제 배포 출처: https://github.com/fyangneil/pavement-crack-detection (저자 Google Drive).
- 논문: Shi et al., **Automatic road crack detection using random structured forests**, 2016; Yang et al., **Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection**, 2019.
- 대상/특징: 도시 도로 포장, 아스팔트 중심의 얇은 균열. CFD 공식 세트와 같은 출처입니다.
- 실제 파일: JPG 155장, PNG 마스크 118개. 대응하는 **118쌍**, 모두 **480×320**을 `prepared/test/`에 연결.
- 마스크: 배포된 PNG를 그대로 사용. 균열 비율 약 1.616%. `../cfd/`의 MAT 영역 변환(2.287%)과 구분해야 합니다.
- 두 CFD 폴더를 독립 데이터셋처럼 합치지 마세요. 동일 출처 이미지가 중복됩니다.
- 원 CFD의 비상업 연구용 조건과 관련 논문 인용 요청을 확인하세요.

`inventory.json`, `manifest.json`에 실측과 해시가 있습니다.
