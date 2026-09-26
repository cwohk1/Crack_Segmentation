# GAPs384 (FPHBN 배포본)

- 배포 출처: https://github.com/fyangneil/pavement-crack-detection (저자 Google Drive).
- 원 논문: Eisenbach et al., **How to Get Pavement Distress Detection Ready for Deep Learning? A Systematic Approach**, IJCNN 2017.
- 픽셀 주석/패치 설명: Yang et al., **Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection**, https://arxiv.org/html/1901.06340 (§IV-B2).
- 대상: 독일 **아스팔트 도로 포장**. 원 GAPs의 도로 손상 이미지에서 균열을 선택해 픽셀 주석을 붙인 버전입니다.
- 특징: 얇은 균열과 도로 표면 질감. 다른 데이터로 학습한 모델의 일반화 평가에 사용된 세트입니다.
- 다운로드 raw: 전체 이미지 JPG 404장(모두 1920×1080), 원본 PNG 마스크 384개. 따라서 전체 404장이 모두 주석 쌍이라고 세지 않습니다.
- 준비한 공식 평가 패치: **509쌍**, 540×640 281장 + 540×440 228장(가로×세로).
- 논문은 패치 640×540을 기술하지만 실제 배포 크기는 위 값입니다. 파일을 회전·resize하지 않고 보존했습니다.
- `prepared/test/`는 제공된 croppedimg/croppedgt 대응을 사용합니다. 원본은 `raw/GAPs384_raw_img_gt/`에 있습니다.
- 사용 조건: 배포 README는 관련 논문 인용을 요청하며 명시적 통합 라이선스는 확인되지 않았습니다.

`inventory.json`, `manifest.json`에 실측과 해시가 있습니다.
