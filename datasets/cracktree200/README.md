# CrackTree200 (FPHBN 배포본)

- 배포 출처: https://github.com/fyangneil/pavement-crack-detection (저자 Google Drive).
- 원 논문: Zou et al., **CrackTree: Automatic crack detection from pavement images**, Pattern Recognition Letters 33(3), 227–238, 2012.
- 데이터 설명: https://arxiv.org/html/1901.06340 (§IV-B3).
- 대상: 도로 포장 균열. 확인한 샘플은 아스팔트 표면이며 개별 재료 라벨은 없습니다.
- 특징: 그림자·가림·저대비·노이즈가 있는 장면, 매우 가는 균열 주석.
- 실제 파일: RGB 206장, 회색조 206장(같은 장면의 표현), PNG 마스크 206개. 모두 **800×600**.
- 준비한 데이터: RGB–마스크 **206쌍**을 `prepared/test/`에 연결했습니다. 회색조는 추가 독립 샘플로 세지 않습니다.
- 원본은 `raw/cracktree200rgb/`, `raw/cracktree200gray/`, `raw/cracktree200_gt/`.
- 사용 조건: 배포 README는 원 논문 인용을 요청하며 명시적 통합 라이선스는 확인되지 않았습니다.

`inventory.json`, `manifest.json`에 실측과 해시가 있습니다.
