# AEL: Aigle-RN + ESAR + LCMS

- 배포 출처: https://github.com/fyangneil/pavement-crack-detection (저자 Google Drive).
- 관련 논문: Amhaz et al., **Automatic Crack Detection on Two-Dimensional Pavement Images: An Algorithm Based on Minimal Path Selection**.
- 묶음 설명: https://arxiv.org/html/1901.06340 (§IV-B5).
- 대상: **도로 포장**, 샘플에서 아스팔트 질감을 확인했습니다. 개별 이미지의 재료 종류 정답은 없습니다.
- 특징: 서로 다른 촬영 시스템의 소규모 평가 데이터. 논문은 Aigle-RN을 프랑스 도로의 주행 촬영, ESAR를 조명 통제 없는 정지 촬영으로 설명합니다.

| 하위 출처 | 실제 쌍 | 실제 가로×세로 |
|---|---:|---|
| Aigle-RN | 38 | 991×462 19장 + 311×462 19장 |
| ESAR | 15 | 768×512 |
| LCMS | 5 | 700×1000 |

총 **58쌍**. 원본 폴더를 `raw/img/img/<출처>/`, `raw/gt/gt/<출처>/`에 보존합니다.
원본은 **검정=균열, 흰색=배경**입니다. 샘플 시각 확인 후 `prepared/test/masks/`는 값 <128을 균열 255로 바꾸었습니다.
이미지는 원본 링크, 마스크만 변환 PNG이며 균열 픽셀 비율은 약 0.674%입니다.
이름 대응은 이미지 `Im_` 접두어/`or` 접미어를 제거해 정답과 연결하고 prepared에서는 이미지 stem에 맞췄습니다.
사용 조건: 배포 README는 원 논문 인용을 요청하며 명시적 통합 라이선스는 확인되지 않았습니다.

`inventory.json`, `manifest.json`에 실측과 해시가 있습니다.
