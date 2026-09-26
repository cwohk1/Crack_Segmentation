# CRACK500

- 원저자 출처: https://github.com/fyangneil/pavement-crack-detection
- 실제 다운로드: 원저자 README의 Google Drive 묶음(`../fphbn/source.zip`). 제3자 미러를 사용하지 않았습니다.
- 논문: Zhang et al., **Road crack detection using deep convolutional neural network**, ICIP 2016; Yang et al., **Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection**, 2019, https://arxiv.org/html/1901.06340 (§IV-B1).
- 대상: Temple University 캠퍼스의 도로 포장 균열. 샘플에서 아스팔트 표면을 확인했으나 전체 이미지에 재료별 정답은 없습니다.
- 특징: 스마트폰 촬영, 복잡한 배경과 다양한 폭의 균열, 픽셀 영역 주석. 논문은 원본 500장(250/50/200), 약 2000×1500과 16분할 후 균열 픽셀 수를 기준으로 패치 선택을 설명합니다.

## 실제 다운로드 파일

| 분할 | 패치 쌍 | 가로×세로별 개수 |
|---|---:|---|
| train | 1,896 | 640×360: 1,754 / 360×640: 86 / 648×484: 56 |
| val | 348 | 640×360: 324 / 360×640: 7 / 648×484: 17 |
| test | 1,124 | 640×360: 1,105 / 360×640: 19 |

총 3,368쌍. `prepared/`는 배포본의 공식 패치 분할을 보존합니다.
원본 val 50장과 test 200장도 raw에 있습니다. 실제 원본 크기는 2560×1440, 2592×1936, 3264×2448 등이며 논문의 대표값과 다릅니다.
원본 train 전체용 `traindata.zip`은 이번 배포 묶음에 없고 `traincrop.zip`만 있습니다.

split 간 동일 SHA-256 이미지는 없지만 원본 이름 접두어가 겹치는 경우 6개가 있습니다:
`20160328_154452`, `20160328_154454`, `20160329_103609`, `20160329_103800`, `20160329_103807`, `20160329_111111`.
이름만으로 동일 장면 여부를 확정할 수 없으므로, 누수 검토가 필요한 후보로 기록합니다.

사용 시 원저자 논문을 인용하세요. 공식 README에서 명시적 통합 라이선스는 확인하지 못했습니다.
실측·해시는 `inventory.json`, `manifest.json`에 있습니다.
