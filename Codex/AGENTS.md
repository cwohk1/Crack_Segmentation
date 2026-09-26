# 프로젝트 작업 규칙

- WSL Ubuntu에서 `.venv312/bin/python`으로 실행한다.
- 최상위 실행 파일은 `trainer.py`, `test.py`, `dataloader.py`로 유지한다.
- 모델 구현과 선택 함수는 `models/scripts/`, 체크포인트는 `models/weights/<run>/`에 둔다.
- 로그, 검증 결과, 예측 이미지와 작업용 테스트는 `Codex/runs/`에 둔다.
- 다운로드 데이터와 출처 설명은 `datasets/`에 유지한다. 데이터 준비 도구는 `datasets/prepare.py` 하나로 관리한다.
- 일회성 조사·수정 스크립트나 새로운 최상위 폴더를 남기지 않는다.
- 사용자 변경을 보존한다. 데이터 원본, 학습 가중치, 가상환경은 임의로 삭제하지 않는다.
- 모델 구조를 정리할 때 기존 state_dict와 optimizer 재개 호환성을 유지한다.
- 코드 변경은 관련 테스트로 검증하고, 실행하지 않은 학습·평가 결과를 주장하지 않는다.
- 기본 검증: `.venv312/bin/python -m pytest -q Codex/runs/test_pipeline.py -p no:cacheprovider`.
- 요청 없이 리팩토링 검증을 위해 장시간 재학습하지 않는다.
