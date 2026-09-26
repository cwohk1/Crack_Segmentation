# 프로젝트 작업 규칙

- WSL Ubuntu에서 `.venv312/bin/python`으로 실행한다.
- 최상위 실행 파일은 `trainer.py`, `test.py`, `dataloader.py`, `hub.py`로 유지한다. `hub.py`는 Hugging Face 전송을 담당한다.
- 모델 구현과 선택 함수는 `models/scripts/`, 체크포인트와 실행 결과는 `models/weights/<model>/<dataset>/`에 함께 둔다. 여러 실험은 그 아래 `<experiment>/`로 구분한다.
- 설정·학습 이력·평가 JSON·로그·예측 이미지는 해당 가중치 폴더에 둔다. 작업용 테스트는 `Codex/test_pipeline.py`에 둔다.
- 예: `models/weights/unet_18/deepcrack/`에 `best.pt`, `last.pt`, `config.json`, `history.json`, `test.json`, `train.log`, `predictions/`를 함께 둔다. `Codex/runs/`를 다시 만들거나 기록과 가중치를 별도 경로로 분리하지 않는다.
- 모델 폴더명은 `models/scripts/__init__.py`의 `MODEL_DIRS`를 따른다: `segnet`, `segnet_vgg16`, `fphbn`, `unet_18`, `unet_resnet50`, `unet_vgg16`. 모델 선택 이름과 폴더 이름을 혼동하지 않는다.
- 학습·평가·추론·다운로드가 같은 결과 폴더를 사용하게 한다. 새 튜닝 실험은 `<model>/<dataset>/<experiment>/`로 구분하고, 기존 결과를 덮어써서 실험을 시작하지 않는다. 재개 시 실제 체크포인트의 부모 폴더를 사용한다.
- 다운로드 데이터와 출처 설명은 `datasets/`에 유지한다. 데이터 준비 도구는 `datasets/prepare.py` 하나로 관리한다.
- Hugging Face 데이터셋은 비공개 dataset 저장소 `cwohk/crack-segmentation`의 `datasets/`에, 가중치는 공개 model 저장소 `cwohk/crack-segmentation-weights`의 `weights/<model>/<dataset>/`에 둔다. 실험 하위 폴더도 로컬과 동일하게 유지한다.
- 전송은 `hub.py`를 사용한다. 예: `python hub.py upload weights unet_18/deepcrack`. 저장소 변경은 `--repo`, `HF_DATASET_REPO`, `HF_MODEL_REPO`로 지정하며 토큰은 코드·설정 파일·Git에 기록하지 않는다.
- 데이터셋은 실제 파일 내용과 출처·검증 정보를 ZIP에 담고, 가중치는 `.pt` 그대로 설정·이력·평가 기록과 함께 전송한다. 다운로드 시 체크섬을 검증한다. 예측 이미지는 기본 전송에서 제외하고 필요한 경우에만 `--predictions`를 사용한다.
- 데이터셋의 비공개 상태를 유지하고 사용자의 명시적 요청 없이 공개로 변경하지 않는다. 공개 가중치에 원본 이미지나 overlay를 자동으로 포함하지 않는다. 출처 표기를 재배포 허가로 간주하지 않는다.
- 폴더 이동 시 가중치와 기존 결과를 보존하고 파일 해시로 이동 전후 동일성을 확인한다. 원격 경로를 바꿀 때는 새 경로 업로드와 검증을 먼저 마친 뒤 이전 경로를 정리한다. 커밋 이력 삭제나 저장 용량 회수를 임의로 수행하지 않는다.
- 일회성 조사·수정 스크립트나 새로운 최상위 폴더를 남기지 않는다. 단, 아래 장기 학습 절차의 `tmp/`는 분석 완료 전까지 허용한다.
- 사용자 변경을 보존한다. 데이터 원본, 학습 가중치, 가상환경은 임의로 삭제하지 않는다.
- 모델 구조를 정리할 때 기존 state_dict와 optimizer 재개 호환성을 유지한다.
- 모델은 이해하기 쉬운 단순한 구현을 우선한다. 불필요한 추상화, 옵션, 파일 분리를 늘리지 않는다.
- 새 모델도 기존 `build_model(name, pretrained)`, N×1×H×W logits 출력, 학습·평가·체크포인트 흐름을 유지한다. 모델별 추가 처리는 해당 모델 파일에 모으고 공통 코드 변경은 최소화한다.
- 논문은 핵심 아이디어와 구조를 참고하되 완전히 동일하게 재현할 필요는 없다. 핵심 원리는 유지하고 프로젝트에 맞게 단순화하며, 달라진 구조·손실·학습 조건과 생략한 부분을 코드 주석과 README에 명시한다. 단순화한 구현을 논문의 완전 재현이나 동일 성능으로 표현하지 않는다.
- 코드 변경은 관련 테스트로 검증하고, 실행하지 않은 학습·평가 결과를 주장하지 않는다.
- 기본 검증: `.venv312/bin/python -m pytest -q Codex/test_pipeline.py -p no:cacheprovider`.
- 전체 모델 튜닝 계획은 `models/weights/<model>/training-plan.json`과 `training-plan.md`에, 비교 요약은 `<model>/<dataset>/tuning-summary.json`에 둔다. 조건별 기록은 `<model>/<dataset>/<experiment>/`에 저장한다. 데이터 분할 seed와 초기화 seed를 구분하고, 조건 선택은 검증 데이터로만 수행한다. 충분한 학습 요청에서는 고정 epoch 도달뿐 아니라 검증 정체 여부도 확인한다.
- 요청 없이 리팩토링 검증을 위해 장시간 재학습하지 않는다.

## 장기 학습 실행과 사후 분석

- 5분 이상 걸릴 것으로 예상되는 학습은 Codex가 직접 실행하거나 기다리지 않는다. 필요한 파라미터 조합과 연속 실행 명령을 `tmp/train.sh` 하나에 모두 작성한다.
- 스크립트는 WSL Bash에서 프로젝트 루트를 기준으로 `.venv312/bin/python -u`를 사용한다. 실행 권한을 부여해 사용자가 프로젝트 루트에서 `./tmp/train.sh` 한 번으로 전체 학습을 순차 실행할 수 있게 한다.
- 모든 실행의 표준 출력과 표준 오류를 `tmp/train.log`에 저장한다. 화면에도 출력할 경우 `tee`를 사용할 수 있다. 조건별 가중치와 영구 학습 이력은 기존 `models/weights/<model>/<dataset>/<experiment>/` 구조에 계속 저장한다.
- 스크립트 작성 후에는 실행 방법만 알려주고 즉시 턴을 종료한다. 학습을 직접 시작하거나, 완료를 기다리거나, 중간 상태를 조회하거나, polling하지 않는다. 백그라운드 실행이나 자동 모니터링으로 이 규칙을 우회하지 않는다. 실행하지 않는 구문 검사(`bash -n`)는 허용한다.
- 사용자가 학습이 끝났다고 알려준 뒤에만 `tmp/train.log`와 기존 history를 읽고 결과를 분석한다.
- 분석이 끝나면 필요한 분석 내용과 선정 근거를 해당 실험의 history 및 결과 기록에 반영한다. 원래 측정된 epoch별 수치는 임의로 바꾸지 않으며, 보존할 로그와 실패 원인도 영구 결과 폴더에 남긴다.
- 필요한 기록을 보존한 직후 `tmp/` 아래 임시 파일을 삭제한다. 삭제 대상이 프로젝트의 `tmp/` 내부인지 확인하고, 데이터셋·가중치·영구 history는 삭제하지 않는다.
- `tmp/` 전체는 Git에서 제외한다.
