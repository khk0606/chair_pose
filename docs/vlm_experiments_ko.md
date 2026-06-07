# Seat-Contact VLM 실험 보고서

## 목표

의자 전체가 아니라 사람이 실제로 엉덩이로 접촉할 수 있는 좌판 표면을
VLM이 이미지에서 찾아 좌표로 출력하도록 파인튜닝한다.

베이스 모델은 `HuggingFaceTB/SmolVLM-500M-Instruct`, 학습 방식은 PEFT LoRA,
실행 장치는 Apple Silicon MPS를 사용했다.

## 데이터

- LabelMe 이미지/JSON 쌍
- 라벨 이름: `seat_contact`
- 원본 유효 annotation: 150개
- 중복 stem 제거 후 유니크 샘플: 약 132개
- hardcase 복제/추가 후 최근 데이터셋: 182개
- `v18` split: train 164, validation 18

파일 수에 비해 유니크한 의자 형태와 촬영 조건은 적다. 중복 annotation은 거의
같은 polygon이어서 데이터 다양성을 크게 늘리지 못했다.

## 실험 변화

| Version | Target | Input | Mean IoU (all) | Hardcase | 결과 |
| --- | --- | --- | ---: | ---: | --- |
| v14 | 12-point polygon | full image, 224 | 0.0103 | 실패 | JSON 형식 일부 학습, geometry 붕괴 |
| v15 | polygon + hardcase oversampling | full image, 224 | 0.0037 | 0.0132 | 기존 샘플 복제로 개선되지 않음 |
| v16 | 4-point quad | full image, 224 | 약 0.0001 | 0.0 | 네 점이 한 점으로 collapse |
| v17 | pixel bbox | full image, 224 | 0.0161 | 0.0 | 좌상단 고정 박스 출력 |
| v18 | 16x16 grid bbox | chair crop, 224 | 0.0935 | 0.0 | 평균 개선, JSON schema 불안정 |

버전별 데이터 split이 완전히 동일하지 않아 수치는 절대적인 모델 순위보다
실험 방향의 변화로 해석해야 한다.

## v18

문제 설정:

- COCO chair detector로 의자 ROI crop
- crop을 `224 x 224`로 resize
- seat-contact bbox를 `16 x 16` grid로 양자화
- VLM은 grid index 네 개를 포함한 JSON 생성

학습:

- max steps: 45
- learning rate: `2e-5`
- batch size: 1
- gradient accumulation: 8
- completion-only loss
- final train loss: `0.6028`
- final eval loss: `0.4380`

`checkpoint-45` validation:

| Metric | Value |
| --- | ---: |
| Valid JSON rate | 0.5556 |
| Schema-valid rate | 0.5000 |
| Mean IoU, all | 0.0935 |
| Mean IoU, schema-valid only | 0.1871 |
| Median IoU, schema-valid only | 0.1758 |
| Best sample IoU | 0.2930 |

## 확인된 실패 패턴

1. Python dictionary 형식

모델이 JSON의 큰따옴표 대신 작은따옴표를 출력한다.

2. 중복 필드

`seat_contact_box` 내부에 같은 좌표 key를 여러 번 출력한다.

3. schema 혼합

`image_size` 내부에 bbox 좌표를 넣거나, 학습에 없는 `box_min_x` 같은 필드를
추가한다.

4. 상수 좌표 collapse

어려운 이미지에서 전체 grid `(0, 0, 16, 16)` 또는 작은 고정 박스를 출력한다.

5. hardcase 일반화 실패

대표 hardcase 한 장에서 valid JSON과 schema-valid가 모두 0이고 IoU도 0이었다.
따라서 현재 체크포인트를 실제 좌석 검출기로 사용할 수 없다.

## 해석

chair crop과 grid 좌표는 full-image pixel 좌표보다 효과가 있었다. 검증 Mean IoU가
약 `0.016`에서 `0.0935`로 상승했기 때문이다.

하지만 현재 모델은 시각적 localization과 정형 텍스트 생성을 동시에 안정적으로
수행하지 못한다. 특히 데이터 분포 밖 이미지에서는 좌판을 찾기 전에 출력 schema가
무너진다.

학습 loss 감소만으로 localization 성공을 판단하면 안 된다. 체크포인트 선택에는
반드시 다음 지표가 함께 필요하다.

- JSON validity
- schema validity
- Mean/Median IoU
- hardcase IoU
- 실제 overlay 확인

## 다음 실험

1. 출력 형식을 `{"box":[x1,y1,x2,y2]}`로 단순화한다.
2. 256개 좌표 token 대신 더 작은 categorical region 표현을 검토한다.
3. hardcase와 유사한 새로운 의자 이미지를 추가한다.
4. 같은 의자의 복제보다 형태, 재질, 각도, 배경 다양성을 늘린다.
5. constrained decoding 또는 grammar-based decoding을 적용한다.
6. random validation과 별도로 고정 hardcase validation을 운영한다.
7. 좌석 위치를 맞힌 샘플과 JSON 형식만 맞힌 샘플을 분리해 분석한다.
