# Hair detection

# HBB와 OBB 기반 모발 객체 검출 라벨링 기법 비교 분석
<https://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE11743304>

<img src="https://github.com/gyugyukim/Survival-analysis/assets/135569056/18085a28-b777-41b1-a2ab-3a3f549ca616">
----

# Data Class Information
<img src=https://github.com/gyugyukim/Survival-analysis/assets/135569056/f3dd2387-85d5-4eca-910b-1be64d17c34c>
<img src=https://github.com/gyugyukim/Survival-analysis/assets/135569056/b682a6f3-6cd1-4fb6-8633-83e9fbca5b60>

(a) Type1 - 하나의 모낭 근처에 모발이 한 개
(b) Type2 - 하나의 모낭 근처에 모발이 두 개
(c) Type3 - 하나의 모낭 근처에 모발이 세 개
(d) Type4 - 하나의 모낭 근처에 모발이 네 개 이상

# Model
1. YOLOv5
2. YOLOv5OBB

# Performance metrics

1. mean average precision(mAP)
2. Mean Absolute Error(MAE)

----

# Results

<img src=https://github.com/gyugyukim/Survival-analysis/assets/135569056/b28556ef-d1c6-4c1c-9784-053d04802a02>

- Batch Size 별 mAP 성능 비교
<img src=https://github.com/gyugyukim/Survival-analysis/assets/135569056/72c9252b-b2f5-4b36-91f4-c2ae2acf5680>

- Batch size에서 우수한 성능 기준, Class 별 mAP 성능 비교
<img src=https://github.com/gyugyukim/Survival-analysis/assets/135569056/0d2e881a-2569-48b1-a56a-452ab47acd74>

- 프랙탈 구조를 가진 모발의 특성에서 HBB의 경우, Type 1 과 Type 2의 특징을 함께 가지고 있는 Type 3 경우에서 검출 시 어려움을 겪는 경향을 확인
- OBB의 경우, 정확히 모발의 유형을 검출하는 반면 노란색 박스처럼 모델이 과소 예측 하는 경향이 존재하는 것을 확인
- Bounding Box 방법에 따라 강점을 가지는 경우가 존재함을 확인
  
<img src=https://github.com/gyugyukim/Survival-analysis/assets/135569056/cee5150d-f591-4ea8-968e-572762820f98>

- 평균 절대 오차 기준,  HBB는 2.89개, OBB는 3.30개로 나타나는 것을 확인.
----
# HBB와 OBB의 결과를 비교 분석

- mAP와 MAE의 결과 값들을 비교하기 위해 t-검정을 진행
- mAP의 경우, p-value는 유의수준 0.05 하에, 0.1299로 두 방법 간에 유의미한 차이가 없음을 보임
- MAE의 경우, p-value는 유의 수준 0.05 하에, 0.0254로 Bounding Box 방법에 따라 모발 개수 예측률에 유의미한 차이가 존재하는 것을 확인.
- 이것은 실제로 모델이 예측한 바운딩 박스를 기준으로, OBB를 통한 검출이 과소 예측되어 이러한 차이가 MAE 값의 증가를 초래한 것으로 예상
