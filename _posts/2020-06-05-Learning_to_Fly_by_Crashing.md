---
layout: post
title: Learning to Fly by Crashing 논문 리뷰
featured-img: Learning_to_Fly_by_Clashing/Learning_to_Fly_by_Crashing_모델구조
---

## 목적
적은 비용으로 indoor 환경에서 드론이 장애물과 부딪히지 않는 자율주행 학습
[(영상)](https://www.youtube.com/watch?v=HbHqC8HimoI)

## 선행연구
localization 및 path_planning 에 대한 연구들
1. slam
  *  비싼 computational cost으로 real time에는 적합하지 않음
2. depth estimation method
  * 짧은 배터리와 적재 용량의 문제
3. stereo vision bsaed estimation
  * 하얀 벽과 같은 plain surface에서는 localization이 힘듬
4. monocular camera based methods use vanishing points as a guidance
  * 여전히 range sensor에 의존
5. 사람이 운행한 데이터를 통한 학습
  * 데이터를 수집하기 어렵고, 실패에 대한 데이터를 찾기가 힘듬.
6. 시뮬레이션을 통한 학습
  * 실제 환경과의 괴리가 있어 제대로 적용이 안됨.

RL을 통한 학습은 많은 데이터가 필요하기 때문에 supervised learning을 통해 학습을 사용함.

## 연구과정
1. 데이터 수집
  * 수동 컨트롤 수집 -> 사람의 intuition이 들어가므로 X
  * slam based인 자율주행을 통한 수집 -> bias issue로 X 
  * naive random straight line traectories로 수집
    - 랜덤으로 주행하다가 부딪히면 origin으로 돌아오고, 다시 실험하고를 반복하여 데이터를 수집
    - origin위치로 돌아오는 것은 IMU가 row accuracy이기 때문에 PTAM모듈을 사용함 
    [PTAM 참고](https://darkpgmr.tistory.com/129)
  
2. 데이터 전처리
  * 이미지를 close to the colliding object(negative data)와 그 외의 이미지(positive data)로 나눔
  
3. learning methodology
  * 현재 이미지를 토대로 지금 방향으로 진행할지 아닐지 총 2가지 클래스 중 선택하는 모델을 만듬
  * 넣어주는 이미지는 왼쪽, 정면, 오른쪽 이미지로, 한 timestamp에서 모델을 총 3번 돌려서 최적의 방향을 결정
  * AlexNet-architecture를 사용
  * initialization을 위해 imageNet-Pretrained weights를 사용
  
  policy 학습 의사코드: 
  
  ![policy 학습 의사코드](https://raw.githubusercontent.com/SUNGBEOMCHOI/SungBeomChoi.github.io/master/assets/img/posts/Learning_to_Fly_by_Clashing/Policy_for_flying_indoor.jpg)
  
  Learning to Fly by Crashing 모델 구조:
  
  ![Learning to Fly by Crashing 모델 구조](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/Learning_to_Fly_by_Clashing/Learning_to_Fly_by_Crashing_%EB%AA%A8%EB%8D%B8%EA%B5%AC%EC%A1%B0.jpg?raw=true)
  
## 연구결과
### 비교군은 총 3가지임.
1. straight line policy : 그냥 직진만 함
2. Depth prediction based policy : monocular camera를 통한 depth map을 이용하는 방법
3. 사람이 직접 운행하는 방법 : 사람은 드론에서 전송되는 monocular 이미지를 보고, 조이스틱으로 사람이 직접 운행

### 실험장소는 총 6장소
  - Glass door
  - NSH 4th Floor
  - NSH Entrance
  - Hallway
  - Hallway with Chairs
  - Wean Hall
  
![연구 결과](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/Learning_to_Fly_by_Clashing/%EC%8B%A4%ED%97%98%EA%B2%B0%EA%B3%BC.jpg?raw=true)

### 결과 분석
- glass door과 같은 투명하거나 반사성이 있는 물체는 depth 측정이 힘들다. 본 연구는 문에 있는 걸쇠 등과 같은 특징으로 학습한 듯하다.
- hallway with chair 환경같은 cluttered한 환경에서는 사람보다 더 좋은 성능을 냈다.
- hallway는 untextured environment에서는 depth base모델이 학습이 힘든듯하다.
