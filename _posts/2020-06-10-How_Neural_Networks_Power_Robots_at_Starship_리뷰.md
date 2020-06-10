---
layout: post
title: How Neural Networks Power Robots at Starship 리뷰
featured_img: 2020-06-10-How_Neural_Networks_Power_Robots_at_Starship_review/starship_technology
---

스타쉽이 어떻게 적은 computational resources와 lidar와 같은 비싼 센서를 쓰지 않으면서 실시간으로 잘 주행할 수 있을까? 

### Using machine learning to detect objects
자율주행차의 영역의 경우 자율주행로봇 영역에 비해 more structured and predictable하다. 그리고 라인을 따라서 가고, 방향을 자주 바꾸지 않는다. 
그러나 자율주행로봇의 영역은 사람은 자주 방향을 바꾸고, 개들도 있다. 이런 상황을 인지하기 위해 물체 인식 모듈이 필요하다. 

### improving the robot’s ability to adapt and learn
로봇 소프트웨어 안에는 훈련할 수 있는 units(mostly neural net)이 있고, 이것을 알아서 학습한다.
우리들은 모델을 훈련시킬 때 여러 요소를 더 자세히 생각했다.
- 창문에 비친 차에 대해서는 어떻게 페널티를 줄 것인가
- 포스터에 있는 사람이 비춰지면 어떻게 하는가
- 트레일러위에 있는 모든 차들을 다 체크해야할까
![od_concenrn](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-06-10-How_Neural_Networks_Power_Robots_at_Starship_review/od_concenrn.jpg?raw=true)

### 효과적으로 데이터 사용하기
사람과 차의 사진은 많지만 오토바이나 스케이트타는 사람에 대한 자료는 부족하다. 이런 자료들은 더 확보를 해야한다. 
또한 여러 나라들과 여러 날씨, 계절에 대한 데이터도 필요하다.
보통 pixel-wise segmentation를 통해 인도인지 도로인지 구분한다. 
우리의 경우 더 좋은 모델을 얻기 위해 global image-level clues를 암호화하여 뉴럴넷에 넣어주었다.(prior knowledge를 뉴럴넷에 적용)

### Neural networks in resource-constrained settings
딥러닝은 컴퓨팅파워가 많이 필요하고, 이것은 큰 문제이다. 
객체 인식분야에서 state of art인 maskRCNN의 경우 5FPS를 지원하고, real-time 객체 인식 모듈도 100FPS 보다 낮다.
우리는 360도에 대한 이해가 필요해서 5개의 이미지에 대해 객체 인식 모듈을 적용해야한다.
우리는 5장의 이미지에 대해 2000FPS에서 작동할 수 있다. (한 장의 이미지에 대해서는 10000FPS)

### Fixing neural network bugs is challenging
아래 그림처럼 너무 넓게 bounding box를 그리거나 낮은 confidence(확신)의 box이거나 다른 박스를 그리는 것에 대해 해결해야했다.
![od_bug](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-06-10-How_Neural_Networks_Power_Robots_at_Starship_review/od_bug.jpg?raw=true)

뉴럴넷은 속을 알 수 없는 black box이기 때문에 우리는 분석하기 위해 웨이트들을 시각화하는 과정이 필요했다.
첫번째 층은 가로, 세로와 같은 간단한 특성을 익혔고, 두 번째 레이어부터는 좀더 복잡한 특성을 학습했다.
![weight_visualize](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-06-10-How_Neural_Networks_Power_Robots_at_Starship_review/weight_visualize.jpg?raw=true)

### Challenges in using neural networks in real-world production systems
우리들은 여러 환경에 대해 대처하기 여러 모델을 만들었다. 이 모델들은 상황에 따라 적용되는 성능이 다르다.
전혀 학습되지 않은 데이터에 대해서는 다른 센서들의 도움을 받거나 사람의 원격 조종의 도움을 받는다.
![different_models_at_different_env](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-06-10-How_Neural_Networks_Power_Robots_at_Starship_review/different_models_at_different_env.jpg?raw=true)

