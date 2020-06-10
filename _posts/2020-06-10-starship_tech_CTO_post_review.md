---
layout: post
title: Starship Technology CTO Post Review
featured-img: 2020-06-10-starship_tech_CTO_post_review/starship_cto
---

라스트마일 모빌리티 업체 스타쉽 테크놀로지의 CTO가 4년간의 경험을 공유한 포스트를 전달하고자 한다.
스타쉽의 CTO의 입장에서 글을 전달해보도록 하겠다.

4년간의 여행에서 우리는 컴퓨터비전, 경로 계획, 물체 인식 분야에 대해 개발했다.
우리는 Levenberg-Marquardt알고리즘은 fine tuning했고, 3가지를 개발했다.
- 센서 캘리브레이션 자동화
- 각 배달과정에서 로봇의 배터리가 얼마나 필요한지 예측
- 식당에서 물건을 받기 위해 얼마의 시간이 걸리는지 예측

오늘날의 로봇들은 매우 비싸고, 그것들을 연구를 위해서 만든것이지 commercial한 용도가 아니다. 센서 가격만 1000만원에 가깝다.
또한 컴퓨터 파워가 3000와트나 필요하다. 이것은 작고, 안전한 로봇을 위해서는 비현실적이다.

우리는 위의 문제를 해결하기 위해 여러 부분을 고려하였다.
- Advanced image processing on a lower end computational platform.
- Working around hardware issues in software.
- Tracking how often robots need maintenance, and why.
- Developing advanced route planning systems, to make sure we’re using our network of robots efficiently.

우리는 빠르게 움직이는 startup이고, 이것은 연구 그룹이 안되기 위해 중요하다. 우리 직원들을 연구진이 아니기 때문에 low-cost 하드웨어와 같은 상황에서도 빠르게 문제를 해결할 수 있어야 했다.

데이터는 매우 중요하다. 데이터를 축적한 덕분에 우리는 로봇이 어떻게 움직이는지 분석할 수 있다. 우리는 매주 '데이터 회의'를 개최한다.
이 회의에서 데이터를 통해 찾은 점을 공유하고, 분석한다.

항상 새로운 기술에는 회의론과 두려움이 따라온다. 내가 뉴욕에 로봇과 함께 도착했을 때 공항 직원은 그 로봇은 몇 분만에 도둑맞을 것이라고 했다.그러나 우리의 로봇은 사이렌과 10개의 카메라가 있고, 위치 추정 오차도 2cm밖에 되지 않아서 대처할 수 있다. 실제로 200,000km를 주행했음에도 한번도 도둑맞은 적은없었다.

그리고 사람들은 로봇이 인도에 다니면 무서워 할 것이라고 하였으나 실제로 실험을 했을 때 사람들은 그냥 무시했다. 그러니 사람들의 회의론과 두려움은 그리 큰 문제가 아닐 수 있다.
