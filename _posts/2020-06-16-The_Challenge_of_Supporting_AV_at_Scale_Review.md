---
layout: post
title: The Challenge of Supporting AV at Scale 리뷰
featured-img: 2020-06-16-The_Challenge_of_Supporting_AV_at_Scale_Review/mobileye
permalink: /things/The_Challenge_of_Supporting_AV_at_Scale_Review/
category: things
---

#### 이 포스트는 The Challenge of Supporting AV at Scale에 대한 리뷰이다. 
[The Challenge of Supporting AV at Scale](https://medium.com/@amnon.shashua/the-challenge-of-supporting-av-at-scale-7c06196cced2)

 인텔에 한화 17조에 인수된 자율주행차량 회사인 모빌아이의 자율주행 기술에 대해서 다루고 있다.
이 회사는 1월에 25분짜리 무편집본으로 예루살렘의 busy traffic환경의 자율주행영상을 올렸다. 진짜 디테일과 기술에 대단하다는 생각이 들었고, 25분동안 시간가는줄 모르고 봤다. 이 영상에서 유일하게 사람의 조작이 들어간 부분은 차를 찍는 드론의 배터리를 교체할 때였다.

<style>.embed-container { position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; } .embed-container iframe, .embed-container object, .embed-container embed { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }</style><div class='embed-container'><iframe src='https://www.youtube.com/embed/kJD5R_yQ9aw' frameborder='1' allowfullscreen></iframe></div>
                  

**영상에서 발견한 부분과 설명들에 대해 정리해본다.**
- 카메라만을 통한 자율주행시스템을 구축했다.
- 주행하는 도로에 대한 200km의 HD맵이 미리 구축되어있다.
- 12개의 카메라를 통해 3d map을 생성한다.(8개의 long-range 카메라와 4개의 parking 카메라) 이 과정을 단지 2개의 EyeQ5(모빌아이의 autonomous driving을 위한 칩)를 통해 이뤄낸다.
- HD맵에서 오브젝트(차, 사람 등등)에 나타나는 색은 situation을 의미한다. (빨간색, 노란색, 파란색이 나타나는데 위험도를 나타낸다는 추측을 해본다.)
- 객체인식 기술을 통해 장애물을 감지한다.
- 최대 시속은 90km/h까지 낼 수 있다.
- HD맵에서 차선의 중앙은 점선으로 표시되고, 과속방지턱을 표현한다.
- HD맵에 도로에 표시된 사인이나 교통 표지판과 같은 구체적인 부분은 보이지 않음. (시각화만 안해놓았을 듯 하지만 실제로 해놓지 않았다면 예를들어 시속이 30km/h미만의 구역일 경우에는 limit를 걸어주지 않아서 문제가 있을 것으로 생각된다.)
- **예루살렘의 경우 우리나라와는 달리 매우 robust한 환경이다. 차로에 차가 주차되어 있는 경우가 많다. 모빌아이에서는 차가 주행 중인지 주차된 차인지 구별할수 있다.**
- HD 맵은 정확도와 데이터의 풍부함도 중요하지만 업데이트도 중요하다. 모빌아이는 Road Experience Management technology를 통해 매우 빠른 주기로 업데이트를 하고 있다. 이것은 크라우드 소싱을 base로 하여 맵을 만드는 것을 자동화한 것이다. 모빌아이의 ADAS기술이 들어간 여러 vehicles를 통하여 데이터를 수집하고, 이 데이터는 클라우드로 전송된 후 익명화 되어 HD맵을 생성하게 된다.
- 횡단보도로 지나가는 사람을 구분한 후 정지하고, 사람이 지나간후에 움직이게 된다.
- 신호등의 신호를 구별할 수 있다.
- 차로에 나타나는 오렌지선은 대중 교통이 지나다니는 도로를 나타내고, 빨간선은 light rail(트램같은 교통수단)라인이다.
- 좁은 일방통행길을 지나갈 때 사람이 길을 건너려는 뉘앙스일 때는 지나가는 것을 기다리는데 일정 시간이 지나도 건너지 않으면 건너가지 않는것으로 판단하고, 지나간다. (10:04)
- 움직이는 사람과 정지한 사람을 구분한다.
- **신호등이 빨간불이고, 차로에 차가 정지되어 있는 경우에 이 차가 현재 주행 중인 차인지 그냥 멈춰있는 차인지 구별이 어려운데 이것을 구분하여 옆 차선으로 이동한다. (13:15)**
- 차량의 위치 추정이 잘 안된건지, 물체의 위치 추정이 잘 안된건지 HD맵에 나타난 거리와는 약간 달라서 불안한 느낌이 있음. (16:01)
- **차문 여는것도 감지한다. 차만 인식하는게 아니라 차량의 상황도 인지하는 것으로 보인다. (19:30)**
- **트레일러에 여러 대의 차가 있어도 한대의 차로 인식한다.(21:28)**

 2017년에 모빌아이는 safety concept을 발표했다. 이것은 2가지의 관측을 base로 하고 있다. 

 첫 번째는 사고는 판단 착오에 의해 일어나는데 이것은 formal manner와 같은 부분을 명확히 해줌으로서 제거될 수 있다는 것이다. Decision making은 안전과 유용성의 균형을 이루어야한다.(안전을 위해서 완전히 느려지면 안되고, 너무 빠르기 위해 안전에 소홀해서도 안된다.) 모빌아이의 Responsibility Sensitive Safety(RSS)모델은 운전자의 추정을 파라미터로 생성했다.(교차로나 차선이 줄어들 때의 운전자들의 행동 등) RSS모델은 최악의 시나리오를 추정한다. RSS 이론은 만약 추정만 제대로 되고, 이에 따른 미리 정해진 행동을 취한다면 사고가 나지 않을 것임을 말한다. 이를 위해 2019년 말 IEEE에서는 자율주행기술의 decision making을 위한 표준을 정하기 위해 새로운 그룹을 생성했다.

 두 번째는 아무리 좋은 decision making을 한다고 하더라도 인지에서 실수가 일어나면 안된다는 것이다. 물체가 있는데도 없다고 인지되는 문제가 발생할 가능성이 0은 아니다. 이런 부분을 줄이기 위해서 우리는 인지 시스템을 카메라를 통한 시스템과 레이더, 라이더를 통한 시스템, 총 두 가지 서브시스템을 이용하고 있다. 이것은 다른 여러 센서를 통합하여 하나의 인지 시스템을 만드는 다른 회사와는 차별점을 가지고 있다. 카메라만을 통한 인지는 정확하지 않은 depth 추정, perspective, 명암 등과 같은 이유로 어려우나 모빌리티는 이것을 해냈다. 카메라만을 통한 자율주행기술의 배경을 CES에서 설명한 바 있다. 
[CES 2020: An Hour with Amnon - Autonomous Vehicles Powered by Mobileye](https://www.youtube.com/watch?v=HPWGFzqd7pI&t=7s)

 미국에서는 3.2 trillion의 마일을 1년동안 운전하게 되는데 사고는 600만회 일어난다. 이것을 평균시속은 10 mile/h로 가정하여 계산하면 mean-time-between-failures (MTBF)는 50,000시간이다. 모빌아이는 이것의 10배, 100배, 1000배 좋은 자율주행기술을 design하고 있다. 100,000대의 자율주행 차를 운영한다고 하면 10배의 MTBF는 매일 사고가 나는 것이고, 100배는 매주 한 번, 1000배는 분기당 한 번 사고가 나는 것이다. 사회적인 관점에서 보면 10배의 MTBF만 되어도 큰 진전이지만 재정적으로나 공적으로는 좋은 용납할 수 없는 정도이다. 그래서 1000배의 MTBF는 되는 것이 필수적이다. 
