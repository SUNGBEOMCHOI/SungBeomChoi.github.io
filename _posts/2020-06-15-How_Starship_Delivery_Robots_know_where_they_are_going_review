---
layout: post
title: How Starship Delivery Robots know where they are going Review
featured-img: 2020-06-15-How_Starship_Delivery_Robots_know_where_they_are_going_review/map
---

### 이 포스트는 배달 로봇 회사 스타쉽 테크놀로지의 포스트인 How Starship Delivery Robots know where they are going의 리뷰이다.
[How Starship Delivery Robots know where they are going](https://medium.com/starshiptechnologies/how-starship-delivery-robots-know-where-they-are-going-c97d385a1015)

스타쉽은 100,000회에 이르는 배달을 해왔다. 
배달하기 위해서는 위치 A에서 B까지 가기 위해서는 경로 계획을 해야하는데 이 과정에서 지도의 정보가 필요하다. 이미 공식화된 구글 지도나 open street maps와 같은 지도가 있음에도 이것의 정보만으로는 부족하다. 왜냐하면 이것들은 차를 기반으로 하고있고, 도로를 매핑하였기 때문이다.
배달로봇은 인도를 통해 이동하고, 횡단보도를 통해 길을 건너기 때문에 정확한 지도가 필요하다. 그렇기 때문에 이 포스트에서는 지도를 만드는 과정에 대해 설명한다. 

첫 번째 단계는 위성지도를 통해 길들을 표시하는 것이다. 인도는 초록색, 횡단보도는 빨간색, 차들이 지나가는 곳은 보라색으로 표현하였다. 이 지도는 노드 그래프로 표현하기 때문에 출발지 A와 목적지 B를 노드로 표현하고 주행하는 경로를 만들고, 가는데 걸리는 거리와 시간을 계산할 수 있다.
![step1](https://raw.githubusercontent.com/SUNGBEOMCHOI/SungBeomChoi.github.io/master/assets/img/posts/2020-06-15-How_Starship_Delivery_Robots_know_where_they_are_going_review/step1.jpg)

두 번째 단계는 로봇이 보는 세계를 만들어주는 것이다. 로봇이 처음 주행할 때는 약간의 수동조작이 필요하다. 로봇이 처음 주행할 때 카메라와 여러 센서를 통해서 데이터를 수집한다. 이미지에서는 특징을 가지는 수천개의 선들을 표시한다. 서버에서는 이 선들을 통해 로봇이 사용할 수 있는 3D 지도를 만든다. 로봇은 이 지도를 통해서 그들이 어디있는지 파악할 수 있다. 
![step2](https://raw.githubusercontent.com/SUNGBEOMCHOI/SungBeomChoi.github.io/master/assets/img/posts/2020-06-15-How_Starship_Delivery_Robots_know_where_they_are_going_review/step2.jpg)

마지막 단계는 위성지도를 통해 만든 그래프와 로봇이 수집한 이미지 데이터를 합치는 단계이다. 로봇이 갈 수 있는 길을 정확히 표시해주는 것이다. 
![step3](https://raw.githubusercontent.com/SUNGBEOMCHOI/SungBeomChoi.github.io/master/assets/img/posts/2020-06-15-How_Starship_Delivery_Robots_know_where_they_are_going_review/step3.gif)

물론 세상은 계절이나 건물을 짓는 등의 활동으로 인해서 변한다. 세세한 부분에 대해서는 데이터의 양이 충분하기 때문에 robust하다.  큰 변화에 대해서는 업데이트를 해주어야한다. 이것은 로봇이 매일 주행한 데이터를 통하여 업데이트를 해주게된다. 
