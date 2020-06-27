---
layout: post
title: Modeling of 4Wheel Skid Steering Mobile Robot
featured-img: 2020-06-19-Modeling_of_4Wheel_Skid_Steering_Mobile_Robot/capture2
---

#### 이글은 4wheel skid steering mobile robot의 물리적 모델링에 관한 포스트이다.

4wheel skid steering mobile robot(SSMR)은 robust한 환경에서도 잘 견딜 수 있는 성질덕분에 우주 탐사 등과 같은 환경에서 쓰이는 형태의 로봇이다. 그러나 제어에 있어서는 challenging task이다. 

### Robot Coordinate and World Coordinate
![robot coordinate and world coordinate](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-06-19-Modeling_of_4Wheel_Skid_Steering_Mobile_Robot/capture2.jpg?raw=true)

로봇의 운동을 규정하기 위해서는 좌표계를 생각해볼 필요가 있다. 좌표계는 robot coordinate와 world coordinate로 나뉘게 된다.

SSMR에서는 z축 방향이나 roll, pitch의 운동은 생각할 필요가 없다. 따라서 로봇의 속도에 대해 robot coordinate에서는 linear velocity는 <img src="https://latex.codecogs.com/gif.latex?\left[\begin{array}{lll}v_{x}&space;&&space;v_{y}&space;&&space;0\end{array}\right]^{T}" title="\left[\begin{array}{lll}v_{x} & v_{y} & 0\end{array}\right]^{T}" /> 이고, angular velocity는 <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\omega}=\left[\begin{array}{ccc}0&space;&&space;0&space;&&space;\omega\end{array}\right]^{T}" title="\boldsymbol{\omega}=\left[\begin{array}{ccc}0 & 0 & \omega\end{array}\right]^{T}" />로 쓸 수 있다. 

world coordinate상에서 상태는 X, Y 위치와 X축과의 각도 $\theta$로 표현할 수 있다.  따라서 world coordinate상에서 로봇의 상태는 <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{q}=\left[\begin{array}{lll}X&space;&&space;Y&space;&&space;\theta\end{array}\right]^{T}" title="\boldsymbol{q}=\left[\begin{array}{lll}X & Y & \theta\end{array}\right]^{T}" />로 표현하고 속도는 <img src="https://latex.codecogs.com/gif.latex?\dot{\boldsymbol{q}}=\left[\begin{array}{lll}\dot{X}&space;&&space;\dot{Y}&space;&&space;\dot{\theta}\end{array}\right]^{T}" title="\dot{\boldsymbol{q}}=\left[\begin{array}{lll}\dot{X} & \dot{Y} & \dot{\theta}\end{array}\right]^{T}" />이다. 

world coordinate와 robot coordinate간의 속도변환은 회전변환행렬을 사용하면 아래와 같이 나타낼 수 있다. 
<img src="https://latex.codecogs.com/gif.latex?\left[\begin{array}{c}&space;\dot{X}&space;\\&space;\dot{Y}&space;\end{array}\right]=\left[\begin{array}{cc}&space;\cos&space;\theta&space;&&space;-\sin&space;\theta&space;\\&space;\sin&space;\theta&space;&&space;\cos&space;\theta&space;\end{array}\right]\left[\begin{array}{c}&space;v_{x}&space;\\&space;v_{y}&space;\end{array}\right]" title="\left[\begin{array}{c} \dot{X} \\ \dot{Y} \end{array}\right]=\left[\begin{array}{cc} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{array}\right]\left[\begin{array}{c} v_{x} \\ v_{y} \end{array}\right]" />

로봇 좌표계에서 yaw축에 대한 회전은 world 좌표계에서 yaw축의 회전과 동일하다. 따라서  <img src="https://latex.codecogs.com/gif.latex?\dot{\theta}=\omega" title="\dot{\theta}=\omega" /> 로 표현할 수 있다.

따라서 로봇좌표계와 world 좌표계의 속도에 대한 완전한 변환은 다음과 같이 쓸 수 있다. 

<img src="https://latex.codecogs.com/gif.latex?\left[\begin{array}{c}&space;\dot{X}&space;\\&space;\dot{Y}&space;\\&space;\dot{\theta}&space;\end{array}\right]=\left[\begin{array}{ccc}&space;\cos&space;\theta&space;&&space;-\sin&space;\theta&space;&&space;0&space;\\&space;\sin&space;\theta&space;&&space;\cos&space;\theta&space;&&space;0&space;\\&space;0&space;&&space;0&space;&&space;1&space;\end{array}\right]\left[\begin{array}{c}&space;v_{x}&space;\\&space;v_{y}&space;\\&space;w&space;\end{array}\right]" title="\left[\begin{array}{c} \dot{X} \\ \dot{Y} \\ \dot{\theta} \end{array}\right]=\left[\begin{array}{ccc} \cos \theta & -\sin \theta & 0 \\ \sin \theta & \cos \theta & 0 \\ 0 & 0 & 1 \end{array}\right]\left[\begin{array}{c} v_{x} \\ v_{y} \\ w \end{array}\right]" />

### 바퀴의 운동 modeling
![wheel](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-06-19-Modeling_of_4Wheel_Skid_Steering_Mobile_Robot/capture3.jpg?raw=true)

i번째 바퀴의 회전 속도를 <img src="https://latex.codecogs.com/gif.latex?\omega_{i}(t)" title="\omega_{i}(t)" />라 하자. 

여기서는 3가지 가정을 한다.
 - 여기서는 바퀴의 두께는 무시한다.
 - 지면과 한점 <img src="https://latex.codecogs.com/gif.latex?P_{i}" title="P_{i}" />에서 만난다.
 - longitudinal 방향으로의 미끄러짐은 없다

로봇이 정면 방향으로 움직이지 않는 이상 바퀴의 lateral 방향 속도는 0이 아니다. 바퀴는 longitudinal 방향으로는 완전히 구르고, lateral 방향으로는 미끄러지게 된다. 

longitudinal 방향으로의 속도는 다음과 같이 바퀴의 반지름과 바퀴의 각속도로 표현할 수 있다.
<img src="https://latex.codecogs.com/gif.latex?v_{i&space;x}=r_{i}&space;\omega_{i}" title="v_{i x}=r_{i} \omega_{i}" />

### 4바퀴의 운동이 합쳐져 만들어지는 로봇의 움직임
![robot and wheel](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-06-19-Modeling_of_4Wheel_Skid_Steering_Mobile_Robot/capture4.jpg?raw=true)

로봇 회전의 중심(ICR)으로부터 각 i 바퀴와 로봇의 무게중심 COM으로 가는 거리벡터를 다음과 같이 쓸 수 있다.

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{d}_{i}=\left[\begin{array}{ll}&space;d_{i&space;x}&space;&&space;d_{i&space;y}&space;\end{array}\right]^{T}&space;\text&space;{&space;and&space;}&space;\boldsymbol{d}_{C}=\left[\begin{array}{ll}&space;d_{C&space;x}&space;&&space;d_{C&space;y}&space;\end{array}\right]^{T}" title="\boldsymbol{d}_{i}=\left[\begin{array}{ll} d_{i x} & d_{i y} \end{array}\right]^{T} \text { and } \boldsymbol{d}_{C}=\left[\begin{array}{ll} d_{C x} & d_{C y} \end{array}\right]^{T}" />

ICR에 대한 i 번째 바퀴의 회전 속도와 COM의 회전속도는 같으므로 다음과 같이 쓸 수 있다. (스칼라 개념)

<img src="https://latex.codecogs.com/gif.latex?\frac{\left\|\boldsymbol{v}_{i}\right\|}{\left\|\boldsymbol{d}_{i}\right\|}=\frac{\|\boldsymbol{v}\|}{\left\|\boldsymbol{d}_{C}\right\|}=|\omega|" title="\frac{\left\|\boldsymbol{v}_{i}\right\|}{\left\|\boldsymbol{d}_{i}\right\|}=\frac{\|\boldsymbol{v}\|}{\left\|\boldsymbol{d}_{C}\right\|}=|\omega|" />

더 자세한 형태는 벡터로 다음과 같이 나타낼 수 있다.

<img src="https://latex.codecogs.com/gif.latex?\frac{v_{i&space;x}}{-d_{i&space;y}}=\frac{v_{x}}{-d_{C&space;y}}=\frac{v_{i&space;y}}{d_{i&space;x}}=\frac{v_{y}}{d_{C&space;x}}=\omega" title="\frac{v_{i x}}{-d_{i y}}=\frac{v_{x}}{-d_{C y}}=\frac{v_{i y}}{d_{i x}}=\frac{v_{y}}{d_{C x}}=\omega" />

위와는 반대로 로봇의 무게중심(로봇 좌표계)을 기준으로 ICR을 나타내면 다음과 같다.

<img src="https://latex.codecogs.com/gif.latex?\mathrm{ICR}=\left(x_{\mathrm{ICR}},&space;y_{\mathrm{ICR}}\right)=\left(-d_{C&space;x},-d_{C&space;y}\right)" title="\mathrm{ICR}=\left(x_{\mathrm{ICR}}, y_{\mathrm{ICR}}\right)=\left(-d_{C x},-d_{C y}\right)" />

위의 두 식을 이용해서 다음과 같이 나타낼 수 있다. 

<img src="https://latex.codecogs.com/gif.latex?\frac{v_{x}}{y_{\mathrm{ICR}}}=-\frac{v_{y}}{x_{\mathrm{ICR}}}=\omega" title="\frac{v_{x}}{y_{\mathrm{ICR}}}=-\frac{v_{y}}{x_{\mathrm{ICR}}}=\omega" />

회전중심을 기준으로하는 바퀴거리벡터와 무게중심거리벡터는 a, b, c를 이용해 다음과 같이 나타낼 수 있다. (a, b, c는 로봇 좌표계를 기준으로 +의 방향의 벡터이다.)

<img src="https://latex.codecogs.com/gif.latex?\begin{array}{l}&space;d_{1&space;y}=d_{2&space;y}=d_{C&space;y}&plus;c&space;\\&space;d_{3&space;y}=d_{4&space;y}=d_{C&space;y}-c&space;\\&space;d_{1&space;x}=d_{4&space;x}=d_{C&space;x}-c&space;\\&space;d_{2&space;x}=d_{3&space;x}=d_{C&space;x}&plus;b&space;\end{array}" title="\begin{array}{l} d_{1 y}=d_{2 y}=d_{C y}+c \\ d_{3 y}=d_{4 y}=d_{C y}-c \\ d_{1 x}=d_{4 x}=d_{C x}-c \\ d_{2 x}=d_{3 x}=d_{C x}+b \end{array}" />

<img src="https://latex.codecogs.com/gif.latex?\frac{v_{i&space;x}}{-d_{i&space;y}}=\frac{v_{x}}{-d_{C&space;y}}=\frac{v_{i&space;y}}{d_{i&space;x}}=\frac{v_{y}}{d_{C&space;x}}=\omega" title="\frac{v_{i x}}{-d_{i y}}=\frac{v_{x}}{-d_{C y}}=\frac{v_{i y}}{d_{i x}}=\frac{v_{y}}{d_{C x}}=\omega" />식과 위의 식을 결합하면 다음과 같은 식을 얻을 수 있다. (로봇 좌표계에서 x방향의 거리벡터는 (1, 2), (3, 4)바퀴가 같고, y방향의 거리벡터는 (1, 4), (2, 3)바퀴가 같기 때문이다.)

<img src="https://latex.codecogs.com/gif.latex?\begin{array}{l}&space;v_{L}=v_{1&space;x}=v_{2&space;x}&space;\\&space;v_{R}=v_{3&space;x}=v_{4&space;x}&space;\\&space;v_{F}=v_{2&space;y}=v_{3&space;y}&space;\\&space;v_{B}=v_{1&space;y}=v_{4&space;y_{y}}&space;\end{array}" title="\begin{array}{l} v_{L}=v_{1 x}=v_{2 x} \\ v_{R}=v_{3 x}=v_{4 x} \\ v_{F}=v_{2 y}=v_{3 y} \\ v_{B}=v_{1 y}=v_{4 y_{y}} \end{array}" />

위의 식을 모두 종합하면 바퀴의 속도는 로봇의 속도의 관계는 다음 식으로 정리된다.

<img src="https://latex.codecogs.com/gif.latex?\left[\begin{array}{c}&space;v_{L}&space;\\&space;v_{R}&space;\\&space;v_{F}&space;\\&space;v_{B}&space;\end{array}\right]=\left[\begin{array}{cc}&space;1&space;&&space;-c&space;\\&space;1&space;&&space;c&space;\\&space;0&space;&&space;-x_{\mathrm{ICR}}&plus;b&space;\\&space;0&space;&&space;-x_{\mathrm{ICR}}-a&space;\end{array}\right]\left[\begin{array}{c}&space;v_{x}&space;\\&space;\omega&space;\end{array}\right]" title="\left[\begin{array}{c} v_{L} \\ v_{R} \\ v_{F} \\ v_{B} \end{array}\right]=\left[\begin{array}{cc} 1 & -c \\ 1 & c \\ 0 & -x_{\mathrm{ICR}}+b \\ 0 & -x_{\mathrm{ICR}}-a \end{array}\right]\left[\begin{array}{c} v_{x} \\ \omega \end{array}\right]" />

그리고 <img src="https://latex.codecogs.com/gif.latex?\begin{array}{l}&space;v_{L}=v_{1&space;x}=v_{2&space;x}&space;\\&space;v_{R}=v_{3&space;x}=v_{4&space;x}&space;\end{array}" title="\begin{array}{l} v_{L}=v_{1 x}=v_{2 x} \\ v_{R}=v_{3 x}=v_{4 x} \end{array}" /> 이므로 모든 바퀴의 반지름이 같다는 가정하에 왼쪽바퀴와 오른쪽바퀴의 각속도를 묶어서 전체 바퀴의 각속도를 다음과 같이 쓸 수 있다.

<img src="https://latex.codecogs.com/gif.latex?\omega_{w}=\left[\begin{array}{c}&space;\omega_{L}&space;\\&space;\omega_{R}&space;\end{array}\right]=\frac{1}{r}\left[\begin{array}{l}&space;v_{L}&space;\\&space;v_{R}&space;\end{array}\right]" title="\omega_{w}=\left[\begin{array}{c} \omega_{L} \\ \omega_{R} \end{array}\right]=\frac{1}{r}\left[\begin{array}{l} v_{L} \\ v_{R} \end{array}\right]" />

위의 두 식을 묶으면 로봇 전체(COM)의 속도에 대한 식은 각 바퀴의 각속도로 다음과 같이 표현 가능하다.

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\eta}=\left[\begin{array}{c}&space;v_{x}&space;\\&space;\omega&space;\end{array}\right]=r\left[\begin{array}{c}&space;\frac{\omega_{L}&plus;\omega_{R}}{2}&space;\\&space;\frac{-\omega_{L}&plus;\omega_{R}}{2&space;c}&space;\end{array}\right]" title="\boldsymbol{\eta}=\left[\begin{array}{c} v_{x} \\ \omega \end{array}\right]=r\left[\begin{array}{c} \frac{\omega_{L}+\omega_{R}}{2} \\ \frac{-\omega_{L}+\omega_{R}}{2 c} \end{array}\right]" />

(Caracciolo L., De Luca A. and Iannitti S. (1999): Trajectory tracking control of a four-wheel differentially driven mobile robot) 논문에 따르면 SSMR 로봇에서는 다음과 같은 제약이 들어간다. (왜 이런 제약이 필요한지는 이해가 되지는 않음 )

<img src="https://latex.codecogs.com/gif.latex?v_{y}&plus;x_{\mathrm{ICR}}&space;\dot{\theta}=0" title="v_{y}+x_{\mathrm{ICR}} \dot{\theta}=0" />

로봇 좌표계와 world 좌표계의 변환을 통해 <img src="https://latex.codecogs.com/gif.latex?v_{y}=-\sin&space;\theta&space;\cdot&space;\dot{X}&plus;\cos&space;\theta&space;\cdot&space;\dot{Y}" title="v_{y}=-\sin \theta \cdot \dot{X}+\cos \theta \cdot \dot{Y}" />를 얻을 수 있고, 위의 식과의 결합을 통해 제약을 다시 다음과 같이 표현할 수 있다.

<img src="https://latex.codecogs.com/gif.latex?\left[-\sin&space;\theta&space;\quad&space;\cos&space;\theta&space;\quad&space;x_{\mathrm{ICR}}\right][\dot{X}&space;\quad&space;\dot{Y}&space;\quad&space;\dot{\theta}]^{T}=A(\boldsymbol{q})&space;\dot{\boldsymbol{q}}=\mathbf{0}" title="\left[-\sin \theta \quad \cos \theta \quad x_{\mathrm{ICR}}\right][\dot{X} \quad \dot{Y} \quad \dot{\theta}]^{T}=A(\boldsymbol{q}) \dot{\boldsymbol{q}}=\mathbf{0}" />

그리고 world 좌표계에서의 속도를 각 바퀴와 COM의 운동에 대해 변환하면 다음과 같이 나타낼 수 있다.

<img src="https://latex.codecogs.com/gif.latex?\dot{\boldsymbol{q}}=\boldsymbol{S}(\boldsymbol{q})&space;\boldsymbol{\eta}" title="\dot{\boldsymbol{q}}=\boldsymbol{S}(\boldsymbol{q}) \boldsymbol{\eta}" />

where 

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{S}^{T}(\boldsymbol{q})&space;\boldsymbol{A}^{T}(\boldsymbol{q})=\mathbf{0}" title="\boldsymbol{S}^{T}(\boldsymbol{q}) \boldsymbol{A}^{T}(\boldsymbol{q})=\mathbf{0}" />

and

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{S}(\boldsymbol{q})=\left[\begin{array}{cc}&space;\cos&space;\theta&space;&&space;x_{\mathrm{ICR}}&space;\sin&space;\theta&space;\\&space;\sin&space;\theta&space;&&space;-x_{\mathrm{ICR}}&space;\cos&space;\theta&space;\\&space;0&space;&&space;1&space;\end{array}\right]" title="\boldsymbol{S}(\boldsymbol{q})=\left[\begin{array}{cc} \cos \theta & x_{\mathrm{ICR}} \sin \theta \\ \sin \theta & -x_{\mathrm{ICR}} \cos \theta \\ 0 & 1 \end{array}\right]" />

주목해야할점은 로봇의 운동은 2차원(<img src="https://latex.codecogs.com/gif.latex?v_{x}$,&space;$\omega" title="v_{x}$, $\omega" />) 으로 표현되고, world 좌표계에서의 움직임은 3차원(<img src="https://latex.codecogs.com/gif.latex?\dot{X}&space;\quad&space;\dot{Y}&space;\quad&space;\dot{\theta}" title="\dot{X} \quad \dot{Y} \quad \dot{\theta}" />)으로 표현된다는 것이다. 또 흥미로운점은 SSMR에 대해 물리적으로 기술한 식들이 two-wheel mobile robot과 상당히 유사하다는 점이다.

### 바퀴에서의 마찰력을 고려한 modeling
![wheel dynamic model](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-06-19-Modeling_of_4Wheel_Skid_Steering_Mobile_Robot/capture5.jpg?raw=true)

바퀴에서의 마찰력이 있는 경우를 고려해보자. 마찰력은 longitudinal 방향에서의 <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{F}_{s&space;i}" title="\boldsymbol{F}_{s i}" /> 그리고 lateral 방향에서의 <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{F}_{l&space;i}" title="\boldsymbol{F}_{l i}" />가 있다. 그리고 바퀴에서 앞으로 가려는 힘은 토크와 바퀴의 반지름으로 <img src="https://latex.codecogs.com/gif.latex?F_{i}=\frac{\tau_{i}}{r}" title="F_{i}=\frac{\tau_{i}}{r}" /> 처럼 표현할 수 있다. 

![robot](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-06-19-Modeling_of_4Wheel_Skid_Steering_Mobile_Robot/capture6.jpg?raw=true)

각 바퀴에 가해지는 수직항력은 COM과 바퀴사이의 거리 a, b를 통해 다음과 같이 나타낼 수 있다.

<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;N_{1}&space;a&space;&=N_{2}&space;b&space;\\&space;N_{4}&space;a&space;&=N_{3}&space;b&space;\\&space;\sum_{i=1}^{4}&space;N_{i}&space;&=m&space;g&space;\end{aligned}" title="\begin{aligned} N_{1} a &=N_{2} b \\ N_{4} a &=N_{3} b \\ \sum_{i=1}^{4} N_{i} &=m g \end{aligned}" />

(1, 4) 바퀴와 COM사이 거리가 같고, (2, 3) 바퀴와 COM사이 거리가 같다. 

<img src="https://latex.codecogs.com/gif.latex?\begin{array}{l}&space;N_{1}=N_{4}=\frac{b}{2(a&plus;b)}&space;m&space;g&space;\\&space;N_{2}=N_{3}=\frac{a}{2(a&plus;b)}&space;m&space;g&space;\end{array}" title="\begin{array}{l} N_{1}=N_{4}=\frac{b}{2(a+b)} m g \\ N_{2}=N_{3}=\frac{a}{2(a+b)} m g \end{array}" />

마찰력을 modeling하는 것은 상당히 복잡하다. 여기서는 간단하게 Coulumb 마찰과 Viscous 마찰로만 표현을 한다.

<img src="https://latex.codecogs.com/gif.latex?F_{f}(\sigma)=\mu_{c}&space;N&space;\operatorname{sgn}(\sigma)&plus;\mu_{v}&space;\sigma" title="F_{f}(\sigma)=\mu_{c} N \operatorname{sgn}(\sigma)+\mu_{v} \sigma" />

여기서 <img src="https://latex.codecogs.com/gif.latex?\sigma" title="\sigma" />는 linear 속도, N은 수직항력, <img src="https://latex.codecogs.com/gif.latex?\mu_{c}" title="\mu_{c}" />은 coulomb 마찰계수, <img src="https://latex.codecogs.com/gif.latex?\mu_{v}" title="\mu_{v}" />은 viscous 마찰계수이다. lateral 방향에서 미끄러짐이 일어날 때 <img src="https://latex.codecogs.com/gif.latex?\sigma" title="\sigma" />는 매우 작다. 따라서 <img src="https://latex.codecogs.com/gif.latex?\mu_{c}&space;N&space;\gg\left|\mu_{v}&space;\sigma\right|" title="\mu_{c} N \gg\left|\mu_{v} \sigma\right|" /> 근사가 가능하다. 따라서 viscous 마찰은 무시가 가능하다. 

<img src="https://latex.codecogs.com/gif.latex?\operatorname{sgn}(\sigma)" title="\operatorname{sgn}(\sigma)" />항 때문에 <img src="https://latex.codecogs.com/gif.latex?\sigma" title="\sigma" />가 0일 때는 마찰력이 smooth하지 않다.(미분불가능하기 때문) 따라서 아래와 같은 근사식을 이용한다.
 <img src="https://latex.codecogs.com/gif.latex?\widehat{\operatorname{sgn}}(\sigma)=\frac{2}{\pi}&space;\arctan&space;\left(k_{s}&space;\sigma\right)" title="\widehat{\operatorname{sgn}}(\sigma)=\frac{2}{\pi} \arctan \left(k_{s} \sigma\right)" />

<img src="https://latex.codecogs.com/gif.latex?k_{s}&space;\gg&space;1" title="k_{s} \gg 1" />에서는 다음과 같은 근사가 가능하다.

<img src="https://latex.codecogs.com/gif.latex?\lim&space;_{k_{s}&space;\rightarrow&space;\infty}&space;\frac{2}{\pi}&space;\arctan&space;\left(k_{s}&space;\sigma\right)=\operatorname{sgn}(x)" title="\lim _{k_{s} \rightarrow \infty} \frac{2}{\pi} \arctan \left(k_{s} \sigma\right)=\operatorname{sgn}(x)" />

lateral 방향, longitudinal 방향의 마찰력을 다음과 같이 나타낼 수 있다.

<img src="https://latex.codecogs.com/gif.latex?\begin{array}{l}&space;F_{l&space;i}=\mu_{l&space;c&space;i}&space;m&space;g&space;\widehat{\operatorname{sgn}}\left(v_{y&space;i}\right)&space;\\&space;F_{s&space;i}=\mu_{s&space;c&space;i}&space;m&space;g&space;\widehat{\operatorname{sgn}}\left(v_{x&space;i}\right)&space;\end{array}" title="\begin{array}{l} F_{l i}=\mu_{l c i} m g \widehat{\operatorname{sgn}}\left(v_{y i}\right) \\ F_{s i}=\mu_{s c i} m g \widehat{\operatorname{sgn}}\left(v_{x i}\right) \end{array}" />

### 에너지 보존을 이용한 운동 modeling
로봇의 potential 에너지는 0으로 가정한다. ( <img src="https://latex.codecogs.com/gif.latex?P&space;E(\boldsymbol{q})=0" title="P E(\boldsymbol{q})=0" /> ) 그러면 Lagrange-Euler 식([Lagrange-Euler 참고](https://blog.naver.com/at3650/220597986325)), 그리고 제약조건을 붙이기 위해 Lagrange multiplier을 이용한다. ([Lagrange multiplier 참고](https://blog.naver.com/gobyoungmin/221310459939))Lagrangian L은 다음과 같이 나타낼 수 있다.

<img src="https://latex.codecogs.com/gif.latex?L(\boldsymbol{q},&space;\dot{\boldsymbol{q}})=T(\boldsymbol{q},&space;\dot{\boldsymbol{q}})" title="L(\boldsymbol{q}, \dot{\boldsymbol{q}})=T(\boldsymbol{q}, \dot{\boldsymbol{q}})" />

바퀴의 회전 에너지를 무시한다고 가정하고, 로봇의 운동에너지는 로봇 좌표계에서 다음과 같다.

<img src="https://latex.codecogs.com/gif.latex?T=\frac{1}{2}&space;m&space;\boldsymbol{v}^{T}&space;\boldsymbol{v}&plus;\frac{1}{2}&space;I&space;\omega^{2}" title="T=\frac{1}{2} m \boldsymbol{v}^{T} \boldsymbol{v}+\frac{1}{2} I \omega^{2}" />
<img src="https://latex.codecogs.com/gif.latex?I" title="I" />는 로봇 COM의 회전모멘트(moi)이다. 

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{v}^{T}&space;\boldsymbol{v}=v_{x}^{2}&plus;v_{u}^{2}=\dot{X}^{2}&plus;\dot{Y}^{2}" title="\boldsymbol{v}^{T} \boldsymbol{v}=v_{x}^{2}+v_{u}^{2}=\dot{X}^{2}+\dot{Y}^{2}" />이기 때문에 로봇의 운동에너지는 world 좌표계에서는 다음과 같다.

<img src="https://latex.codecogs.com/gif.latex?T=\frac{1}{2}&space;m\left(\dot{X}^{2}&plus;\dot{Y}^{2}\right)&plus;\frac{1}{2}&space;I&space;\dot{\theta}^{2}" title="T=\frac{1}{2} m\left(\dot{X}^{2}+\dot{Y}^{2}\right)+\frac{1}{2} I \dot{\theta}^{2}" />

운동에너지의 시간에 대한 미분항은 [<img src="https://latex.codecogs.com/gif.latex?X&space;\quad&space;Y&space;\quad&space;\theta" title="X \quad Y \quad \theta" />] 각 성분에 대해 다음과 같다.

<img src="https://latex.codecogs.com/gif.latex?\frac{\mathrm{d}}{\mathrm{d}&space;t}\left(\frac{\partial&space;E_{k}}{\partial&space;\dot{\boldsymbol{q}}}\right)=\left[\begin{array}{c}&space;m&space;\ddot{X}&space;\\&space;m&space;\ddot{Y}&space;\\&space;I&space;\ddot{\theta}&space;\end{array}\right]=M&space;\ddot{q}" title="\frac{\mathrm{d}}{\mathrm{d} t}\left(\frac{\partial E_{k}}{\partial \dot{\boldsymbol{q}}}\right)=\left[\begin{array}{c} m \ddot{X} \\ m \ddot{Y} \\ I \ddot{\theta} \end{array}\right]=M \ddot{q}" />

where

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{M}=\left[\begin{array}{ccc}&space;m&space;&&space;0&space;&&space;0&space;\\&space;0&space;&&space;m&space;&&space;0&space;\\&space;0&space;&&space;0&space;&&space;I&space;\end{array}\right]" title="\boldsymbol{M}=\left[\begin{array}{ccc} m & 0 & 0 \\ 0 & m & 0 \\ 0 & 0 & I \end{array}\right]" />

world 좌표계에서의 X, Y 성분에 대한 마찰은 4개의 바퀴의 마찰력의 합으로 다음과 같이 쓸 수 있다.
<img src="https://latex.codecogs.com/gif.latex?F_{r&space;x}(\dot{\boldsymbol{q}})=\cos&space;\theta&space;\sum_{i=1}^{4}&space;F_{s&space;i}\left(v_{x&space;i}\right)-\sin&space;\theta&space;\sum_{i=1}^{4}&space;F_{l&space;i}\left(v_{y&space;i}\right)" title="F_{r x}(\dot{\boldsymbol{q}})=\cos \theta \sum_{i=1}^{4} F_{s i}\left(v_{x i}\right)-\sin \theta \sum_{i=1}^{4} F_{l i}\left(v_{y i}\right)" />

<img src="https://latex.codecogs.com/gif.latex?F_{r&space;y}(\dot{\boldsymbol{q}})=\sin&space;\theta&space;\sum_{i=1}^{4}&space;F_{s&space;i}\left(v_{x&space;i}\right)&plus;\cos&space;\theta&space;\sum_{i=1}^{4}&space;F_{l&space;i}\left(v_{y&space;i}\right)" title="F_{r y}(\dot{\boldsymbol{q}})=\sin \theta \sum_{i=1}^{4} F_{s i}\left(v_{x i}\right)+\cos \theta \sum_{i=1}^{4} F_{l i}\left(v_{y i}\right)" />

그리고 COM에 대한 회전 저항력 <img src="https://latex.codecogs.com/gif.latex?M_{r}" title="M_{r}" />은 다음과 같다.

<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;M_{r}(\dot{\boldsymbol{q}})=&-a&space;\sum_{i=1,4}&space;F_{l&space;i}\left(v_{y&space;i}\right)&plus;b&space;\sum_{i=2,3}&space;F_{l&space;i}\left(v_{y&space;i}\right)&space;\\&space;&&plus;c\left[-\sum_{i=1,2}&space;F_{s&space;i}\left(v_{x&space;i}\right)&plus;\sum_{i=3,4}&space;F_{s&space;i}\left(v_{x&space;i}\right)\right]&space;\end{aligned}" title="\begin{aligned} M_{r}(\dot{\boldsymbol{q}})=&-a \sum_{i=1,4} F_{l i}\left(v_{y i}\right)+b \sum_{i=2,3} F_{l i}\left(v_{y i}\right) \\ &+c\left[-\sum_{i=1,2} F_{s i}\left(v_{x i}\right)+\sum_{i=3,4} F_{s i}\left(v_{x i}\right)\right] \end{aligned}" />

따라서 전체적인 resistive forces는  [<img src="https://latex.codecogs.com/gif.latex?X&space;\quad&space;Y&space;\quad&space;\theta" title="X \quad Y \quad \theta" />] 에 따라 다음과 같다.

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{R}(\dot{\boldsymbol{q}})=\left[\begin{array}{lll}&space;F_{r&space;x}(\dot{\boldsymbol{q}})&space;&&space;F_{r&space;y}(\dot{\boldsymbol{q}})&space;&&space;M_{r}(\dot{\boldsymbol{q}})&space;\end{array}\right]^{T}" title="\boldsymbol{R}(\dot{\boldsymbol{q}})=\left[\begin{array}{lll} F_{r x}(\dot{\boldsymbol{q}}) & F_{r y}(\dot{\boldsymbol{q}}) & M_{r}(\dot{\boldsymbol{q}}) \end{array}\right]^{T}" />

전체 로봇에 힘은 input으로 넣어주는 각 바퀴의 힘의 합이다. world 좌표계에서 로봇에 input으로 넣어주는 힘은 다음과 같다.

<img src="https://latex.codecogs.com/gif.latex?F_{x}=\cos&space;\theta&space;\sum_{i=1}^{4}&space;F_{i}" title="F_{x}=\cos \theta \sum_{i=1}^{4} F_{i}" />

<img src="https://latex.codecogs.com/gif.latex?F_{y}=\sin&space;\theta&space;\sum_{i=1}^{4}&space;F_{i}" title="F_{y}=\sin \theta \sum_{i=1}^{4} F_{i}" />

그리고 COM에 작용하는 input으로 들어가는 torque는 다음과 같다. 

<img src="https://latex.codecogs.com/gif.latex?M=c\left(-F_{1}-F_{2}&plus;F_{3}&plus;F_{4}\right)" title="M=c\left(-F_{1}-F_{2}+F_{3}+F_{4}\right)" />

따라서 전체 로봇에 input으로 들어가는 힘은  [<img src="https://latex.codecogs.com/gif.latex?X&space;\quad&space;Y&space;\quad&space;\theta" title="X \quad Y \quad \theta" />] 에 대해 다음과 같다.

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{F}=\left[\begin{array}{lll}&space;F_{x}&space;&&space;F_{y}&space;&&space;M&space;\end{array}\right]^{T}" title="\boldsymbol{F}=\left[\begin{array}{lll} F_{x} & F_{y} & M \end{array}\right]^{T}" />

<img src="https://latex.codecogs.com/gif.latex?=\frac{1}{r}\left[\begin{array}{c}&space;\cos&space;\theta&space;\sum_{i=1}^{4}&space;\tau_{i}&space;\\&space;\sin&space;\theta&space;\sum_{i=1}^{4}&space;\tau_{i}&space;\\&space;c\left(-\tau_{1}-\tau_{2}&plus;\tau_{3}&plus;\tau_{4}\right)&space;\end{array}\right]" title="=\frac{1}{r}\left[\begin{array}{c} \cos \theta \sum_{i=1}^{4} \tau_{i} \\ \sin \theta \sum_{i=1}^{4} \tau_{i} \\ c\left(-\tau_{1}-\tau_{2}+\tau_{3}+\tau_{4}\right) \end{array}\right]" />

바퀴의 왼쪽 바퀴 쌍의 토크, 오른쪽 바퀴 쌍의 토크는 다음 식과 같다.

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{\tau}=\left[\begin{array}{c}&space;\tau_{L}&space;\\&space;\tau_{R}&space;\end{array}\right]=\left[\begin{array}{c}&space;\tau_{1}&plus;\tau_{2}&space;\\&space;\tau_{3}&plus;\tau_{4}&space;\end{array}\right]" title="\boldsymbol{\tau}=\left[\begin{array}{c} \tau_{L} \\ \tau_{R} \end{array}\right]=\left[\begin{array}{c} \tau_{1}+\tau_{2} \\ \tau_{3}+\tau_{4} \end{array}\right]" />

위의 두 식을 결합하여 로봇의 총 input으로 들어가는 힘은 다음과 같이 정리할 수 있다.

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{F}=\boldsymbol{B}(\boldsymbol{q})&space;\boldsymbol{\tau}" title="\boldsymbol{F}=\boldsymbol{B}(\boldsymbol{q}) \boldsymbol{\tau}" />

where

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{B}(\boldsymbol{q})=\frac{1}{r}\left[\begin{array}{cc}&space;\cos&space;\theta&space;&&space;\cos&space;\theta&space;\\&space;\sin&space;\theta&space;&&space;\sin&space;\theta&space;\\&space;-c&space;&&space;c&space;\end{array}\right]" title="\boldsymbol{B}(\boldsymbol{q})=\frac{1}{r}\left[\begin{array}{cc} \cos \theta & \cos \theta \\ \sin \theta & \sin \theta \\ -c & c \end{array}\right]" />

**실제 로봇의 운동에너지 + 마찰력으로 손실되는 에너지 = input으로 들어가는 에너지**로 다음과 같이 표현할 수 있다.

<img src="https://latex.codecogs.com/gif.latex?M(\boldsymbol{q})&space;\ddot{\boldsymbol{q}}&plus;\boldsymbol{R}(\dot{\boldsymbol{q}})=\boldsymbol{B}(\boldsymbol{q})&space;\tau" title="M(\boldsymbol{q}) \ddot{\boldsymbol{q}}+\boldsymbol{R}(\dot{\boldsymbol{q}})=\boldsymbol{B}(\boldsymbol{q}) \tau" />

그리고 앞에서 기술한대로 SSMR에는 제약이 들어가 식은 다시 아래와 같이 정리된다.(Lagrange multiplier를 참고하세요)

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{M}(\boldsymbol{q})&space;\ddot{\boldsymbol{q}}&plus;\boldsymbol{R}(\dot{\boldsymbol{q}})=\boldsymbol{B}(\boldsymbol{q})&space;\boldsymbol{\tau}&plus;\boldsymbol{A}^{T}(\boldsymbol{q})&space;\boldsymbol{\lambda}" title="\boldsymbol{M}(\boldsymbol{q}) \ddot{\boldsymbol{q}}+\boldsymbol{R}(\dot{\boldsymbol{q}})=\boldsymbol{B}(\boldsymbol{q}) \boldsymbol{\tau}+\boldsymbol{A}^{T}(\boldsymbol{q}) \boldsymbol{\lambda}" />

정리를 위해 위의 식에 양변에 <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{S}^{T}(\boldsymbol{q})" title="\boldsymbol{S}^{T}(\boldsymbol{q})" />를 곱해주면 다음과 같다.

<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{S}^{T}(\boldsymbol{q})&space;\boldsymbol{M}(\boldsymbol{q})&space;\ddot{\boldsymbol{q}}&plus;\boldsymbol{S}^{T}(\boldsymbol{q})&space;\mathbf{R}(\dot{\mathbf{q}})" title="\boldsymbol{S}^{T}(\boldsymbol{q}) \boldsymbol{M}(\boldsymbol{q}) \ddot{\boldsymbol{q}}+\boldsymbol{S}^{T}(\boldsymbol{q}) \mathbf{R}(\dot{\mathbf{q}})" />

<img src="https://latex.codecogs.com/gif.latex?=\boldsymbol{S}(\boldsymbol{q})^{T}&space;\boldsymbol{B}(\boldsymbol{q})&space;\boldsymbol{\tau}&plus;\boldsymbol{S}^{T}(\boldsymbol{q})&space;\boldsymbol{A}^{T}(\boldsymbol{q})&space;\boldsymbol{\lambda}" title="=\boldsymbol{S}(\boldsymbol{q})^{T} \boldsymbol{B}(\boldsymbol{q}) \boldsymbol{\tau}+\boldsymbol{S}^{T}(\boldsymbol{q}) \boldsymbol{A}^{T}(\boldsymbol{q}) \boldsymbol{\lambda}" />

또한 위에서 구한 <img src="https://latex.codecogs.com/gif.latex?\dot{\boldsymbol{q}}=\boldsymbol{S}(\boldsymbol{q})&space;\boldsymbol{\eta}" title="\dot{\boldsymbol{q}}=\boldsymbol{S}(\boldsymbol{q}) \boldsymbol{\eta}" />식을 미분하면 다음과 같다.

<img src="https://latex.codecogs.com/gif.latex?\ddot{\boldsymbol{q}}=\dot{\boldsymbol{S}}(\boldsymbol{q})&space;\boldsymbol{\eta}&plus;\boldsymbol{S}(\boldsymbol{q})&space;\dot{\boldsymbol{\eta}}" title="\ddot{\boldsymbol{q}}=\dot{\boldsymbol{S}}(\boldsymbol{q}) \boldsymbol{\eta}+\boldsymbol{S}(\boldsymbol{q}) \dot{\boldsymbol{\eta}}" />

**위의 식을 통해 전체적으로 정리하면 다음과 같다.**


<img src="https://latex.codecogs.com/gif.latex?\bar{M}&space;\dot{\eta}&plus;\bar{C}&space;\eta&plus;\bar{R}=\bar{B}&space;\tau" title="\bar{M} \dot{\eta}+\bar{C} \eta+\bar{R}=\bar{B} \tau" />

where

<img src="https://latex.codecogs.com/gif.latex?\overline{\boldsymbol{C}}=\boldsymbol{S}^{T}&space;\boldsymbol{M}&space;\dot{\boldsymbol{S}}=m&space;x_{\mathrm{ICR}}\left[\begin{array}{cc}&space;0&space;&&space;\dot{\theta}&space;\\&space;-\dot{\theta}&space;&&space;\dot{x}_{\mathrm{ICR}}&space;\end{array}\right]" title="\overline{\boldsymbol{C}}=\boldsymbol{S}^{T} \boldsymbol{M} \dot{\boldsymbol{S}}=m x_{\mathrm{ICR}}\left[\begin{array}{cc} 0 & \dot{\theta} \\ -\dot{\theta} & \dot{x}_{\mathrm{ICR}} \end{array}\right]" />

<img src="https://latex.codecogs.com/gif.latex?\overline{\boldsymbol{M}}=\boldsymbol{S}^{T}&space;\boldsymbol{M}&space;\boldsymbol{S}=\left[\begin{array}{cc}&space;m&space;&&space;0&space;\\&space;0&space;&&space;m&space;x_{\mathrm{ICR}}^{2}&plus;I&space;\end{array}\right]" title="\overline{\boldsymbol{M}}=\boldsymbol{S}^{T} \boldsymbol{M} \boldsymbol{S}=\left[\begin{array}{cc} m & 0 \\ 0 & m x_{\mathrm{ICR}}^{2}+I \end{array}\right]" />

<img src="https://latex.codecogs.com/gif.latex?\overline{\boldsymbol{R}}=\boldsymbol{S}^{T}&space;\boldsymbol{R}=\left[\begin{array}{c}&space;F_{r&space;x}(\dot{\boldsymbol{q}})&space;\\&space;x_{\mathrm{ICR}}&space;F_{r&space;y}(\dot{\boldsymbol{q}})&plus;M_{r}&space;\end{array}\right]" title="\overline{\boldsymbol{R}}=\boldsymbol{S}^{T} \boldsymbol{R}=\left[\begin{array}{c} F_{r x}(\dot{\boldsymbol{q}}) \\ x_{\mathrm{ICR}} F_{r y}(\dot{\boldsymbol{q}})+M_{r} \end{array}\right]" />

<img src="https://latex.codecogs.com/gif.latex?\overline{\boldsymbol{B}}=\boldsymbol{S}^{T}&space;\boldsymbol{B}=\frac{1}{r}\left[\begin{array}{cc}&space;1&space;&&space;1&space;\\&space;-c&space;&&space;c&space;\end{array}\right]" title="\overline{\boldsymbol{B}}=\boldsymbol{S}^{T} \boldsymbol{B}=\frac{1}{r}\left[\begin{array}{cc} 1 & 1 \\ -c & c \end{array}\right]" />


**각 바퀴에 input으로 넣어주어야하는 torque를 구하기 위한 식을 풀기 위해서 알아야하는 변수들은 아래 표로 정리해두었다.**
|    |    |    |    |    |
|:---------:|:---------:|:---------:|:---------:|:---------:|
| <img src="https://latex.codecogs.com/gif.latex?m" title="m" /> | <img src="https://latex.codecogs.com/gif.latex?x_{\mathrm{ICR}}" title="x_{\mathrm{ICR}}" /> | <img src="https://latex.codecogs.com/gif.latex?I" title="I" /> | <img src="https://latex.codecogs.com/gif.latex?\dot{v}_{x}" title="\dot{v}_{x}" /> | <img src="https://latex.codecogs.com/gif.latex?\dot{\theta}" title="\dot{\theta}" /> |
| <img src="https://latex.codecogs.com/gif.latex?\dot{x}_{\mathrm{ICR}}" title="\dot{x}_{\mathrm{ICR}}" /> | <img src="https://latex.codecogs.com/gif.latex?w_{L}" title="w_{L}" /> | <img src="https://latex.codecogs.com/gif.latex?w_{R}" title="w_{R}" /> | <img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" /> | <img src="https://latex.codecogs.com/gif.latex?\mu_{l&space;c&space;i}" title="\mu_{l c i}" /> |
| <img src="https://latex.codecogs.com/gif.latex?\mu_{s&space;c&space;i}" title="\mu_{s c i}" /> | <img src="https://latex.codecogs.com/gif.latex?k_{s}" title="k_{s}" />|<img src="https://latex.codecogs.com/gif.latex?v_{x&space;i}" title="v_{x i}" />  |<img src="https://latex.codecogs.com/gif.latex?v_{y&space;i}" title="v_{y i}" />  | <img src="https://latex.codecogs.com/gif.latex?a" title="a" /> |
| <img src="https://latex.codecogs.com/gif.latex?b" title="b" /> | <img src="https://latex.codecogs.com/gif.latex?c" title="c" /> |  |  | 
