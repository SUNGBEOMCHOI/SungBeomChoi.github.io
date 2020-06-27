---
layout: post
title: Modeling of 4Wheel Skid Steering Mobile Robot
featured-img: 2020-06-19-Modeling_of_4Wheel_Skid_Steering_Mobile_Robot/capture2
---

#### 이글은 4wheel skid steering mobile robot의 물리적 모델링에 관한 포스트이다.

4wheel skid steering mobile robot(SSMR)은 robust한 환경에서도 잘 견딜 수 있는 성질덕분에 우주 탐사 등과 같은 환경에서 쓰이는 형태의 로봇이다. 그러나 제어에 있어서는 challenging task이다. 

### Robot Coordinate and World Coordinate
로봇의 운동을 규정하기 위해서는 좌표계를 생각해볼 필요가 있다. 좌표계는 robot coordinate와 world coordinate로 나뉘게 된다.

SSMR에서는 z축 방향이나 roll, pitch의 운동은 생각할 필요가 없다. 따라서 로봇의 속도에 대해 robot coordinate에서는 linear velocity는 $\left[\begin{array}{lll}v_{x} & v_{y} & 0\end{array}\right]^{T}$ 이고, angular velocity는 $\boldsymbol{\omega}=\left[\begin{array}{ccc}0 & 0 & \omega\end{array}\right]^{T}$로 쓸 수 있다. 

world coordinate상에서 상태는 X, Y 위치와 X축과의 각도 $\theta$로 표현할 수 있다.  따라서 world coordinate상에서 로봇의 상태는 $\boldsymbol{q}=\left[\begin{array}{lll}X & Y & \theta\end{array}\right]^{T}$로 표현하고 속도는 $\dot{\boldsymbol{q}}=\left[\begin{array}{lll}\dot{X} & \dot{Y} & \dot{\theta}\end{array}\right]^{T}$이다. 

world coordinate와 robot coordinate간의 속도변환은 회전변환행렬을 사용하면 아래와 같이 나타낼 수 있다. 
$$\left[\begin{array}{c}
\dot{X} \\
\dot{Y}
\end{array}\right]=\left[\begin{array}{cc}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{array}\right]\left[\begin{array}{c}
v_{x} \\
v_{y}
\end{array}\right]$$

로봇 좌표계에서 yaw축에 대한 회전은 world 좌표계에서 yaw축의 회전과 동일하다. 따라서  $\dot{\theta}=\omega$ 로 표현할 수 있다.

따라서 로봇좌표계와 world 좌표계의 속도에 대한 완전한 변환은 다음과 같이 쓸 수 있다. 
$$\left[\begin{array}{c}
\dot{X} \\
\dot{Y} \\
\dot{\theta}
\end{array}\right]=\left[\begin{array}{ccc}
\cos \theta & -\sin \theta & 0 \\
\sin \theta & \cos \theta & 0 \\
0 & 0 & 1
\end{array}\right]\left[\begin{array}{c}
v_{x} \\
v_{y} \\
w
\end{array}\right]$$

![coordinate1](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-06-19-Modeling_of_4Wheel_Skid_Steering_Mobile_Robot/capture1.jpg?raw=true) 


![robot coordinate and world coordinate](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-06-19-Modeling_of_4Wheel_Skid_Steering_Mobile_Robot/capture2.jpg?raw=true)

### 바퀴의 운동 modeling
![wheel](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-06-19-Modeling_of_4Wheel_Skid_Steering_Mobile_Robot/capture3.jpg?raw=true)
i번째 바퀴의 회전 속도를 $\omega_{i}(t)$라 하자. 

여기서는 3가지 가정을 한다.
 - 여기서는 바퀴의 두께는 무시한다.
 - 지면과 한점 $P_{i}$에서 만난다.
 - longitudinal 방향으로의 미끄러짐은 없다

로봇이 정면 방향으로 움직이지 않는 이상 바퀴의 lateral 방향 속도는 0이 아니다. 바퀴는 longitudinal 방향으로는 완전히 구르고, lateral 방향으로는 미끄러지게 된다. 

longitudinal 방향으로의 속도는 다음과 같이 바퀴의 반지름과 바퀴의 각속도로 표현할 수 있다.
$$v_{i x}=r_{i} \omega_{i}$$

### 4바퀴의 운동이 합쳐져 만들어지는 로봇의 움직임
![robot and wheel](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-06-19-Modeling_of_4Wheel_Skid_Steering_Mobile_Robot/capture4.jpg?raw=true)
로봇의 회전의 중심(ICR)으로부터 각 i 바퀴와 로봇의 무게중심 COM으로 가는 거리벡터를 다음과 같이 쓸 수 있다.
$$\boldsymbol{d}_{i}=\left[\begin{array}{ll}
d_{i x} & d_{i y}
\end{array}\right]^{T} \text { and } \boldsymbol{d}_{C}=\left[\begin{array}{ll}
d_{C x} & d_{C y}
\end{array}\right]^{T}$$

ICR에 대한 i 번째 바퀴의 회전 속도와 COM의 회전속도는 같으므로 다음과 같이 쓸 수 있다. (스칼라 개념)
$$\frac{\left\|\boldsymbol{v}_{i}\right\|}{\left\|\boldsymbol{d}_{i}\right\|}=\frac{\|\boldsymbol{v}\|}{\left\|\boldsymbol{d}_{C}\right\|}=|\omega|$$

더 자세한 형태는 벡터로 다음과 같이 나타낼 수 있다.
$$\frac{v_{i x}}{-d_{i y}}=\frac{v_{x}}{-d_{C y}}=\frac{v_{i y}}{d_{i x}}=\frac{v_{y}}{d_{C x}}=\omega$$

위와는 반대로 로봇의 무게중심(로봇 좌표계)을 기준으로 ICR을 나타내면 다음과 같다.
$$\mathrm{ICR}=\left(x_{\mathrm{ICR}}, y_{\mathrm{ICR}}\right)=\left(-d_{C x},-d_{C y}\right)$$

위의 두 식을 이용해서 다음과 같이 나타낼 수 있다. 
$$\frac{v_{x}}{y_{\mathrm{ICR}}}=-\frac{v_{y}}{x_{\mathrm{ICR}}}=\omega$$

회전중심을 기준으로하는 바퀴거리벡터와 무게중심거리벡터는 a, b, c를 이용해 다음과 같이 나타낼 수 있다. (a, b, c는 로봇 좌표계를 기준으로 +의 방향의 벡터이다.)
$$\begin{array}{l}
d_{1 y}=d_{2 y}=d_{C y}+c \\
d_{3 y}=d_{4 y}=d_{C y}-c \\
d_{1 x}=d_{4 x}=d_{C x}-c \\
d_{2 x}=d_{3 x}=d_{C x}+b
\end{array}$$

$\frac{v_{i x}}{-d_{i y}}=\frac{v_{x}}{-d_{C y}}=\frac{v_{i y}}{d_{i x}}=\frac{v_{y}}{d_{C x}}=\omega$식과 위의 식을 결합하면 다음과 같은 식을 얻을 수 있다. (로봇 좌표계에서 x방향의 거리벡터는 (1, 2), (3, 4)바퀴가 같고, y방향의 거리벡터는 (1, 4), (2, 3)바퀴가 같기 때문이다.)
$$\begin{array}{l}
v_{L}=v_{1 x}=v_{2 x} \\
v_{R}=v_{3 x}=v_{4 x} \\
v_{F}=v_{2 y}=v_{3 y} \\
v_{B}=v_{1 y}=v_{4 y_{y}}
\end{array}$$

위의 식을 모두 종합하면 바퀴의 속도는 로봇의 속도의 관계는 다음 식으로 정리된다.
$$\left[\begin{array}{c}
v_{L} \\
v_{R} \\
v_{F} \\
v_{B}
\end{array}\right]=\left[\begin{array}{cc}
1 & -c \\
1 & c \\
0 & -x_{\mathrm{ICR}}+b \\
0 & -x_{\mathrm{ICR}}-a
\end{array}\right]\left[\begin{array}{c}
v_{x} \\
\omega
\end{array}\right]$$

그리고 $\begin{array}{l}
v_{L}=v_{1 x}=v_{2 x} \\
v_{R}=v_{3 x}=v_{4 x}
\end{array}$ 이므로 모든 바퀴의 반지름이 같다는 가정하에 왼쪽바퀴와 오른쪽바퀴의 각속도를 묶어서 전체 바퀴의 각속도를 다음과 같이 쓸 수 있다.
$$\omega_{w}=\left[\begin{array}{c}
\omega_{L} \\
\omega_{R}
\end{array}\right]=\frac{1}{r}\left[\begin{array}{l}
v_{L} \\
v_{R}
\end{array}\right]$$

위의 두 식을 묶으면 로봇 전체(COM)의 속도에 대한 식은 각 바퀴의 각속도로 다음과 같이 표현 가능하다.
$$\boldsymbol{\eta}=\left[\begin{array}{c}
v_{x} \\
\omega
\end{array}\right]=r\left[\begin{array}{c}
\frac{\omega_{L}+\omega_{R}}{2} \\
\frac{-\omega_{L}+\omega_{R}}{2 c}
\end{array}\right]$$

(Caracciolo L., De Luca A. and Iannitti S. (1999): Trajectory tracking control of a four-wheel differentially driven mobile robot) 논문에 따르면 다음과 같은 제약이 들어간다. (왜 이런 제약이 필요한지는 이해가 되지는 않지만 위의 $\frac{v_{y}}{d_{C x}}=\omega$ 식을 통해 유도는 할 수 있다. )
$$v_{y}+x_{\mathrm{ICR}} \dot{\theta}=0$$

로봇 좌표계와 world 좌표계의 변환을 통해 $v_{y}=-\sin \theta \cdot \dot{X}+\cos \theta \cdot \dot{Y}$를 얻을 수 있고, 위의 식과의 결합을 통해 다음과 같이 표현할 수 있다.
$$\left[-\sin \theta \quad \cos \theta \quad x_{\mathrm{ICR}}\right][\dot{X} \quad \dot{Y} \quad \dot{\theta}]^{T}=A(\boldsymbol{q}) \dot{\boldsymbol{q}}=\mathbf{0}$$

그리고 world 좌표계에서의 속도를 각 바퀴와 COM의 운동에 대해 변환하면 다음과 같이 나타낼 수 있다.
$$\dot{\boldsymbol{q}}=\boldsymbol{S}(\boldsymbol{q}) \boldsymbol{\eta}$$
where 
$$\boldsymbol{S}^{T}(\boldsymbol{q}) \boldsymbol{A}^{T}(\boldsymbol{q})=\mathbf{0}$$
and
$$\boldsymbol{S}(\boldsymbol{q})=\left[\begin{array}{cc}
\cos \theta & x_{\mathrm{ICR}} \sin \theta \\
\sin \theta & -x_{\mathrm{ICR}} \cos \theta \\
0 & 1
\end{array}\right]$$

주목해야할점은 로봇의 운동은 2차원($v_{x}$, $\omega$) 으로 표현되고, world 좌표계에서의 움직임은 3차원($\dot{X} \quad \dot{Y} \quad \dot{\theta}$)으로 표현된다는 것이다. 또 흥미로운점은 SSMR에 대해 물리적으로 기술한 식들이 two-wheel mobile robot과 상당히 유사하다는 점이다.
