---
layout: post
title: Smart Patrolling An Efficient Road Surface Monitoring Using Smartphone Sensor and Crowdsourcing Sensors Review
featured-img: 2020-07-05-Smart_Patrolling_An_Efficient_Road_Surface_Monitoring_Using_Smartphone_Sensor_and_Crowdsourcing/figure7
---

#### 이 글은 [Smart Patrolling: An Efficient Road Surface Monitoring Using Smartphone Sensors and Crowdsourcing](https://www.sciencedirect.com/science/article/abs/pii/S1574119216301262) 논문의 리뷰이다.

## introduction
road surface monitoring기술은 사고의 예방차원에서 중요하다. 요즘에는  smartphone을 통해 monitoring system을 개발하는 것이 트렌드이다. 

이 논문에서는 기존의 threshold-based와 machine learning의 한계를 극복하기 위해 DTW technique를 사용했다. DTW는 미리 뽑아놓은 reference template과 센서 데이터를 비교하여 유사도를 계산하고, anomaly인지 판단하는 방법을 뜻한다.

기존의 방식에 비교해서 DTW 방식의 장점은 다음과 같다.
- ML방식은 시간복잡도가 <img src="https://latex.codecogs.com/gif.latex?\mathrm{O}\left(m*n^{2}\right)" title="\mathrm{O}\left(m*n^{2}\right)" /> 이지만 DTW방식은 <img src="https://latex.codecogs.com/gif.latex?\mathrm{O}\left(n^{2}\right)" title="\mathrm{O}\left(n^{2}\right)" />이다.(n는 표본수, m은 class수)
- ML 방식과는 달리 학습이 필요하지 않고, 많은 데이터가 필요하지 않다.
- threshold based는 환경에 따른 threshold가 다 따로 있어야하지만 이것은 하나만 있어도 괜찮다.

## 과정
![process](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-07-05-Smart_Patrolling_An_Efficient_Road_Surface_Monitoring_Using_Smartphone_Sensor_and_Crowdsourcing/figure3.jpg?raw=true)

### filter
gravity와 vehicle의 vibration을 지우기 위해 filter를 적용해야한다. accelerometer data를 smoothen하고, normalize한다.

- Speed
vehicle이 stationary하면 acclerometer가 significant change를 보이지 않는다. 따라서 이런 data는 없애야한다. GPS를 통해 속도를 계산하고, 5kmph인 데이터는 제거해주었다.
- Virtual Orientation
sensor의 좌표를 vehicle의 좌표로 변경해주어야한다. 이상적인 값은 아래와 같다.   
<img src="https://latex.codecogs.com/gif.latex?a_{x}=0&space;m&space;/&space;s^{2},&space;a_{y}=0&space;m&space;/&space;s^{2},&space;a_{z}=9.81&space;\mathrm{m}&space;/&space;\mathrm{s}^{2}" title="a_{x}=0 m / s^{2}, a_{y}=0 m / s^{2}, a_{z}=9.81 \mathrm{m} / \mathrm{s}^{2}" />
<img src="https://latex.codecogs.com/gif.latex?\alpha(\text&space;{&space;roll&space;angle&space;})" title="\alpha(\text { roll angle })" />, <img src="https://latex.codecogs.com/gif.latex?\beta(\text&space;{&space;pitch&space;angle&space;})" title="\beta(\text { pitch angle })" />, <img src="https://latex.codecogs.com/gif.latex?\gamma(\text&space;{&space;yaw&space;angle&space;})" title="\gamma(\text { yaw angle })" /> 이고, sensor의 값으로 표현하면 다음과 같다. 
<img src="https://latex.codecogs.com/gif.latex?\alpha=\tan&space;^{-1}\left(\frac{a_{y}}{a_{z}}\right)" title="\alpha=\tan ^{-1}\left(\frac{a_{y}}{a_{z}}\right)" />  , <img src="https://latex.codecogs.com/gif.latex?\beta=\tan&space;^{-1}\left(\frac{-a_{x}}{\sqrt{a_{y}^{2}&plus;a_{z}^{2}}}\right)" title="\beta=\tan ^{-1}\left(\frac{-a_{x}}{\sqrt{a_{y}^{2}+a_{z}^{2}}}\right)" />

   reoriented acceleration은 다음과 같이 계산된다.   
<img src="https://latex.codecogs.com/gif.latex?\begin{array}{l}&space;a_{x}^{\prime}=\cos&space;(\beta)&space;a_{x}&plus;\sin&space;(\beta)&space;\sin&space;(\alpha)&space;a_{y}&plus;\cos&space;(\alpha)&space;\sin&space;(\beta)&space;a_{z}&space;\\&space;a_{y}^{\prime}=\cos&space;(\alpha)&space;a_{y}-\sin&space;(\alpha)&space;a_{z}&space;\\&space;a_{z}^{\prime}=-\sin&space;(\beta)&space;a_{x}&plus;\cos&space;(\beta)&space;\sin&space;(\alpha)&space;a_{y}&plus;\cos&space;(\beta)&space;\cos&space;(\alpha)&space;a_{z}&space;\end{array}" title="\begin{array}{l} a_{x}^{\prime}=\cos (\beta) a_{x}+\sin (\beta) \sin (\alpha) a_{y}+\cos (\alpha) \sin (\beta) a_{z} \\ a_{y}^{\prime}=\cos (\alpha) a_{y}-\sin (\alpha) a_{z} \\ a_{z}^{\prime}=-\sin (\beta) a_{x}+\cos (\beta) \sin (\alpha) a_{y}+\cos (\beta) \cos (\alpha) a_{z} \end{array}" />

- Filtering Z-axis
X, Y-axis는 제거되고, Z-axis만 취한다.

- SMA(Simple Moving Average)
SMA를 통해 accelerometer를 부드럽게 만들고, vehicle vibration noise를 제거한다.
- Band pass filter
hardware sensitivity에 의한 noise를 제거해준다. low and high pass filter 중 하나인 BPF를 적용했다. 
   - Using Low pass filter   
<img src="https://latex.codecogs.com/gif.latex?\begin{array}{l}&space;g_{x_{n}}=\delta&space;*&space;g_{x_{n-1}}&plus;(1-\delta)&space;*&space;a_{x_{x}}&space;\\&space;g_{y_{n}}=\delta&space;*&space;g_{y_{n-1}}&plus;(1-\delta)&space;*&space;a_{y}&space;\\&space;g_{z_{n}}=\delta&space;*&space;g_{z_{n-1}}&plus;(1-\delta)&space;*&space;a_{z}&space;\end{array}" title="\begin{array}{l} g_{x_{n}}=\delta * g_{x_{n-1}}+(1-\delta) * a_{x_{x}} \\ g_{y_{n}}=\delta * g_{y_{n-1}}+(1-\delta) * a_{y} \\ g_{z_{n}}=\delta * g_{z_{n-1}}+(1-\delta) * a_{z} \end{array}" />
   - Using High pass filter   
<img src="https://latex.codecogs.com/gif.latex?\begin{array}{l}&space;a_{x}^{\prime}=a_{x}-g_{x_{n}}&space;\\&space;a_{y}^{\prime}=a_{y}-g_{y_{n}}&space;\\&space;a_{z}^{\prime}=a_{z}-g_{z&space;n}&space;\end{array}" title="\begin{array}{l} a_{x}^{\prime}=a_{x}-g_{x_{n}} \\ a_{y}^{\prime}=a_{y}-g_{y_{n}} \\ a_{z}^{\prime}=a_{z}-g_{z n} \end{array}" />

### DTW
DTW는 two sequences of time series의 유사도를 계산한다. 이 방식은 speech recognition과 같은 분야에서 먼저 쓰이던 방식이다. 시간 복잡도는 <img src="https://latex.codecogs.com/gif.latex?\mathrm{O}\left(n^{2}\right)" title="\mathrm{O}\left(n^{2}\right)" />이다.

![DTW](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-07-05-Smart_Patrolling_An_Efficient_Road_Surface_Monitoring_Using_Smartphone_Sensor_and_Crowdsourcing/figure9.jpg?raw=true)

warping cost는 다음과 같다. warping cost가 작을수록 유사도가 높은 것이다.   
<img src="https://latex.codecogs.com/gif.latex?\mathrm{DTW}(\mathrm{A},&space;\mathrm{B})=\min&space;\{\sqrt{\sum_{k=1}^{K}&space;w_{k}}\}" title="\mathrm{DTW}(\mathrm{A}, \mathrm{B})=\min \{\sqrt{\sum_{k=1}^{K} w_{k}}\}" />

<img src="https://latex.codecogs.com/gif.latex?w_{k}" title="w_{k}" />는 warping path W의 matrix element <img src="https://latex.codecogs.com/gif.latex?(i,&space;j)_{k}" title="(i, j)_{k}" /> 이다.

warping path는 다음식을 dynamic programming을 통해 푼다.   
<img src="https://latex.codecogs.com/gif.latex?\gamma(\mathrm{i},&space;\mathrm{j})=\mathrm{d}\left(a_{i},&space;b_{j}\right)&plus;\min&space;\{\gamma(\mathrm{i}-1,&space;\mathrm{j}-1),&space;\gamma(\mathrm{i}-1,&space;\mathrm{j}),&space;\gamma(\mathrm{i},&space;\mathrm{j}-1)\}" title="\gamma(\mathrm{i}, \mathrm{j})=\mathrm{d}\left(a_{i}, b_{j}\right)+\min \{\gamma(\mathrm{i}-1, \mathrm{j}-1), \gamma(\mathrm{i}-1, \mathrm{j}), \gamma(\mathrm{i}, \mathrm{j}-1)\}" />

### Template Reference
Template reference는 비교를 위해 anomalies에 대한 특성을 미리 저장해두는 것이다. DTW방식의 성능은 template reference에 따라 달라진다. Template은 여러 샘플에 대해서 recognition rate를 비교하고, 높은 것들을 저장한다.

## 평가
### Filter에 따른 결과
![filter](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-07-05-Smart_Patrolling_An_Efficient_Road_Surface_Monitoring_Using_Smartphone_Sensor_and_Crowdsourcing/figure5.jpg?raw=true)
EWMA와 Holt Winter 방식은 evnet와 output이 비슷하게 나왔으나, data가 smooth하지 않다. SMA+BPF 방식이 가장 data가 smooth하고, event와 output도 비슷했다.

### Reorientation 에 따른 결과
![reorient](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-07-05-Smart_Patrolling_An_Efficient_Road_Surface_Monitoring_Using_Smartphone_Sensor_and_Crowdsourcing/figure6.jpg?raw=true)
reorientation된 데이터가 안된것보다 data가 더 smooth하고, event부분과 아닌 부분의 차이가 더 컸다.

### Template reference
![template](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-07-05-Smart_Patrolling_An_Efficient_Road_Surface_Monitoring_Using_Smartphone_Sensor_and_Crowdsourcing/figure7.jpg?raw=true)
Template을 보면 pothole에서는 acceleration값이 처음에는 감소하고, 나중에 증가하지만 bump에서는 처음에는 증가하고, 나중에 감소한다.   

<img src="https://latex.codecogs.com/gif.latex?U&space;B=M&space;E&space;A&space;N(k)-\mu&space;*&space;S&space;T&space;D&space;E&space;V(k)" title="U B=M E A N(k)-\mu * S T D E V(k)" />
k는 number of windows이고,  <img src="https://latex.codecogs.com/gif.latex?\mu" title="\mu" /> 는 경험적으로 1.1로 설정되었다. DTW를 통해 계산된 distance가 UB보다 작을 경우 그 위치는 anomaly로 체크된다.

### 다른 approach와의 비교
![other approach](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-07-05-Smart_Patrolling_An_Efficient_Road_Surface_Monitoring_Using_Smartphone_Sensor_and_Crowdsourcing/figure8.jpg?raw=true)
wolverine은 ML based중 SVM을 활용한 기법이고, Nericell은 threshold based의 방법 중 하나이다. 이 연구들과 비교하여 DTW방식은 FN와 Accuracy모두 높았다.
