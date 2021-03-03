---
layout: post
title: (시작하세요! 텐서플로 2.0 프로그래밍) 4장. 회귀(Regression)
featured-img: 2021-03-03-ch4_Regression/fig6
permalink: /book_review/2021-03-03-ch4_Regression
category: book_review

---



## 선형 회귀

선형회귀(Linear Regression)는 데이터의 경향성을 가장 잘 설명하는 하나의 직선을 예측하는 것이다.

<br>

### 데이터 준비

아래 코드는 지역의 인구 증가율과 고령인구비율의 자료이다.

```python
import matplotlib.pyplot as plt
population_inc = [0.3, -0.78, 1.26, 0.03, 1.11, 15.17, 0.24, -0.24, -0.47, -0.77,
                  -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
population_old = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 
                  12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]
plt.plot(population_inc,population_old,'bo')
plt.xlabel('Population Growth Rate')
plt.ylabel('Elderly Population rate')
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-03-ch4_Regression/fig1.JPG?raw=true)

<br>

### outlier 제거

오른쪽 아래에 치우친 하나의 점이 있다. 이것은 극단치(outlier)라고 부르며 일반적인 경향에서 벗어나는 사례이다. 데이터의 일반적인 경향을 파악하기 위해 극단치는 제거한다.

```python
import matplotlib.pyplot as plt

population_inc = [0.3, -0.78, 1.26, 0.03, 1.11, 15.17, 0.24, -0.24, -0.47, -0.77,
                  -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
population_inc = population_inc[:5] + population_inc[6:]
population_old = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 
                  12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]
population_old = population_old[:5] + population_old[6:]
plt.plot(population_inc,population_old,'bo')
plt.xlabel('Population Growth Rate')
plt.ylabel('Elderly Population rate')
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-03-ch4_Regression/fig2.JPG?raw=true)

<br>

### 최소제곱법으로 회귀선 구하기

데이터의 경향성을 잘 설명하는 하나의 직선과 각 데이터의 차이를 잔차(residual)라고 한다. 이런 잔차의 제곱을 최소화하는 알고리즘을 최소제곱법(Least Square Method)라고 한다.

최소제곱법으로 직선 <img src="https://latex.codecogs.com/gif.latex?y=a&space;x&plus;b" title="y=a x+b" />의 a(기울기)와 b(y절편)를 구할 수 있습니다. 자세한 유도 과정은 생략합니다. a, b는 다음처럼 나타낼 수 있습니다.

<img src="https://latex.codecogs.com/png.latex?a=\frac{\sum_{i=1}^{n}\left(y_{i}-\bar{y}\right)\left(x_{i}-\bar{x}\right)}{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}},&space;\quad&space;b=\bar{y}-a&space;\bar{x}" title="a=\frac{\sum_{i=1}^{n}\left(y_{i}-\bar{y}\right)\left(x_{i}-\bar{x}\right)}{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}}, \quad b=\bar{y}-a \bar{x}" />

```python
import numpy as np
import matplotlib.pyplot as plt

population_inc = [0.3, -0.78, 1.26, 0.03, 1.11, 15.17, 0.24, -0.24, -0.47, -0.77,
                  -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
X = population_inc[:5] + population_inc[6:]
population_old = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 
                  12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]
Y = population_old[:5] + population_old[6:]

# X, Y의 평균을 구합니다.
x_bar = sum(X) / len(X)
y_bar = sum(Y) / len(Y)

# 최소제곱법으로 a, b를 구합니다.
a = sum([(y - y_bar) * (x - x_bar) for y, x in list(zip(Y, X))])
a /= sum([(x - x_bar) ** 2 for x in X])
b = y_bar - a * x_bar
print('a:', a, 'b:', b) # a: -0.355834147915461 b: 15.669317743971302

# 그래프를 그리기 위해 회귀선의  x, y 데이터를 구합니다.
line_x = np.arange(min(X), max(X), 0.01)
line_y = a * line_x + b

# 붉은색 실선으로 회귀선을 그립니다.
plt.plot(line_x,line_y,'r-')

plt.plot(X,Y,'bo')
plt.xlabel('Population Growth Rate')
plt.ylabel('Elderly Population rate')
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-03-ch4_Regression/fig3.JPG?raw=true)

<br>

### 텐서플로를 이용해 회귀선 구하기

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

population_inc = [0.3, -0.78, 1.26, 0.03, 1.11, 15.17, 0.24, -0.24, -0.47, -0.77,
                  -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
X = population_inc[:5] + population_inc[6:]
population_old = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 
                  12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]
Y = population_old[:5] + population_old[6:]

# a와 b를 랜덤한 값으로 초기화
a = tf.Variable(random.random())
b = tf.Variable(random.random())

# 잔차의 제곱의 평균을 반환하는 함수
def compute_loss():
  y_pred = a * X + b
  loss = tf.reduce_mean((Y - y_pred) ** 2)
  return loss

optimizer = tf.optimizers.Adam(lr=0.07)
for i in range(1001):
  # 잔차의 제곱의 평균을 최소화한다.
  optimizer.minimize(compute_loss, var_list=[a,b])

  if i % 200 == 0:
    print(i, 'a:', a.numpy(), 'b:', b.numpy(), 'loss:', compute_loss().numpy())

line_x = np.arange(min(X), max(X), 0.01)
line_y = a * line_x + b

# 그래프를 그린다.
plt.plot(line_x,line_y,'r-')
plt.plot(X,Y,'bo')
plt.xlabel('Population Growth rate (%)')
plt.ylabel('Elderly Population Rate (%)')
plt.show()

#---------------------출력---------------------#
0 a: 0.56431437 b: 0.36996573 loss: 243.31781
200 a: -0.11882495 b: 11.208977 loss: 29.629295
400 a: -0.32204953 b: 15.033493 loss: 10.184139
600 a: -0.35367328 b: 15.628651 loss: 9.782454
800 a: -0.35577506 b: 15.668211 loss: 9.780806
1000 a: -0.3558334 b: 15.669302 loss: 9.780804

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-03-ch4_Regression/fig4.JPG?raw=true)

<br>

<br>

## 다항 회귀

지금까지는 직선 즉 x의 차수가 1인 직선을 이용했습니다. 하지만 다항회귀에서는 차수가 2이상인 다항식을 이용해 회귀를 합니다.

<img src="https://latex.codecogs.com/png.latex?a&space;x^{2}&plus;b&space;x&plus;c" title="a x^{2}+b x+c" /> 를 회귀선으로 써보겠습니다.

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

population_inc = [0.3, -0.78, 1.26, 0.03, 1.11, 15.17, 0.24, -0.24, -0.47, -0.77,
                  -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
X = population_inc[:5] + population_inc[6:]
population_old = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 
                  12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]
Y = population_old[:5] + population_old[6:]

# a와 b를 랜덤한 값으로 초기화
a = tf.Variable(random.random())
b = tf.Variable(random.random())
c = tf.Variable(random.random())

# 잔차의 제곱의 평균을 반환하는 함수
def compute_loss():
  y_pred = a * X * X + b * X + c
  loss = tf.reduce_mean((Y - y_pred) ** 2)
  return loss

optimizer = tf.optimizers.Adam(lr=0.07)
for i in range(1001):
  # 잔차의 제곱의 평균을 최소화한다.
  optimizer.minimize(compute_loss, var_list=[a,b,c])

  if i % 200 == 0:
    print(i, 'a:', a.numpy(), 'b:', b.numpy(), 'c:', c.numpy(), 'loss:', compute_loss().numpy())

line_x = np.arange(min(X), max(X), 0.01)
line_y = a * line_x * line_x + b * line_x + c

# 그래프를 그린다.
plt.plot(line_x,line_y,'r-')
plt.plot(X,Y,'bo')
plt.xlabel('Population Growth rate (%)')
plt.ylabel('Elderly Population Rate (%)')
plt.show()

#---------------------출력---------------------#
0 a: 0.4410265 b: -0.0031149536 c: 0.39959562 loss: 233.03741
200 a: 2.89437 b: -4.606719 c: 10.1042385 loss: 32.95041
400 a: 0.35997823 b: -0.88201916 c: 14.4913025 loss: 11.144037
600 a: -0.4147796 b: 0.25616017 c: 15.83819 loss: 9.500473
800 a: -0.55013686 b: 0.45499793 c: 16.073593 loss: 9.456526
1000 a: -0.56382567 b: 0.4751067 c: 16.0974 loss: 9.456111

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-03-ch4_Regression/fig5.JPG?raw=true)

<br>

<br>

## 딥러닝 네트워크를 이용한 회귀

### 모델 생성

2층의 네트워크를 만들어본다. 첫 번째 층에서의 activation 함수는 tanh를 사용하고, optimizer는 SGD를 사용한다.

tanh는 입력을 받아 -1과 1사이의 출력을 반환합니다.

<img src="https://latex.codecogs.com/png.latex?\tanh&space;(x)=\frac{e^{x}-e^{-x}}{e^{x}&plus;e^{-x}}" title="\tanh (x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}" />

```python
import tensorflow as tf
import numpy as np

population_inc = [0.3, -0.78, 1.26, 0.03, 1.11, 15.17, 0.24, -0.24, -0.47, -0.77,
                  -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
X = population_inc[:5] + population_inc[6:]
population_old = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 
                  12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]
Y = population_old[:5] + population_old[6:]

model = tf.keras.Sequential([
              tf.keras.layers.Dense(units=6, activation='tanh', input_shape=(1,)),
              tf.keras.layers.Dense(units=1)
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mse')

model.summary()

#---------------------출력---------------------#
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 6)                 12        
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 7         
=================================================================
Total params: 19
Trainable params: 19
Non-trainable params: 0
_________________________________________________________________

```

<br>

### 학습

```python
model.fit(X,Y,epochs=10)

#---------------------출력---------------------#
Epoch 1/10
1/1 [==============================] - 0s 384ms/step - loss: 258.9460
Epoch 2/10
1/1 [==============================] - 0s 4ms/step - loss: 111.8914
Epoch 3/10
1/1 [==============================] - 0s 3ms/step - loss: 10.2768
Epoch 4/10
1/1 [==============================] - 0s 3ms/step - loss: 9.5281
Epoch 5/10
1/1 [==============================] - 0s 3ms/step - loss: 9.4042
Epoch 6/10
1/1 [==============================] - 0s 3ms/step - loss: 9.3591
Epoch 7/10
1/1 [==============================] - 0s 3ms/step - loss: 9.3197
Epoch 8/10
1/1 [==============================] - 0s 3ms/step - loss: 9.2789
Epoch 9/10
1/1 [==============================] - 0s 4ms/step - loss: 9.2366
Epoch 10/10
1/1 [==============================] - 0s 8ms/step - loss: 9.1938

```

딥러닝에서는 충분히 학습했다고 판단하면 학습을 종료해야한다. 그렇지 않으면 학습 데이터에 overfitting되어 새로운 데이터가 들어왔을 때 유연하게 대처를 못할 수 있다.

따라서 학습 데이터 중 일부를 떼어내어 검증 데이터(validation data)를 설정하는 것은 학습을 언제 멈출지 결정하는 데 좋은 판단 기준이 된다.

<br>

### 그래프 그리기

```python
import matplotlib.pyplot as plt

line_x = np.arange(min(X), max(X), 0.01)
line_y = model.predict(line_x)

plt.plot(line_x,line_y,'r-')
plt.plot(X,Y,'bo')
plt.xlabel('Population Growth rate (%)')
plt.ylabel('Elderly Population Rate (%)')
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-03-ch4_Regression/fig6.JPG?raw=true)

<br>

<br>

## 보스턴 주택 가격 데이터셋

보스턴 주택 가격 데이터셋은 1978년 미국 보스턴 지역의 주택 가격으로, 506개 타운의 주택 가격 중앙값을 1,000 달러 단위로 나타낸다. 범죄율, 주택당 방 개수, 고속도로까지의 거리 등 13가지 데이터를 이용해 주택 가격을 예측해야한다.

<br>

### 데이터셋 불러오기

학습 데이터는 404개, 시험 데이터는 102개이다.

```python
from tensorflow.keras.datasets import boston_housing

(train_X, train_Y), (test_X, test_Y) = boston_housing.load_data()
print(len(train_X), len(test_X))
print(train_X[0])
print(train_Y[0])

#---------------------출력---------------------#
404 102
[  1.23247   0.        8.14      0.        0.538     6.142    91.7
   3.9769    4.      307.       21.      396.9      18.72   ]
15.2

```

<br>

### 데이터셋 전처리

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-03-ch4_Regression/fig9.JPG?raw=true)

데이터의 특성을 살펴보면 비율인것도 있고, 0/1로 나타내는 데이터도 있다. 이런 데이터를 전처리로 정규화를 할 것이다.

데이터를 정규화하려면 각 데이터에서 평균값을 뺀 다음 표준편차로 나눈다. 이것은 데이터의 분포를 정규분포로 옮기는 역할을 한다.

```python
x_mean = train_X.mean(axis=0)
x_std = train_X.std(axis=0)
train_X -= x_mean
train_X /= x_std
test_X -= x_mean
test_X /= x_std

Y_mean = train_Y.mean(axis=0)
Y_std = train_Y.std(axis=0)
train_Y -= Y_mean
train_Y /= Y_std
test_Y -= Y_mean
test_Y /= Y_std

print(train_X[0])
print(train_Y[0])

#---------------------출력---------------------#
[-0.27224633 -0.48361547 -0.43576161 -0.25683275 -0.1652266  -0.1764426
  0.81306188  0.1166983  -0.62624905 -0.59517003  1.14850044  0.44807713
  0.8252202 ]
-0.7821526033779157

```

### 네트워크 구현

```python
model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=52, activation='relu', input_shape=(13,)),
                tf.keras.layers.Dense(units=39, activation='relu'),
                tf.keras.layers.Dense(units=26, activation='relu'),
                tf.keras.layers.Dense(units=1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='mse')

model.summary()

#---------------------출력---------------------#
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_4 (Dense)              (None, 52)                728       
_________________________________________________________________
dense_5 (Dense)              (None, 39)                2067      
_________________________________________________________________
dense_6 (Dense)              (None, 26)                1040      
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 27        
=================================================================
Total params: 3,862
Trainable params: 3,862
Non-trainable params: 0
_________________________________________________________________

```

<br>

### 학습

validation_split을 설정하면 학습 데이터셋에서 일부를 잘라서 validation loss를 출력해준다.

```python
history = model.fit(train_X, train_Y, epochs=25, batch_size=32, validation_split=0.25)

#---------------------출력---------------------#
Epoch 1/25
10/10 [==============================] - 1s 20ms/step - loss: 1.5722 - val_loss: 1.6583
Epoch 2/25
10/10 [==============================] - 0s 5ms/step - loss: 0.7087 - val_loss: 0.5186
Epoch 3/25
10/10 [==============================] - 0s 5ms/step - loss: 0.4286 - val_loss: 0.3726
Epoch 4/25
10/10 [==============================] - 0s 4ms/step - loss: 0.2472 - val_loss: 0.2991
Epoch 5/25
10/10 [==============================] - 0s 4ms/step - loss: 0.1706 - val_loss: 0.2234
(이하 생략)

```

<br>

### 그래프 그리기

```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-03-ch4_Regression/fig7.JPG?raw=true)

훈련 데이터의 손실이 비교적 꾸준히 감소하는데 비해 붉은색 점선으로 표시된 검증 데이터의 손실은 항상 감소하지 않는다.

<br>

### 평가

위에서의 훈련 데이터의 loss값(0.07)과 시험 데이터의 loss값을 비교하면 차이가 있다. 이는 모델이 overfitting 되었음을 시사한다.

```python
model.evaluate(test_X, test_Y)

#---------------------출력---------------------#
4/4 [==============================] - 0s 2ms/step - loss: 0.502

```

<br>

<br>

### Early Stopping

위와같이 모델이 overfitting 되는것을 막기 위해서는 학습 도중에 끼어서 학습을 멈춰야 한다. 여기서는 val_loss가 3번 연속으로 감소하지 않는다면 학습을 멈춥니다. model.fit에 callback을 전달하여 Earlystopping을 하도록 합니다.

```python
model = tf.keras.Sequential([
                tf.keras.layers.Dense(units=52, activation='relu', input_shape=(13,)),
                tf.keras.layers.Dense(units=39, activation='relu'),
                tf.keras.layers.Dense(units=26, activation='relu'),
                tf.keras.layers.Dense(units=1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='mse')

history = model.fit(train_X, train_Y, epochs=25, batch_size=32, validation_split=0.25,
         callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')])

#---------------------출력---------------------#
Epoch 1/25
10/10 [==============================] - 0s 15ms/step - loss: 1.7776 - val_loss: 0.6405
Epoch 2/25
10/10 [==============================] - 0s 5ms/step - loss: 0.2946 - val_loss: 0.2502
Epoch 3/25
10/10 [==============================] - 0s 5ms/step - loss: 0.3182 - val_loss: 0.4117
...
(부분 생략)
...
Epoch 12/25
10/10 [==============================] - 0s 5ms/step - loss: 0.1675 - val_loss: 0.3281
Epoch 13/25
10/10 [==============================] - 0s 4ms/step - loss: 0.1875 - val_loss: 0.2602

```

25번의 epoch에서 13번만 하고 끝난 것을 확인할 수 있습니다.

<br>

#### 그래프 그리기

```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-03-ch4_Regression/fig8.JPG?raw=true)

<br>

#### 평가

Earlystopping을 하지 않았을 때보다 loss가 낮아진 것을 확인할 수 있다.

```python
model.evaluate(test_X, test_Y)

#---------------------출력---------------------#
4/4 [==============================] - 0s 2ms/step - loss: 0.3032

```