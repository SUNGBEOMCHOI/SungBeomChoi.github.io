---
layout: post
title: (시작하세요! 텐서플로 2.0 프로그래밍) 3장. 텐서플로 2.0 시작하기
featured-img: 2021-03-03-ch3_Start_Tensorflow/fig2
permalink: /book_review/2021-03-03-ch3_Start_Tensorflow
category: book_review

---


기본적으로 코드들은 코랩에서 실행한다.

<br>

## 텐서플로 2.x 버전 가져오기

```python
try:
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf
print(tf.__version__) # 2.4.1

```
<br>
<br>

## 난수 생성

```python
# 모두 같은 확률로 난수를 뽑음
rand = tf.random.uniform([1], 0, 1) # tf.random.uniform(shape, 최소값, 최대값)
print(rand) # tf.Tensor([0.6305238], shape=(1,), dtype=float32)

```
<br>
<br>

```python
# 정규분포로 난수를 뽑음
rand = tf.random.normal([4], 0, 1) # tf.random.noraml(shape, 평균, 표준편차)
print(rand)  
# tf.Tensor([-1.389584   -0.10909464 -0.3136603  -0.4502219 ], shape=(4,), dtype=float32)

```

<br>
<br>

## 퍼셉트론 생성

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-03-ch3_Start_Tensorflow/fig2.JPG?raw=true)

```python
# 활성화함수 sigmoid
import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

```

```python
x = 1
y = 0
w = tf.random.normal([1], 0, 1)  # 난수생성
output = sigmoid(x * w)  # 활성화 함수 적용

```

<br>
<br>

## 경사 하강법

output을 최대한 정답에 가깝게 맞추도록 가중치를 갱신하는 방법이다. 반복될수록 error는 점점 줄어들고, output은 정답인 0에 가까워지는 것을 볼 수 있다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-03-ch3_Start_Tensorflow/fig1.JPG?raw=true)

```python
for i in range(1000):
  output = sigmoid(x * w)
  error = y - output
  w = w + x * 0.1 * error  # 학습률은 0.1로 설정

  if i % 100 == 99:
    print(i, error, output)

#---------------------출력---------------------#
# 99 -0.11310331711500524 0.11310331711500524
# 199 -0.05525516691999154 0.05525516691999154
# 299 -0.036134545309195125 0.036134545309195125
# 399 -0.02675293191878269 0.02675293191878269
# 499 -0.02120766920554343 0.02120766920554343
# 599 -0.01755329196981286 0.01755329196981286
# 699 -0.014966658229093838 0.014966658229093838
# 799 -0.013040807186870356 0.013040807186870356
# 899 -0.011551897156336102 0.011551897156336102
# 999 -0.010366791686728452 0.010366791686728452

```

<br>

#### 만약 x=0인 경우에는 어떻게 될까?

에러도 0.5에서 더 줄어들지 않고, output도 0.5에서 변하지 않는다. x=0이기 때문에 학습이 되지 않는다. 이런 경우를 위해 bias(b)가 필요하다.

```python
x = 0
y = 1
w = tf.random.normal([1], 0, 1)
output = sigmoid(x * w)
print(output)  # 0.5

for i in range(1000):
  output = sigmoid(x * w)
  error = y - output
  w = w + x * 0.1 * error

  if i % 100 == 99:
    print(i, error, output)

#---------------------출력---------------------#
# 99 0.5 0.5
# 199 0.5 0.5
# 299 0.5 0.5
# 399 0.5 0.5
# 499 0.5 0.5
# 599 0.5 0.5
# 699 0.5 0.5
# 799 0.5 0.5
# 899 0.5 0.5
# 999 0.5 0.5

```

<br>


#### bias를 추가한 경우

error가 줄어들고, output도 정답인 1에 거의 근접했다.

```python
x = 0
y = 1
w = tf.random.normal([1], 0, 1)
b = tf.random.normal([1], 0, 1)

for i in range(1000):
  output = sigmoid(x * w + 1 * b)
  error = y - output
  w = w + x * 0.1 * error
  b = b + 1 * 0.1 * error

  if i % 100 == 99:
    print(i, error, output)

#---------------------출력---------------------#
# 99 0.11046792076952594 0.8895320792304741
# 199 0.05457995828296547 0.9454200417170345
# 299 0.03583897739278796 0.964161022607212
# 399 0.026589096309596272 0.9734109036904037
# 499 0.021104036753882838 0.9788959632461172
# 599 0.017481991938524843 0.9825180080614752
# 699 0.014914663244828152 0.9850853367551718
# 799 0.013001207731670572 0.9869987922683294
# 899 0.01152079468956213 0.9884792053104379
# 999 0.010341715869476853 0.9896582841305231

```

<br>
<br>

## 신경망 네트워크

### And 게이트

And 게이트는 입력이 모두 1일때만 1을 출력한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-03-ch3_Start_Tensorflow/fig3.JPG?raw=true)

```python
import numpy as np

x = np.array([[0,0], [1,0], [0,1], [1,1]])
y = np.array([[0], [0], [0], [1]])
w = tf.random.normal([2],0,1)
b = tf.random.normal([1],0,1)
b_x = 1

for i in range(2001):
  error_sum = 0
  for j in range(4):
    output = sigmoid(np.sum(x[j]*w)+b_x*b)
    error = y[j][0] - output
    w = w + x[j] * 0.1 * error
    b = b + b_x * 0.1 * error
    error_sum += error

  if i % 400 == 0:
    print(i, error_sum)

#---------------------출력---------------------#
0 -1.5467700244922273
400 -0.06479694274470607
800 -0.03580663398071523
1200 -0.02458656329598055
1600 -0.01867050284398148
2000 -0.015027795258891878

```

학습된 가중치가 실제로 맞는 값을 가르키는지 살펴보자.

```python
for i in range(4):
  print('input:', x[i], '  output:', sigmoid(np.sum(x[i]*w)+b_x*b))

#---------------------출력---------------------#
input: [1 1]   output: 0.9650488962798548
input: [1 0]   output: 0.024768898672699782
input: [0 1]   output: 0.024844223870939347
input: [0 0]   output: 2.3434300193304377e-05

```

<br>
<br>

### OR 게이트

OR 게이트는 입력 중 하나만 1이어도 1을 출력한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-03-ch3_Start_Tensorflow/fig4.JPG?raw=true)

```python
= np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([[0], [1], [1], [1]])
w = tf.random.uniform([2],0,1)
b = tf.random.uniform([1],0,1)
b_x = 1

for i in range(2001):
  error_sum = 0
  for j in range(4):
    output = sigmoid(np.sum(x[j]*w)+b_x*b)
    error = y[j][0] - output
    w = w + x[j] * 0.1 * error
    b = b + b_x * 0.1 * error
    error_sum += error

  if i % 400 == 0:
    print(i, error_sum)

#---------------------출력---------------------#
0 0.5169845174011798
400 -0.025759148021826647
800 -0.013033428543949538
1200 -0.00866607518100053
1600 -0.006475455255724974
2000 -0.005162616073124557

```

<br>

학습된 가중치가 실제로 맞는 값을 가르키는지 살펴보자.

```python
for i in range(4):
  print('input:', x[i], '  output:', sigmoid(np.sum(x[i]*w)+b_x*b))

#---------------------출력---------------------#
input: [0 0]   output: 0.025601225400094334
input: [1 0]   output: 0.9898147401582366
input: [0 1]   output: 0.9897890663503223
input: [1 1]   output: 0.9999972109037591

```
<br>
<br>

### XOR 게이트

XOR게이트는 입력의 값이 서로 다를때만 1을 출력한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-03-ch3_Start_Tensorflow/fig5.JPG?raw=true)

```python
x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([[0], [1], [1], [0]])
w = tf.random.uniform([2],0,1)
b = tf.random.uniform([1],0,1)
b_x = 1

for i in range(2001):
  error_sum = 0
  for j in range(4):
    output = sigmoid(np.sum(x[j]*w)+b_x*b)
    error = y[j][0] - output
    w = w + x[j] * 0.1 * error
    b = b + b_x * 0.1 * error
    error_sum += error

  if i % 400 == 0:
    print(i, error_sum)

#---------------------출력---------------------#
0 -0.5038278035452097
400 0.00010551105352096801
800 1.6659719226375103e-07
1200 4.746623760709667e-08
1600 4.746623760709667e-08
2000 4.746623760709667e-08

```

<br>

학습된 가중치가 실제로 맞는 값을 가르키는지 살펴보자.

```python
for i in range(4):
  print('input:', x[i], '  output:', sigmoid(np.sum(x[i]*w)+b_x*b))

#---------------------출력---------------------#
input: [0 0]   output: 0.5128175691057347
input: [1 0]   output: 0.49999997485429043
input: [0 1]   output: 0.4871823676059484
input: [1 1]   output: 0.47438160794816525

```

잘 학습이 되지 않았다. 한 층의 퍼셉트론으로는 XOR 게이트를 구현할 수 없다. 여러개의 퍼셉트론을 사용해야한다. tf.Keras를 통해 구현해본다.

<br>
<br>

#### 2개의 층으로 XOR구현

```python
x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = tf.keras.Sequential([
          tf.keras.layers.Dense(units=2, activation='sigmoid', input_shape=(2,)),
          tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mse')

model.summary()

#---------------------출력---------------------#
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 2)                 6         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 3         
=================================================================
Total params: 9
Trainable params: 9
Non-trainable params: 0
_________________________________________________________________

```

<br>


#### 학습

```python
history = model.fit(x, y, epochs=2000, batch_size=1)

```

<br>


#### 확인

x를 넣어서 학습이 잘 되었는지 확인해본다. 조금은 아쉽지만 학습이 되는것은 확인할 수 있다.

```python
model.predict(x)
#---------------------출력---------------------#
array([[0.23168063],
       [0.7207968 ],
       [0.78110605],
       [0.20379293]], dtype=float32)

```

<br>


#### 가중치 확인

```python
for weight in model.weights:
  print(weight)
#---------------------출력---------------------#
<tf.Variable 'dense/kernel:0' shape=(2, 2) dtype=float32, numpy=
array([[-3.1059098,  5.11057  ],
       [ 3.3211062, -4.8253274]], dtype=float32)>
<tf.Variable 'dense/bias:0' shape=(2,) dtype=float32, numpy=array([1.5160486, 2.7006598], dtype=float32)>
<tf.Variable 'dense_1/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[-3.6596467],
       [-3.734692 ]], dtype=float32)>
<tf.Variable 'dense_1/bias:0' shape=(1,) dtype=float32, numpy=array([5.301554], dtype=float32)>

```

<br>
<br>

## 시각화 기초

### matplotlib.pyplot을 이용한 그래프 그리기

#### 꺽은선 그래프

```python
import matplotlib.pyplot as plt
x = range(20)
y = tf.random.normal([20],0,1)
plt.plot(x,y)
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-03-ch3_Start_Tensorflow/fig6.JPG?raw=true)

<br>

#### 분산형 그래프

```python
x = range(20)
y = tf.random.normal([20],0,1)
plt.plot(x,y,'bo')
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-03-ch3_Start_Tensorflow/fig7.JPG?raw=true)

<br>

#### 히스토그램

```python
random_normal = tf.random.normal([100000],0,1)
plt.hist(random_normal, bins=100)
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-03-ch3_Start_Tensorflow/fig9.JPG?raw=true)

<br>

#### XOR 2층의 퍼셉트론 loss 그래프

위에서 만들었던 XOR 2층 퍼셉트론의 학습하면서 기록된 loss그래프를 그려보자.

loss가 주는것을 보면 학습이 된다는 것을 알 수 있다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-03-ch3_Start_Tensorflow/fig8.JPG?raw=true)