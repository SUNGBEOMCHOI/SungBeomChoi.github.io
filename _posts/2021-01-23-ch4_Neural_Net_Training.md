---
layout: post
title: (밑바닥부터 시작하는 딥러닝) 4장. 신경망 학습
featured-img: 2021-01-23-ch4_Neural_Net_Training/fig4
permalink: /book_review/ch4_Neural_Net_Training
category: book_review

---
## 패러다임의 변화

![Paradigm](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-23-ch4_Neural_Net_Training/fig1.jpg?raw=true)

학습이란 훈련 데이터로부터 가중치 매개변수의 최적값을 자동으로 획득하는 것을 뜻한다. 기계학습은 데이터에서 패턴을 발견하여 학습한다.
이미지에서 '5'라는 숫자를 인식하는 프로그램을 구현한다고 해보자. 손글씨를 보고 5라는 숫자를 직접 코드를 짜는 것은 어렵습니다. 초기에는 사람이 생각한 알고리즘으로 시도를 했다.
이후에는 이미지에서 사람이 생각하는 특징을 추출(SIFT, SURF, HOG)하고, 그 특징의 패턴을 기계학습 기술(SVM, KNN)로 학습하였다. 
이 방식에서 더욱 발전하여 이후에는 완전히 데이터를 기계가 학습하는 방식의 딥러닝을 활용하였다.

<br>
<br>

## 손실함수
최적의 매개변수 값을 설정하기 위해서는 평가할 수 있는 지표가 필요하다. 신경망 학습에서는 손실 함수를 사용한다.

-  오차제곱합

회귀문제에서 주로 사용된다.

![MSE](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-23-ch4_Neural_Net_Training/fig6.jpg?raw=true)
 
```python
def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)
```
    
-   교차 엔트로피 오차

분류문제에서 주로 사용한다. 

   ![MSE](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-23-ch4_Neural_Net_Training/fig7.jpg?raw=true)
    
```python
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
```

<br>
<br>

## 미니배치
위 예시처럼 하나의 데이터로 모델을 평가하는 것이 아닌 일부로부터 데이터를 평가한다. 이 일부를 미니배치라고 한다. MNIST에서 미니배치에 대한 에러를 출력해보겠다.

```python
import sys, os
sys.path.append(os.path.join(os.getcwd(), 'deep-learning-from-scratch-master')) #'deep-learning-from-scratch-master' 폴더를 경로추가
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \\
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape) # (60000, 784)
print(x_train.shape) # (60000, 10)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) 
# 0~59999의 수 중 배치 사이즈(10)만큼 무작위로 뽑아 ndarray 리턴

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

def cross_entropy_error(y, t):
    if y.ndim == 1:
				# 답이 레이블형식이면 one-hot으로 변경
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7)) / batch_size
```

<br>
<br>

## 기울기

기울기는 각 요소에 대한 편미분을 모두 합친것이다. 기울기가 가리키는 방향은 각 장소에서 함수의 출력값을 가장 크게 줄이는 방향이다.(local minimum에 빠질 수 있음)

```python
# 미분
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

# 기울기(1차원 리스트만 해당)
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성
    
    for idx in range(x.size):
        tmp_val = x[idx]
        #f(x+h) 계산
        x[idx] = tmp_val+h
        fxh1 = f(x)        
        #f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        
    return grad

```

```python
def function(x):
    return x[0] ** 2 + x[1] ** 2

numerical_gradient(function, np.array([3.0, 4.0]))
```

<br>
<br>

## 경사하강법

최적의 매개변수를 찾아야 하는데, 최적은 손실 함수가 최솟값이 되어야한다. 기울기를 이용해 손실함수의 최솟값을 찾는 방법을 경사하강법이라고 한다. 여기서는 학습률을 설정하여 얼마만큼 기울기를 따라갈지를 정한다.

![gradient descent](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-23-ch4_Neural_Net_Training/fig8.jpg?raw=true)

```python
def gradient_descent(f, init_x, lr=0.01, step_num=100): 
#f에 손실함수, init_x에 가중치 매개변수, lr에 학습률을 넣어줌
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x) # 기울기를 구함
        x -= lr * grad # 경사 하강법
    return x
```

<br>
<br>

## 신경망에서 최적의 매개변수 찾기

### 네트워크 설정 
```python
import sys, os
sys.path.append(os.path.join(os.getcwd(), 'deep-learning-from-scratch-master')) #'deep-learning-from-scratch-master' 폴더를 경로추가
import numpy as np
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads

```

### 실제 학습
```python
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# 데이터 불러옴

iters_num = 10000 #미니배치 반복횟수
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

train_loss_list = [] 
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1) 
#1에폭 = 미니배치로 모든 데이터를 돌았을때

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)
    
    #매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + "," + str(test_acc))
```

### 학습결과
92% 정도의 정확도가 나왔다.

![accuracy](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-23-ch4_Neural_Net_Training/fig5.jpg?raw=true)
