---
layout: post
title: (밑바닥부터 시작하는 딥러닝, 3장) 신경망
featured-img: 2021-01-16-ch3_Neural_Net/fig6
permalink: /book_review/ch3_Neural_Net
category: book_review

---
## 활성화함수
활성화 함수는(activation function)은 입력 신호의 총합을 출력신호로 변환하는 함수를 의미한다.
- 계단함수

![step_function](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-16-ch3_Neural_Net/fig1.jpg?raw=true)

- 시그모이드 함수

![sigmoid](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-16-ch3_Neural_Net/fig2.jpg?raw=true)

- Relu

![Relu](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-16-ch3_Neural_Net/fig3.jpg?raw=true)

- Leaky-Relu

![Relu](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-16-ch3_Neural_Net/fig4.jpg?raw=true)

#### Code
```python
def step_function(x): # 인풋은 numpy array
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    return np.maximum(0.1*x, x)
```

#### Graph

![activation function](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-16-ch3_Neural_Net/fig5.jpg?raw=true)

### 출력층의 활성화함수
출력층의 활성화 함수는 원하는 문제의 유형에 따라서 달라진다.
-   회귀
    -   수치를 예측,
    -   출력층에 항등함수 사용
    -   ex) 점수 0~100점 예측
-   분류
    -   양자화된 데이터를 예측
    -   출력층에 소프트맥스 함수 사용
    -   ex) 점수에 따른 학점 A, B, C 예측

#### 소프트맥스 함수
소프트맥스 함수는 분류 문제에서 자주 사용하는 출력층의 활성화 함수이다.

![softmax](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-16-ch3_Neural_Net/fig7.jpg?raw=true)

```python
def softmax(a): # 인풋은 numpy array
    c = np.max(a) # 오버플로우를 막기위함
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
```


## 3층 신경망 구현
아래 그림과 같은 신경망을 구현해본다. 신경망은 여러층의 퍼셉트론으로 구성되어있다.

![3 layer nn](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-16-ch3_Neural_Net/fig6.jpg?raw=true)

### 네트워크 만들기
네트워크 클래스에는 가중치 초기화를 초기화해주고, 순방향으로 진행하는 forward 메소드만 넣어주었다. 출력 활성화함수는 항등함수이다.
```python
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = a3
    
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y) # [[0.31682708 0.69627909]]
```

## MNIST 데이터셋 예측
네트워크의 경우 책 저자가 제공하는 github에 있는 소스코드를 받아서 사용한다. 
MNIST 데이터, 즉 이미지에서 0~9까지의 숫자를 맞추는 분류 문제를 풀것이기 때문에 출력층은 소프트맥스 함수로 해준다.

```python
import sys, os
sys.path.append(os.path.join(os.getcwd(), 'deep-learning-from-scratch-master')) 
#'deep-learning-from-scratch-master' 폴더를 경로추가
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle

def img_show(img): # 데이터의 이미지를 보여준다.
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    
def get_data(): # MNIST 데이터를 받아온다.
    (x_train, t_train), (x_test, t_test) = \\
    load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network(): # 네트워크 초기화
    with open("./deep-learning-from-scratch-master/ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x): # 이미지를 넣으면 각 숫자에 대한 확률을 예측한다.
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y
```

#### 네트워크 모양
```python
network = init_network()
keys = ['W1', 'W2', 'W3', 'b1', 'b2', 'b3']
for key in keys:
    print(key, 'shape : ', network[key].shape)

#---- 출력 -----#
W1 shape :  (784, 50)
W2 shape :  (50, 100)
W3 shape :  (100, 10)
b1 shape :  (50,)
b2 shape :  (100,)
b3 shape :  (10,)
```

#### 실행
```python
x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1
        
print("Accuracy: " + str( accuracy_cnt / len(x))) # Accuracy : 0.9352
```

## 배치처리
배치는 여러 데이터를 한 번에 처리하는 것이다. 배치로 처리하면 더 빠르게 처리할 수 있다. 이것은 아래와 같은 이유이다.

-   수치 계산 라이브러리 대부분이 큰 배열을 효율적으로 처리할 수 있도록 최적화되어있다.
-   커다란 신경망에서는 데이터 전송이 병목으로 작용하는 경우가 있는데, 배치 처리가 버스에 주는 부하를 줄인다.

```python
x, t = get_data()
network = init_network()

batch_size = 100 # update
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
		# 만약 모양이 처리전에 (100, 50)이면 처리 후에는 (100,)이 됨
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
        
print("Accuracy: " + str( accuracy_cnt / len(x))) # Accuracy: 0.9352
```