---
layout: post
title: (밑바닥부터 시작하는 딥러닝2) 1장. 신경망
featured-img: 2021-11-20-ch1_Reviewing_neural_networks/fig1
permalink: /book_review/ch1_Reviewing_neural_networks
category: book_review

---


### 벡터와 행렬

```python
import numpy as np

x = np.array([1, 2, 3])
print(x.__class__) # numpy.ndarray
print(x.shape) # (3, )
print(x.ndim) # 1

```

```python
W = np.array([[1, 2, 3], [4, 5, 6]])
print(W.shape) # (2, 3)
print(W.ndim) # 2

```


### 행렬의 원소별 연산

```python
W = np.array([[1, 2, 3], [4, 5, 6]])
X = np.array([[0, 1, 2], [3, 4, 5]])
print(W + X)
# array([[ 1,  3,  5],
#       [ 7,  9, 11]])
print(W * X)
# array([[ 0,  2,  6],
#       [12, 20, 30]])

```


### 브로드캐스트

넘파이의 다차원 배열에서는 형상이 다른 배열끼리도 연산할 수 있다.

```python
A = np.array([[1, 2], [3, 4]])
print(A * 10)
# array([[10, 20],
#       [30, 40]])

```

```python
A = np.array([[1, 2], [3, 4]])
b = np.array([10, 20])
print(A * b)
# array([[10, 40],
#       [30, 80]])

```


### 벡터의 내적과 행렬의 곱

```python
# 벡터의 내적
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b)) # 32

# 행렬의 곱
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.matmul(A, B))# [[19 22] [43 50]]

```


## 신경망의 추론

여기서 구현하는 신경망은 입력층에는 뉴런 2개, 은닉층에는 4개, 출력층에 3개를 각각 준비한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig1.JPG?raw=true)

입력으로부터 은닉층으로 수행되는 계산은 아래와 같다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig2.JPG?raw=true)

전체 은닉층의 뉴런의 값은 아래와 같다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig3.JPG?raw=true)

위의 식은 아래처럼 간소화할 수 있다. x는 입력, h는 은닉층의 뉴런, W는 가중치, b는 편향을 뜻한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig4.JPG?raw=true)

이 식을 코드로 구현하면...

```python
W1 = np.random.randn(2, 4) # 가중치
b1 = np.random.randn(4) # 편향
x = np.random.randn(10, 2) # 입력, 총 10개의 데이터
h = np.matmul(x, W1) + b1

```

위의 계산은 선형계산이다. 여기에 비선형효과를 부여하는 것이 바로 활성화함수이다. 비선형효과를 통해 신경망의 표현력을 높일 수 있다. 활성화함수는 다양하지만 여기서는 시그모이드 함수를 사용한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig5.JPG?raw=true)

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
a = sigmoid(h)

```

원래 구현하려던 신경망을 정리하면 다음과 같다.

```python
x = np.random.randn(10, 2)
W1 = np.random.randn(2, 4)
b1 = np.random.randn(4)
W2 = np.random.randn(4, 3)
b2 = np.random.randn(3)

h = np.matmul(x, W1) + b1 # 은닉층
a = sigmoid(a)
s = np.matmul(a, W2) + b2 # 출력층

```


### 계층으로 클래스화 및 순전파 구현

신경망에서 하는 처리를 계층(layer)로 구현해보자. 완전연결계층을 Affine 계층으로, 시그모이드함수에 의한 변환을 Sigmoid 계층으로 구현한다. 이 계층들은 파이썬 클래스로 구현한다. 이렇게 모듈화를 해두면 레고 블록을 조합하듯 신경망을 구축할 수 있다. 모든 계층을 구현할 떄 아래의 규칙을 따른다.

-   모든 계층은 forward()와 backward() 메서드를 가진다.
-   모든 계층은 인스턴스 변수인 params와 grads를 가진다.

forward() 메서드는 순전파, backward() 메서드는 역전파를 수행한다. params는 가중치와 편향같은 매개변수를 담는 리스트이다. grads는 params에 저장된 각 매개변수에 대응하여, 해당 매개변수의 기울기를 보관하는 리스트이다.

여기서는 activation 함수인 sigmoid와 Affine layer를 구현해본다.

```python
class sigmoid:
    def __init__(self):
        self.params = []
        
    def forward(self, x):
        return 1 / (1+np.exp(-x))
    
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        
    def forward(self, x):
        W, b = self.params
        return np.matmul(x, W) + b

```

그리고 다음과 같은 신경망을 만들어본다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig6.JPG?raw=true)

```python
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        
        I, H, O = input_size, hidden_size, output_size
        
        # 가중치와 편향 초기화
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)
        
        self.layers = [Affine(W1, b1), sigmoid(), Affine(W2, b2)]
        
        self.params = [layer.params for layer in self.layers]
        
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

```

```python
x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
model.predict(x)
#---------------------출력---------------------#
array([[ 0.86840912,  3.07671248, -0.90881986],
       [ 0.91251963,  3.44243545, -1.29019432],
       [ 0.05122875,  0.31506273,  1.4837764 ],
       [ 0.14310574,  1.23632037,  0.52287386],
       [ 1.23586839,  3.50685051, -1.12750554],
       [ 0.14531436,  0.95764607,  0.83766979],
       [ 0.07668659,  0.59913049,  1.17453677],
       [ 0.68947268,  2.65659966, -0.57728626],
       [ 1.10975739,  3.64081584, -1.36670496],
       [ 1.03199471,  3.57178048, -1.3452477 ]])

```



## 신경망의 학습

좋은 추론을 위해 학습을 먼저 수행하고, 그 학습된 매개변수를 이용해 추론을 수행한다. 신경망의 학습은 최적의 매개변수 값을 찾는 작업이다.


### 손실함수

신경망 학습에는 학습이 얼마나 잘 되고 있는지 알기위한 척도로 손실(loss)을 사용한다. 손실은 학습 데이터(학습 시 주어진 정답 데이터)와 신경망이 예측한 결과를 비교하여 예측이 얼마나 나쁜가를 산출한 값이다.

신경망의 손실은 손실 함수를 사용해 구한다. 다중 클래스 분류 신경망에서는 손실 함수로 흔히 교차 엔트로피 오차를 이용한다. 교차 엔트로피 오차는 신경망이 출력하는 각 클래스의 확률과 정답 레이블을 이용해 구할 수 있다.

지금까지 다뤄 온 신경망에서 손실을 구해본다. 우선 앞 절의 신경망에 Softmax 계층과 Cross Entropy Error 계층을 추가한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig7.JPG?raw=true)

소프트맥스 함수 식으로 쓰면 다음과 같다. 출력이 총 n개일 때, k번째 출력을 구하는 계산식이다. 소프트맥스 함수의 출력의 각 원소는 0이상 1이하의 실수이다. 이 원소들을 모두 더하면 1.0이 되기 때문에 확률로 해석할 수 있는 것이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig8.JPG?raw=true)

소프트맥스의 출력인 이 확률이 교차 엔트로피 오차에 입력된다. 이 때 교차 엔트로피 오차의 수식은 다음과 같다. 여기서 tk는 k번째 클래스가 정답이면 1, 정답이 아니면 0이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig9.JPG?raw=true)

미니배치 처리를 고려하면 교차 엔트로피 오차의 식은 다음처럼 된다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig10.JPG?raw=true)

```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    if t.size == y.size:
        t = t.argmax(axis=1)
        
    batch_size = y.shape[0]
    
    cross_entropy = np.log(y[np.arange(batch_size), t] + 1e-7)
    loss = -np.sum(cross_entropy) / batch_size
    
    return loss

```


### 미분과 기울기

벡터의 각 원소에 대한 미분을 정리한 것이 기울기(gradient)이다. 행렬 W가 m * n 행렬이라면, L = g(W) 함수의 기울기는 다음과 같이 쓸 수 있다. W와 기울기의 형상이 같다는 것이 중요한 특성이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig11.JPG?raw=true)


### 연쇄 법칙

신경망의 기울기는 오차역전파법(back-propagation)을 통해 구할 수 있다. 오차역전파법을 알기 위해서는 연쇄 법칙(chain rule)을 이해해야한다. 연쇄 법칙이란 합성함수에 대한 미분 법칙이다. y = f(x)와 z = g(y)라는 두 함수가 있다. 그러면 z = g(f(x))가 되어, 최종 출력 z는 두 함수를 조합해 계산할 수 있다. 이 때 이 합성함수의 미분은 다음과 같이 구할 수 있다. x에 대한 z의 미분은 y = f(x)의 미분과 z = g(y)의 미분을 곱하면 구해진다. 이것이 연쇄법칙이다. 연쇄법칙을 통하면 다루는 함수가 아무리 복잡해도 그 미분은 개별 함수의 미분들을 통해 구할 수 있기 때문이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig12.JPG?raw=true)


### 계산 그래프

여러가지 연산에 대한 순전파와 역전파를 알아보자.

-   덧셈 노드

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig13.JPG?raw=true)


-   곱셈 노드

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig14.JPG?raw=true)


-   분기 노드

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig15.JPG?raw=true)


-   Repeat 노드
    
    ![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig16.JPG?raw=true)
    

```python
# Repeat 노드
D, N = 8, 7
x = np.random.randn(1, D) # 입력 (1, 8)
y = np.repeat(x, N, axis=0) # 순전파 (7, 8)
dy = np.random.randn(N, D) # 무작위 기울기 (7, 8)
dx = np.sum(dy, axis=0, keepdims=True) # 역전파 (1, 8)

```


-   Sum 노드

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig25.JPG?raw=true)

```python
# Sum 노드
D, N = 8, 7
x = np.random.randn(N, D) # 입력 (7, 8)
y = np.sum(x, axis=0, keepdims=True) # 순전파 (1, 8)
dy = np.random.randn(1, D) # 무작위 기울기 (1, 8)
dx = np.repeat(dy, N, axis=0) # 역전파 (7, 8)

```


-   Matmul 노드

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig17.JPG?raw=true)

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig18.JPG?raw=true)

```python
# Matmul 노드
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
        
    def forward(self, x):
        W,  = self.params
        out = np.matmul(x, W)
        self.x = x
        return out
    
    def backward(self, dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW
        return dx

```



### 기울기 도출과 역전파 구현


-   sigmoid 계층

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig19.JPG?raw=true)

```python
class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None
        
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

```


-   Affine 계층

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig20.JPG?raw=true)

```python
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
        
    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        self.x = x
        return out
    
    def backward(self, dout):
        W, b = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis=0)
        
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

```


-   Softmax with Loss 계층

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig21.JPG?raw=true)

```python
def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None # softmax의 출력
        self.t = None # 정답 레이블
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        
        if self.t.size == self.y.size: # 정답 레이블이 원핫으로 표현된경우
            self.t = self.t.argmax(axis=1) # t를 답의 인덱스로 변경
            
        loss = cross_entropy_error(self.y, self.t)
        return loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx /= batch_size
        
        return dx

```


### 가중치 갱신

학습을 할 때 먼저 오차역전파법으로 가중치의 기울기를 얻는다. 이 기울기는 현재의 가중치 매개변수에서 손실을 가장 크게 하는 방향을 가리킨다. 따라서 매개변수를 그 기울기와 반대 방향으로 갱신하면 손실을 줄일 수 있다. 이것이 바로 경사하강법이다. 가중치 갱신 기법의 종류는 아주 다양한데, 여기서는 확률적경사하강법(SGD)을 구현한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig22.JPG?raw=true)

```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

```



## 신경망으로 문제 풀기

### 스파이럴 데이터셋

스파이럴 데이터셋은 아래와 같은 데이터셋이다. x는 입력데이터, t는 정답레이블이다. x, t는 각각 300개의 샘플 데이터를 담고 있으며, x는 2차원 데이터이고, t는 3차원데이터이다. 즉 t는 원핫 벡터로, 정답에 해당하는 클래스에는 1이, 그 외에는 0이 레이블되어 있다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig23.JPG?raw=true)

```python
def load_data(seed=1984):
    np.random.seed(seed)
    N = 100 # 클래스당 샘플 수
    DIM = 2 # 데이터 요소 수
    CLS_NUM = 3 # 클래스 수
    
    x = np.zeros((N*CLS_NUM, DIM))
    t = np.zeros((N*CLS_NUM, CLS_NUM), dtype=np.int)
    
    for j in range(CLS_NUM):
        for i in range(N):
            rate = i / N
            radius = 1.0*rate
            theta = j*4.0 + 4.0*rate + np.random.randn()*0.2
            
            ix = N*j + i
            x[ix] = np.array([radius*np.sin(theta), radius*np.cos(theta)]).flatten()
            t[ix, j] = 1
            
    return x, t

x, t = load_data()
print('x', x.shape) # (300, 2)
print('t', t.shape) # (300, 3)

```


### 신경망 구현

학습을 시킬 모델을 구현해보자.

```python
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O  = input_size, hidden_size, output_size
        
        # 가중치와 편향 초기화
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)
        
        # 계층 생성
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss()
        
        # 모든 가중치와 기울기를 리스트에 모은다.
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
            
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

```


### 학습용 코드

실제로 학습을 진행해보자.

```python
# 하이퍼파라미터 설정
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# 데이터 읽기, 모델과 옵티마이저 생성
x, t = load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer =SGD(lr=learning_rate)

# 학습에 사용하는 변수
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    # 데이터 뒤섞기
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]
    
    for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]
        
        # 기울기를 구해 매개변수 갱신
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        
        total_loss += loss
        loss_count += 1
        
        # 정기적으로 학습 경과 출력
        if (iters+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print('| 에폭 %d | 반복 %d / %d | 손실 %.2f' % (epoch + 1, iters + 1, max_iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0
#---------------------출력---------------------#
| 에폭 1 | 반복 10 / 10 | 손실 1.13
| 에폭 2 | 반복 10 / 10 | 손실 1.13
| 에폭 3 | 반복 10 / 10 | 손실 1.12
| 에폭 4 | 반복 10 / 10 | 손실 1.12
...
| 에폭 298 | 반복 10 / 10 | 손실 0.11
| 에폭 299 | 반복 10 / 10 | 손실 0.11
| 에폭 300 | 반복 10 / 10 | 손실 0.11

```

학습결과를 그래프로 그려보자.

```python
import matplotlib.pyplot as plt

x = np.arange(len(loss_list))
plt.plot(x, loss_list, label='train')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig24.JPG?raw=true)

손실이 에폭이 지날수록 감소하는 것을 확인할 수 있다.


### Trainer 클래스

Trainer 클래스는 앞에서 본 학습을 하는 코드를 행해주는 클래스이다. fit() 메서드를 호출해 학습을 시작한다. fit() 메서드가 받는 인수는 아래와 같다.

-   x : 입력데이터
-   t : 정답 레이블
-   max_epoch : 학습을 수행하는 에폭 수
-   batch_size : 미니배치 크기
-   eval_interval : 결과를 출력하는 간격
-   max_grad : 기울기 최대 노름

```python
import time

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0
        
    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0
        
        start_time = time.time()
        for epoch in range(max_epoch):
            # 뒤섞기
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]
            
            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]
                
                # 기울기를 구해 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = model.params, model.grads
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1
                
                # 평가
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 손실 %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1
        
    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('Iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('Loss')
        plt.show()

```

Trainer를 통해 학습을 다시 시켜보자.

```python
# 하이퍼파라미터 설정
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# 데이터 읽기, 모델과 옵티마이저 생성
x, t = load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer =SGD(lr=learning_rate)

trainer = Trainer(model, optimizer)
trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
trainer.plot()
#---------------------출력---------------------#
| 에폭 1 |  반복 1 / 10 | 시간 0[s] | 손실 1.10
| 에폭 2 |  반복 1 / 10 | 시간 0[s] | 손실 1.12
| 에폭 3 |  반복 1 / 10 | 시간 0[s] | 손실 1.13
...
| 에폭 298 |  반복 1 / 10 | 시간 0[s] | 손실 0.11
| 에폭 299 |  반복 1 / 10 | 시간 0[s] | 손실 0.11
| 에폭 300 |  반복 1 / 10 | 시간 0[s] | 손실 0.11

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-20-ch1_Reviewing_neural_networks/fig24.JPG?raw=true)



## 계산 고속화

신경망의 학습과 추론에 드는 연산량은 상당하다. 빠르게 계산하기 위해 비트정밀도와 GPU에 대해 알아본다.


### 비트 정밀도

넘파이의 부동소수점 수는 기본적으로 64비트 데이터 타입을 사용한다. 실제로 64비트 부동소수점 수가 사용되는지는 다음 코드로 확인할 수 있다.

```python
a = np.random.randn(3)
print(a.dtype) # dtype('float64')

```

신경망의 추론과 학습은 32비트로도 충분히 수행가능하다. 또 신경망 계산시 데이터를 전송하는 버스 대역폭이 병목이 되는 경우가 있다. 이런 경우에도 데이터 타입이 작은게 유리하다. 계산속도 측면에서도 32비트 부동소수점 수가 더 빠르다. 비트수를 변경하기 위해서는 넘파이의 astype을 사용하면 된다.

```python
b = np.random.randn(3).astype(np.float32)
print(b.dtype) # dtype('float32')

```

```python
c = np.random.randn(3).astype('f')
print(c.dtype) # dtype('float32')

```


### GPU(쿠파이)

딥러닝의 계산은 대량의 곱하기 연산으로 구성된다. 이 대량의 곱하기 연산 대부분은 병렬로 계산할 수 있는데, 이 때 CPU보다 GPU가 더 유리하다. 쿠파이(cupy)는 gpu를 이용해 병렬계산을 수행해주는 라이브러리이다. 컴퓨터에 쿠다를 설치하고, 그 쿠다 버전에 맞는 쿠파이를 설치해주면 된다.

```python
import cupy as cp
x = cp.arange(6).reshape(2, 3).astype('f')

```