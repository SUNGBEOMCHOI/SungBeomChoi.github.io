---
layout: post
title: (밑바닥부터 시작하는 딥러닝) 6장. 학습 관련 기술들
featured-img: 2021-02-07-ch6_Update_Parameter/fig3
permalink: /book_review/ch6_Update_Parameter
category: book_review

---

## 매개변수 갱신
신경망 학습의 목적은 손실 함수의 값을 가능한 낮추는 매개변수를 찾는 것이고, 이를 최적화(optimization)이라고 한다.

### 확률적 경사 하강법(SGD)

5장에서 매개변수의 기울기를 구해, 기울어진 방향으로 매개변수 값을 갱신하는 일을 몇 번이고 반복해서 점점 최적의 값으로 다가갔다. 이는 SGD라는 방법이다.

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig1.jpg?raw=true)

- W : 갱신할 가중치 매개변수 
- η : 학습률

```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

```

<br>
아래와 같이 사용할 수 있다.

```python
network = TwoLayerNet(...)
optimizer = SGD()

for i in range(10000):
    ...
    x_batch, t_batch = get_mini_batch(...)
    grads = network.gradient(x_batch, t_batch)
    params = network.params
    optimizer.update(params, grads)
    ...

```
<br>

#### SGD의 단점

다음과 같은 함수의 최솟값을 구하는 문제를 생각해보자.

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig2.jpg?raw=true)

함수의 기울기를 생각해보면 y축 방향은 크지만 , x축 방향은 작다. 학습 그래프를 그려보면 다음과 같다.

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig3.jpg?raw=true)

즉 SGD의 단점은 예시로 들은 함수같은 비등방성 함수에서는 탐색 경로가 비효율적이라는 것이다. 따라서 SGD의 단점을 해결해주는 모멘텀, AdaGrad, Adam과 같은 방법이 새로 고안되었다.

<br>

### 모멘텀

모멘텀은 관성을 가지고 있어서 이전 기울기가 진행하던 방향으로 진행하려는 성질을 가진다.

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig4.jpg?raw=true)

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig5.jpg?raw=true)

```python
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

```

SGD에서 예를 들은 함수의 모멘텀 학습 그래프는 다음과 같다.

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig6.jpg?raw=true)

SGD보다 x축 방향으로 빠르게 다가간다. 그러나 동일하게 y축 방향의 속도는 안정적이지 않다.

<br>

### AdaGrad

학습에서 학습률의 값이 너무 작으면 학습 시간이 너무 길어지고, 반대로 너무 크면 발산하여 학습이 제대로 이뤄지지 않는다.

이 학습률을 정하는 기술로 학습률 감소(learning rate decay)가 있다. 이는 학습을 진행하면서 학습률을 점차 줄여가는 방법이다.

AdaGrad에서는 각각의 매개변수에 맞춤형 학습률을 만들어준다.

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig7.jpg?raw=true)

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig8.jpg?raw=true)

```python
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

```

Adagrad 학습 그래프는 다음과 같다.

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig9.jpg?raw=true)

최솟값을 향해 효율적으로 움직인다. y축 방향은 기울기가 커서 처음에는 크게 움직이지만, 큰 움직임에 비례해 갱신 정도도 큰 폭으로 작아진다.

<br>

### Adam

모멘텀은 부드러운 움직임을 보였고, AdaGrad는 효율적인 움직임을 보여주었다. 이 둘을 합친것이 Adam이다.

```python
class Adam:
		def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

```

Adam 학습 그래프는 다음과 같다.

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig10.jpg?raw=true)

<br>
<br>

## 가중치의 초깃값

가중치 감소(weight decay) : 가중치 매개변수의 값이 작아지도록 학습하는 방법, 오버피팅 억제에 효과적이다.

가중치를 적게 만들기 위해서는 초깃값도 작은 것이 좋다.

그럼 초깃값을 모두 0으로 설정하면 → 모든 가중치의 값이 똑같이 갱신되어 학습이 제대로 이루어지지 않는다. 따라서 초깃값을 무작위로 설정해야 한다.

가중치의 초깃값에 따라 은닉층 활성화값들이 어떻게 변화하는지 살펴보자.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): # 활성화 함수
    return 1 / (1+np.exp(-x))

x = np.random.randn(1000, 100) # 1000개의 데이터
node_num = 100 # 각 은닉층의 노드수
hidden_layer_size = 5 # 은닉층이 5개
activations = {} # 활성화값을 저장

# forward를 진행
for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1] #이전 layer에서의 활성화값을 가져옴
        
    w = np.random.randn(node_num, node_num) * 1 # 표준편차가 1인 shape 100*100의 행렬을 생성
    a = np.dot(x, w)
    z = sigmoid(a)
    
    activations[i] = z

# 히스토그램 그리기
fig = plt.figure(figsize = (25, 5))
for i, a in activations.items():
    ax = fig.add_subplot(1, len(activations), i+1)
    ax.set_title(str(i+1) + '-layer')
    ax.hist(a.flatten(), 30, range=(0, 1))
plt.show()

```
<br>

### 가중치를 표준편차가 1인정규분포로 초기화한 경우
    
데이터가 0과 1에 치우쳐 분포하며, 이렇게 되면 역전파의 기울기 값이 점점 작아지다가 사라진다. 이것이 기울기 소실(gradient vanishing) 문제이다.

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig11.jpg?raw=true)

  <br>
    
### 가중치를 표준편차가 0.01인 정규분포로 초기화한 경우
활성값들이 0.5 정도로 치우침. 다수의 뉴런이 거의 같은 값을 출력하기 때문에 뉴런을 여러개 둔 의미가 없어진다.    

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig12.jpg?raw=true)

<br>

###  Xavier 초깃값

앞 계층의 노드가 n개라면 표준편차가 sqrt(1/n)인 분포를 사용한다.

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig13.jpg?raw=true)

<br>

###  He 초깃값

ReLU를 사용할 때는 He 초깃값을 사용하는 것이 좋다. 앞 계층의 노드가 n개라면 표준편차가 sqrt(2/n)인 분포를 사용한다.

<br>

### MNIST 데이터셋으로 본 가중치 초깃값 비교

표준편차가 0.01인 경우는 거의 학습이 되지 않았고, He와 Xavier는 학습이 되었다. 이처럼 초깃값을 설정하는 것은 매우 중요하다.

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig14.jpg?raw=true)



## 배치 정규화

가중치의 초깃값을 적절히 설정하면 각 층의 활성화값 분포가 적당히 퍼지면서 학습이 원할하게 수행된다. 이런 생각으로 각 층의 활성화를 고의로 퍼뜨리도록 하는 것이 배치 정규화(batch normalization)이다.

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig15.jpg?raw=true)

<br>

### 배치 정규화의 장점
-   학습 속도 개선
-   초깃값에 크게 의존하지 않는다.
-   오버피팅을 억제한다.

<br>

### 배치 정규화의 효과
학습 속도를 높인다. 그리고 가중치 초깃값에 크게 의존하지 않을 수 있다.

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig16.jpg?raw=true)

<br>
<br>

## 바른 학습을 위해

### 오버피팅

오버피팅은 신경망이 훈련데이터에만 지나치게 적응되어 그 외의 데이터는 제대로 대응 못하는 상태를 말한다.

오버피팅은 주로 다음의 두 경우에 일어난다.

-   매개변수가 많고 표현력이 높은 모델
-   훈련 데이터가 적음
<br>
#### 오버피팅의 예시

```python
import os
import sys
sys.path.append('./deep-learning-from-scratch-master')
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

#오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10)
optimizer = SGD(lr=0.01)

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(100000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

```

```python
# 그래프 그리기
markers = {'train' : 'o', 'test' : 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='o', label='test', markevery=10)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.show()

```

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig20.jpg?raw=true)

그래프에서 보는것처럼 train_set의 정확도는 거의 100퍼센트지만 test_set의 정확도는 그 보다 한 참 못미친다. 이것이 오버피팅이다.

<br>

### 가중치 감소(weight decay)

오버피팅 억제용으로 가중치 감소를 사용하기도 한다. 오버피팅은 가중치 매개변수의 값이 커서 발생하는 경우가 많다. 따라서 가중치를 일부러 작게 학습되도록 유도하는데 이러한 방법이 가중치 감소이다.

가중치 감소에서는 모든 가중치 각각의 손실 함수에 가중치의 제곱에 비례하는 값을 더해준다. 즉 가중치가 크면 페널티를 받는 것이다.

아래 그래프를 보면 가중치 감소를 사용했을 때가 쓰지 않았을 때보다 train과 test 사이의 정확도의 차이가 조금은 줄어든다.

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig17.jpg?raw=true)

<br>

### 드롭아웃

드롭아웃은 뉴런을 임의로 삭제하면서 학습하는 방법이다. 훈련 때 은닉층의 뉴런을 무작위로 골라 삭제하고, 시험 때는 모든 뉴런에 신호를 전달한다.

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig18.jpg?raw=true)

```python
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        
    def forward(self, x, train_flg = True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio # dropout_ratio보다 큰 값만 True이다.
            return x * self.mask # 출력층의 일부가 꺼진다.
        else:
            return x * (1.0 - self.dropout_ratio) # 시험때는 dropout 비율만큼 뺀값을 출력값에 곱해준다.

```

왼쪽은 드롭아웃 없이, 오른쪽은 드롭아웃을 적용한 결과이다. <br>
확실히 학습셋과 테스트셋의 정확도 차이가 줄었고, 학습셋의 정확도는 100%가 아니다.

![SGD](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-07-ch6_Update_Parameter/fig19.jpg?raw=true)

<br>
<br>

### 적절한 하이퍼파라미터 값 찾기

하이퍼파라미터를 최적화할 때 대략적인 범위를 설정하고, 그 범위에서 무작위로 하이퍼파라미터 값을 샘플링 후, 훈련시켜 정확도를 평가한다.

#### 검증데이터
학습 데이터, 시험 데이터에 추가로 검증 데이터를 만든다. 이 검증 데이터는 적절한 하이퍼파라미터를 찾는데 사용하게 된다.