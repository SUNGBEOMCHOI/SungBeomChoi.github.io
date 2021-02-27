---
layout: post
title: (밑바닥부터 시작하는 딥러닝) 5장. 오차역전파법
featured-img: 2021-01-30-ch5_Backpropagation/fig3
permalink: /book_review/ch5_Backpropagation
category: book_review

---

수치 미분은 간단하지만 계산 시간이 오래 걸린다는 단점이 있다. 이러한 단점을 오차역전파법(backpropagation)을 통해 해결할 수 있었다.

## 계산 그래프

- 노드: 연산을 표현
- 에지 : 입력과 출력되는 데이터를 표현

사과에서 가격까지가는 방향으로 퍼져가는 것을 순전파 반대 방향으로 퍼지는 것을 역전파라고 합니다.  

![perceptron](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-30-ch5_Backpropagation/fig1.jpg?raw=true)

<br>
<br>

### 역전파
역전파는 output에서 input 방향으로 진행한다. 역전파로 갈 때는 미분값을 활용한다.

![backpropagation](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-30-ch5_Backpropagation/fig2.jpg?raw=true)


<br>
<br>

### 계산그래프의 연산 구현

#### 덧셈 계층
![AddLayer](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-30-ch5_Backpropagation/fig3.jpg?raw=true)
```python
class AddLayer:
    def __init__(self):
        pass
    
    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

```

<br>

#### 곱셈 계층

![MulLayer](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-30-ch5_Backpropagation/fig4.jpg?raw=true)

```python
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

```

<br>
<br>

### 계산그래프의 순전파와 역전파의 구현

![apple problem](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-30-ch5_Backpropagation/fig5.jpg?raw=true)

```python
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# 계층들
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# 순정파
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price= mul_tax_layer.forward(all_price, tax)

# 역전파
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(price) #715
print(dapple_num, dapple, dorange_num, dtax) #110, 2.2, 165, 650

```
<br>
<br>

### 활성함 함수 계층 구현

#### Relu 계층

- 순전파

![relu forward](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-30-ch5_Backpropagation/fig6.jpg?raw=true)

- 역전파

![relu backward](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-30-ch5_Backpropagation/fig7.jpg?raw=true)

```python
class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0) # x 배열에서 0보다 작으면 True, 크면 False인 배열을 반환
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout        
        return dx

```

<br>

#### sigmoid 계층

- 순전파

![softmax forward](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-30-ch5_Backpropagation/fig8.jpg?raw=true)

- 역전파

![softmax backward](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-30-ch5_Backpropagation/fig9.jpg?raw=true)

```python
class Sigmoid:
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out        
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

```
<br>
<br>

### Affine/Softmax-with-Loss 계층 구현

#### Affine 계층

![Affine](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-30-ch5_Backpropagation/fig10.jpg?raw=true)

```python
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        return dx

```

<br>

#### Softmax-with-Loss 계층

![softmax-with-loss](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-30-ch5_Backpropagation/fig11.jpg?raw=true)

```python
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # loss
        self.y = None # softmax의 출력
        self.t = None # 정답 레이블(원-핫 벡터)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

```
<br>
<br>

## 오차역전파법을 적용한 신경망 구현

2층의 레이어를 가지는 네트워크를 만들어본다. Gradient 메소드에서는 backpropagation을 통해서 기울기값을 반환한다.

```python
import sys, os
sys.path.append('./deep-learning-from-scratch-master')
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        #계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    # x : 입력데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: # 1이면 one-hot-lable이 아니고, 1이 아니면 one-hot-lable
            t = np.argmax(t, axis = 1)
            
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
    
    def gradient(self, x, t):
        # 순전파
        self.loss(x, t) # 이 함수를 실행함으로서 self.lastLayer의 forward를 실행함.
        
        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values()) #layers라는 oredered dict의 value값을 뽑음
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        #결과 저장
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        
        return grads

```
<br>
<br>

### 수치미분으로 구한 기울기와 역전파법으로 구한 기울기 비교

아래코드에서는 수치 미분으로 구한 기울기와 역전파법으로 구한 기울기의 차이가 얼마 안나는것을 보인다. 

```python
import sys, os
sys.path.append('./deep-learning-from-scratch-master')
import numpy as np
from dataset.mnist import load_mnist
from ch05.two_layer_net import TwoLayerNet

# 데이터를 불러옴
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

# 네트워크 선언 및 초기화
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 미니배치로 0, 1, 2의 데이터만 불러옴
x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch) #수치미분에 의한 기울기
grad_backprop = network.gradient(x_batch, t_batch) # 역전파에 의한 기울기

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key]-grad_numerical[key]))
    print(key+":"+str(diff))

#----------output-----------#
# 각 방법으로 구한 기울기의 차를 출력
# W1:3.771145507353469e-10
# b1:2.3007621810804038e-09
# W2:4.787288764258565e-09
# b2:1.3988587680979768e-07

```
<br>
<br>

## 오차역전파법을 사용한 학습 구현
Mnist데이터를 예측하고, 파라미터를 backpropagation을 통해 파라미터를 학습해본다.

```python
import sys, os
sys.path.append('./deep-learning-from-scratch-master')
import numpy as np
from dataset.mnist import load_mnist
from ch05.two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size/batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #오차역전파법으로 기울기를 구한다.
    grad = network.gradient(x_batch, t_batch)
    
    #갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        train_acc_list.append(test_acc)
        print(train_acc, test_acc)

#------output--------#
# 0.11371666666666666 0.1118
# 0.90215 0.9053
# 0.9221 0.923
# 0.9343333333333333 0.9355
# 0.94425 0.9428
# 0.94875 0.9458
# 0.9555166666666667 0.9539
# 0.9606333333333333 0.9563
# 0.96475 0.9595
# 0.9670333333333333 0.963
# 0.9696666666666667 0.9667
# 0.97235 0.9658
# 0.97535 0.968
# 0.9762166666666666 0.9683
# 0.97735 0.9699
# 0.9786 0.9686
# 0.9797166666666667 0.9702
# 최종 정확도는 train_set은 98%, test_set은 97% 나왔다.

```