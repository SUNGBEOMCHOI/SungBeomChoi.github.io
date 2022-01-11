---
layout: post
title: (밑바닥부터 시작하는 딥러닝2) 6장. 게이트가 추가된 RNN
featured-img: 2022-01-11-ch6_Gated_RNN/fig17
permalink: /book_review/2022-01-11-ch6_Gated_RNN
category: book_review
use_math: true

---

간단한 RNN은 안타깝게도 성능이 좋지 못하다. 그 원인은 시계열 데이터에서 시간적으로 멀리 떨어진, 장기 의존 관계를 잘 학습할 수 없다는데 있다. 요즘에는 단순한 RNN대신 LSTM이나 GRU라는 계층이 주로 쓰인다. LSTM이나 GRU에는 게이트라는 구조가 더해져 있는데, 이 게이트 덕분에 장기 의존 관계를 학습할 수 있다. 이번 장에서는 기본 RNN의 문제점을 알아보고, 이를 대신하는 계층으로써 LSTM과 GRU를 알아본다.

## RNN의 문제점

### RNN 복습

RNN 계층은 순환 경로를 갖고 있다. 그리고 그 순환을 펼치면 아래 그림과 같이 옆으로 길게 뻗은 신경망이 만들어진다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig1.png?raw=true)

RNN 계층은 시계열 데이터인 𝑥𝑡xt를 입력하면 ℎ𝑡ht를 출력한다. 이 ℎ𝑡ht는 은닉 상태라고 하여, 과거 정보를 저장한다. RNN 계층이 수행하는 처리를 계산 그래프로 나타내면 아래와 같다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig2.png?raw=true)

### 기울기 소실 또는 기울기 폭발

언어 모델은 주어진 단어들을 기초로 다음에 출현할 단어를 예측하는 일을 한다. [5장](https://sungbeomchoi.github.io/book_review/2021-11-30-ch5_RNN) 에서는 RNN을 사용해 언어모델을 구현했다. 그 때의 문제를 다시 생각해보자.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig3.png?raw=true)

"?"에 들어갈 단어는 "Tom"이다. 이 문제에 올바르게 답하려면 "Tom이 방에서 TV를 보고 있음"과 "그 방에 Mary가 들어옴"이란 정보를 은닉 상태에 인코딩해 보관해둬야한다. 이 예를 RNNLM 학습의 관점에서 생각해보자. "Tom"이라는 정답 레이블이 주어졌을 때 과거의 방향으로 기울기가 전달되는 과정을 살펴보자.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig4.png?raw=true)

기울기는 학습해야 할 의미가 있는 정보가 들어 있고, 그것을 과거로 전달함으로써 장기 의존 관계를 학습한다. 만약 이 기울기가 중간에 사그라들면 가중치 매개변수는 전혀 갱신되지 않게 된다. 현재의 단순한 RNN 계층에서는 시간을 거슬러 올라갈수록 기울기가 작아지거나 혹은 커질 수 있다.

### 기울기 소실과 기울기 폭발의 원인

기울기 소실이 일어나는 원인을 살펴보자. 아래 그림에서 시간 방향 기울기 전파에만 주목해보자.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig5.png?raw=true)

기울기는 차례로 'tanh', '+', 'MatMul'연산을 통과한다. '+'의 역전파는 상류에서 전해지는 기울기를 그대로 하류로 흘려보낸다. 그래서 기울기는 변하지 않는다. ${y=tanh(x)}$일 떄의 미분은 $\frac {\partial y}{\partial x} = 1 - y^2$ 이다. 미분값은 1.0이하이고, x가 0르오부터 멀어질수록 작아진다. 즉 역전파에서는 기울기가 tanh 노드를 지날 때마다 값은 계속 작아진다는 뜻이다.

다음은 MatMul의 역전파도 살펴보자. 아래 그림에서는 상류로부터 ${dh}$라는 기울기가 흘러온다고 가정한다. 이때 MatMul 노드에서의 역전파는 ${dh{W_h}^T}$라는 행렬곱으로 기울기를 계산한다. 여기서 주목할 점은 이 행렬 곱셈에서는 매번 똑같은 가중치인 ${W_h}$가 사용된다는 것이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig6.png?raw=true)

코드를 통해 MatMul 노드를 지날때마다 기울기가 어떻게 변화하는지 살펴보자.
```python
import numpy as np
import matplotlib.pyplot as plt

N = 2 # 미니배치 크기
H = 3 # 은닉 상태 벡터의 차원 수
T = 20 # 시계열 데이터의 길이

dh = np.ones((N, H))
np.random.seed(3)
Wh = np.random.randn(H, H)

norm_list = []
for t in range(T):
    dh = np.matmul(dh, Wh.T)
    norm = np.sqrt(np.sum(dh**2)) / N # L2 노름
    norm_list.append(norm)
    
plt.plot(np.arange(len(norm_list)), norm_list)
plt.xlabel('time step')
plt.ylabel('norm')
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig7.png?raw=true)

이번에는 기울기가 지수적으로 감소한다. 이것이 기울기 소실이다. 기울기가 일정 수준 이하로 작아지면 가중치 매개변수가 더 이상 갱신되지 않으므로, 장기 의존 관계를 학습할 수 없다.

이렇게 기울기가 지수적으로 변하는 이유는 행렬 Wh를 T번 반복해서 곱했기 때문이다. 만약 Wh가 스칼라라면 1보다 큰 경우에는 지수적으로 증가하고, 1보다 작은 경우에는 기울기가 지수적으로 작아진다. Wh는 행렬인 경우에는 특잇값이 척도가 된다. 특잇값이 1보다 크면 지수적으로 증가할 가능성이 높고, 1보다 작으면 지수적으로 감소할 가능성이 높다.

### 기울기 폭발 대책

기울기 폭발의 대책을 알아보자. 전통적인 기법으로는 "기울기 클리핑"이 있다. 의사코드는 아래와 같다. 단순한 알고리즘이지만, 많은 경우에 잘 작동한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig8.png?raw=true)

이를 파이썬으로 구현해보자.

```python
dW1 = np.random.rand(3, 3) * 10
dW2 = np.random.rand(3, 3) * 10
grads = [dW1, dW2]
max_norm = 5.0

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate

```

## 기울기 소실과 LSTM

기울기 소실 문제를 해결하려면 RNN 계층의 아키텍처를 근본부터 뜯어고쳐야한다. 여기서 등장하는 것이 게이트가 추가된 RNN이다. 대표적으로는 LSTM과 GRU가 있다.

### LSTM의 인터페이스

먼저 LSTM의 입출력을 RNN과 비교해보자.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig9.png?raw=true)

LSTM에는 c라는 기억셀이라는 LSTM 전용의 기억 메커니즘이 있다. 기억셀의 특징은 LSTM 계층내에서만 주고받는다는 것이다. 즉 LSTM 계층 내에서만 완결되고, 다른 계층으로는 출력하지 않는다.

### LSTM 계층 조립하기

기억셀 ${c_t}$에는 과거로부터 시각 t까지에 필요한 모든 정보가 저장돼 있다고 가정한다. 이 기억을 바탕으로 외부 계층에 은닉 상태 ${h_t}$를 출력한다. 이 때 출력하는 ${h_t}$는 기억셀의 값을 tanh 함수로 변환한 값이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig10.png?raw=true)

LSTM을 잘 이해하기 위해서는 게이트라는 기능에 대해 알아야한다. 게이트는 데이터의 흐름을 제어한다. 게이트는 열기, 닫기 뿐 아니라, 어느 정도 열지를 조절할 수 있다. 이 열림상태는 0~1 사이의 실수로 나타난다. 중요한 것은 게이터를 얼마나 열까라는 것도 데이터로부터 학습한다는 점이다.

### output 게이트

LSTM에서는 ${tanh({c_t})}$에 게이트를 적용하는데, 이 게이트는 다음 은닉 상태 ${h_t}$의 출력을 담당하는 게이트이므로 output 게이트라고 한다. output 게이트의 열림 상태는 입력 ${x_t}$와 이전 상태 ${h_{t-1}}$로부터 아래와 같이 구한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig12.png?raw=true)

이 ${o}$와 ${tanh({c_t})}$의 원소별 곱을 ${h_t}$로 출력한다. 이를 아래와 같이 표현한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig13.png?raw=true)

이 과정을 계산그래프로 그려보면 아래와 같다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig14.png?raw=true)

### forget 게이트

forget 게이트는 기억 셀에 무엇을 잊을까를 지시한다. forget 게이트는 아래와 같이 계산한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig15.png?raw=true)

이 ${f}$와 이전 기억 셀인 ${c_{t-1}}$과의 원소별 곱, 즉 ${c_t = f \odot c_{t-1}}$을 계산한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig16.png?raw=true)

### 새로운 기억셀

새로 기억해야 할 정보를 기억셀에 추가해야한다. 계산그래프로는 아래와 같다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig17.png?raw=true)

tanh 노드는 게이트가 아니며 새로운 정보를 기억셀에 추가하는 것이 목적이다. 따라서 활성화 함수로는 시그모이드 함수가 아닌 tanh 함수가 사용된다. tanh 노드에서 수행하는 계산은 다음과 같다. 이 ${g}$가 이전 시각의 기억 셀인 ${c_{t-1}}$에 더해짐으로써 새로운 기억이 생겨난다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig29.png?raw=true)

### input 게이트

${g}$에 게이트를 추가한다. 이 게이트를 input 게이트라고 한다. input 게이트를 추가하면 계산 그래프가 아래와 같다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig30.png?raw=true)

input 게이트는 ${g}$의 각 원소가 새로 추가되는 정보로써의 가치가 얼마나 큰지를 판단한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig18.png?raw=true)

그 다음 ${i}$와 ${g}$의 원소별 곱 결과를 기억셀에 추가한다.

### LSTM의 기울기 흐름

어떻게 이 구조는 기울기 소실을 없애는 것일까? 그 원리는 기억 셀 c의 역전파에 주목하면 알 수 있다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig19.png?raw=true)

기억 셀의 역전파에서는 '+'와 '${\times}$' 노드만을 지나게 된다. '+' 노드는 상류에서 전해지는 기울기를 그대로 흘릴 뿐이다. '${\times}$'노드는 행렬 곱이 아닌 원소별 곱을 계산한다. 여기서는 매 시각 다른 게이트 값을 이용해 원소별 곱을 계산하기 때문에 곱셈의 효과가 누적되지 않아 기울기 소실이 일어나지 않는 것이다.

## LSTM 구현

LSTM에서 수행하는 계산은 아래처럼 정리할 수 있다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig20.png?raw=true)

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig21.png?raw=true)

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig22.png?raw=true)

위의 네 수식에서는 affine 변환을 개별적으로 수행하지만 이를 하나의 식으로 정리해 계산할 수 있다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig23.png?raw=true)

위의 계산들을 종합하여 계산 그래프를 만들어보면 아래와 같다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig24.png?raw=true)

```python
class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
        
    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape
        
        A = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b
        
        # slice
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]
        
        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)
        
        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)
        
        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

```

### Time LSTM 구현

Time LSTM은 T개분의 시계열 데이터를 한꺼번에 처리하는 계층이다. 그 전체그림은 아래처럼 구성된다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig25.png?raw=true)

```python
class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful
        
    def forward(self, x):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]
        
        self.layers = []
        hs = np.empty((N, T, H), dtype='f')
        
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')
            
        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h
            
            self.layers.append(layer)
            
        return hs
    
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]
        
        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0
        
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
                
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
            
        self.dh = dh
        return dxs
    
    def set_state(self, h, c=None):
        self.h, self.c = h, c
        
    def reset_state(self):
        self.h, self.c = None, None

```

## LSTM을 사용한 언어 모델

아래 그림과 같은 언어모델을 만든다. [5장](https://sungbeomchoi.github.io/book_review/2021-11-30-ch5_RNN)에서 만든 RNN 언어모델과 다른점은 Time RNN이 Time LSTM으로 바뀐것 뿐이다.

```python
import sys
from common.time_layers import *
import pickle

class Rnnlm:
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        # 가중치 초기화
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        
        # 계층 생성
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]
        
        
        # 모든 가중치와 기울기를 리스트에 모은다.
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
            
    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        self.lstm_layer.reset_state()
        
    def save_params(self, file_name='Rnnlm.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)
            
    def load_params(self, file_name='Rnnlm.pkl'):
        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)

```

위의 신경망을 사용해 PTB 데이터셋을 학습해보자.

```python
import sys
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from dataset import ptb
from rnnlm import Rnnlm

# 하이퍼파라미터 설정
batch_size = 20
wordvec_size = 100
hidden_size = 100 # RNN의 은닉 상태 벡터의 원소 수
time_size = 35 # RNN을 펼치는 크기
lr = 20.0
max_epoch = 4
max_grad = 0.25

# 학습 데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_text, _, _ = ptb.load_data('test')
vocab_size = len(word_to_od)
xs = corpus[:-1]
ts = corpus[1:]

# 모델 생성
model = Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# 기울기 클리핑을 적용하여 학습
trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
trainer.plot(ylim=(0, 500))

# 테스트 데이터로 평가
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('테스트 퍼플렉서티: ', ppl_test)

# 매개변수 저장
model.save_params()

```

위의 학습결과(퍼플렉서티)를 살펴보자.퍼플렉서티가 최종적으로는 136정도가 되었다. 즉, 이 모델은 다음에 나올 단어의 후보를 총 10000개 중에서 136개 정도로 줄일 때까지 개선된 것이다. 이 값은 그다지 좋은 결과는 아니다. 조금 더 개선하기위해서는 어떻게 해야할까?

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig26.png?raw=true)

## RNNLM 추가 개선

RNNLM의 개선 포인트 3가지를 살펴보자.

### LSTM 계층 다층화

LSTM 계층을 깊게 쌓아 효과를 볼 수 있다.계층을 더 깊게 쌓음으로서 더 복잡한 패턴을 학습할 수 있다. 쌓는 층 수는 하이퍼파라미터이므로 처리할 문제의 복잡도나 준비된 학습 데이터의 양에 따라 적절하게 결정해야한다. PTB 데이터셋의 언어 모델에서는 LSTM의 층 수는 2~4 정도일 때 좋은 결과를 얻는 것으로 알려져있다.

### 드롭아웃에 의한 과적합 억제

층을 깊게 쌓음으로써 표현력이 풍부한 모델을 만들 수 있지만, 이런 모델은 종종 과적합을 일으킨다. 특히 RNN은 일반적인 피드포워드 신경망보다 쉽게 과적합을 일으킨다. 과적합을 억제하기 위한 일반적인 방법은 '훈련 데이터의 양 늘리기', '모델의 복잡도 줄이기', '정규화'등이 있다. 또 드롭아웃처럼 훈련 시 계층 내의 뉴런 몇 개를 무작위로 무시하고 학습하는 방법도 있다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig27.png?raw=true)

### 가중치 공유

언어 모델을 개선하는 아주 간단한 트릭 중 가중치 공유가 있다. 아래처럼 Embedding 계층과 Softmax 앞단의 Affine 계층의 가중치를 공유시키는 것이다. 이를 통해 학습하는 매개변수의 수를 줄이는 동시에 정확도도 향상시킬 수 있다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2022-1-11-ch6_Gated_RNN/fig28.png?raw=true)