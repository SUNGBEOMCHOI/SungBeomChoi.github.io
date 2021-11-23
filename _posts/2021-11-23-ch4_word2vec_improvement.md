---
layout: post
title: (밑바닥부터 시작하는 딥러닝2) 4장. word2vec 속도 개선
featured-img: 2021-11-23-ch4_word2vec_improvement/fig7
permalink: /book_review/2021-11-23-ch4_word2vec_improvement
category: book_review

---

3장에서는 word2vec의 구조를 배우고, CBOW 모델을 구현했다. 여기서 구현한 모델에는 몇 가지 문제가 있다. 말뭉치에 포함된 어휘수가 많아지면 계산량도 커진다는 것이다. 이번 장에서는 word2vec의 속도 개선을 목표로 한다. 앞 장의 word2vec에 2가지 개선을 한다. 첫 번째는 Embedding이라는 새로운 계층을 도입한다. 두 번째로 네거티브 샘플링이라는 새로운 손실 함수를 도입한다. 이후 PTB 데이터셋을 가지고 학습을 수행할 것이다.

<br>

## word2vec 개선1

앞장에서 구현한 CBOW 모델은 아래와 같다. 어휘가 7개일때는 전혀 문제가 되지 않는다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-23-ch4_word2vec_improvement/fig1.JPG?raw=true)

하지만 어휘가 100만개, 은닉층의 뉴런이 100개인 CBOW 모델을 생각해보자.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-23-ch4_word2vec_improvement/fig2.JPG?raw=true)

위 모델의 계산을 위해서는 많은 시간이 소요되는데, 정확히는 다음의 두 계산에서 병목이 일어난다.

-   입력층의 원핫 표현과 가중치 행렬 W_in의 곱 계산
-   은닉층과 가중치 행렬 W_out의 곱 및 Softmax 계층의 계산

첫 번째 병목을 살펴보자. 어휘가 100만개라면 원핫표현 하나만해도 원소수가 100만개인 벡터가 되어, 상당한 메모리를 차지하게 된다. 이 원핫벡터와 가중치 행렬 W_in을 곱해야 하는데, 이것만으로 계산자원을 많이 사용한다. 이 문제는 Embedding 계층을 도입하는 것으로 해결된다.

두 번째 병목은 은닉층 이후의 계산이다. 은닉층과 가중치 행렬 W_out의 곱도 계산량이 상당하다. 그리고 softmax 계층에서도 어휘가 많아지면 계산량이 증가하게 된다. 이 문제는 네거티브 샘플링이라는 새로운 손실 함수를 도입해 해결한다.

<br>
### Embedding 계층

앞장에서는 단어를 원핫표현으로 바꾸고, 가중치 행렬을 곱했다. 하지만 실제로 수행하는 일은 단지 행렬의 특정 행을 추출하는 것뿐이다. 따라서 원핫 벡터로의 변환과 MatMul 계층의 행렬 곱 계산을 사실 필요없는 것이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-23-ch4_word2vec_improvement/fig3.JPG?raw=true)

<br>
### Embedding 계층 구현

그러면 가중치 매개변수로부터 단어 ID에 해당하는 행을 추출하는 Embedding 계층을 만들어보자. 순전파에서는 특정행을 추출할 때 그저 원하는 행을 명시하면 끝이다. 역전파에서는 출력층으로부터 전해진 기울기를 입력층으로 전해주면되는데, 활성화된 행에만 적용하면 된다. 똑같은 원소가 중복될때에는 기울기를 더해주면 된다.

```python
import numpy as np

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
        
    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        for i, word_id in enumerate(self.idx):
            dW[word_id] += dout[i]
            
        return None

```

<br>
<br>

## word2vec 개선2

앞에서 말했듯이 남은 병목은 은닉층 이후의 처리(행렬 곱과 softmax계층의 계산)이다. 이 병목을 해소하기 위해서는 네거티브 샘플링이라는 기법을 사용한다.

<br>
### 은닉층 이후 계산의 문제점

앞 절처럼 어휘가 100만 개, 은닉층 뉴런이 100개일때의 CBOW 모델을 생각해보자.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-23-ch4_word2vec_improvement/fig4.JPG?raw=true)

시간이 오래 걸리는 곳은 다음의 두 부분이다.

-   은닉층의 뉴런과 가중치 행렬(W_out)의 곱
-   Softmax 계층의 계산

첫 번째는 거대한 행렬을 곱하는 문제이다. 은닉층의 뉴런은 100, 가중치 행렬의 크기는 100 * 100만이다. 이렇게 큰 행렬의 곱은 시간이 많이 걸리기 때문에 행렬곱을 가볍게 만들어줘야 한다.

두 번째로, Softmax에서도 같은 문제가 발생한다. 100만개의 어휘에서는 100만번 이상의 계산을 해야한다. 여기서도 가벼운 계산이 필요하다.

<br>
### 다중 분류에서 이진 분류로

네거티브 샘플링을 이해하기 위한 핵심은 다중 분류를 이진 분류로 근사하는 것이다. 예를 들어 다중분류에서는 신경망에 맥락이 'you'와 'goodbye'를 주고 정답이 무엇인지 물어보았다. 하지만 이진분류에서는 신경망에 맥락이 'you'와 'goodbye'를 주고, 정답이 'say'입니까 라고 물어보는 것이다.

<br>
### 시그모이드 함수와 교차 엔트로피 오차

이진 분류 문제를 신경망으로 풀려면 점수에 시그모이드 함수를 적용해 확률로 변환하고, 손실을 구할 때는 손실 함수로 '교차 엔트로피 오차'를 사용한다. 시그모이드 함수는 아래와 같다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-23-ch4_word2vec_improvement/fig5.JPG?raw=true)

그리고 이진분류에서의 교차 엔트로피 오차는 아래와 같다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-23-ch4_word2vec_improvement/fig6.JPG?raw=true)

이를 적용한 CBOW모델은 아래와 같다. 은닉층 뉴런 h와, 출력 측의 가중치 W_out에서 단어 "say"에 해당하는 단어 벡터와의 내적을 계산한다. 그리고 그 출력을 Sigmoid with Loss 계층에 입력해 최종 손실을 얻는다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-23-ch4_word2vec_improvement/fig7.JPG?raw=true)

<br>
후반부를 더 간단하게 하기위해 Embedding 계층과 내적 처리를 합친 Embedding Dot 계층을 도입한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-23-ch4_word2vec_improvement/fig8.JPG?raw=true)

```python
class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None
        
    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)
        
        self.cache = (h, target_W)
        return out
    
    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)
        
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh

```

<br>
### 네거티브 샘플링

위의 방법을 이용하면 정답에 대해서만 학습을 한 것이다. 다시말해 오답을 입력하면 어떤 결과가 나올지 확실하지 않다. 우리가 원하는 것은 "say"가 나올 확률은 1에 가깝도록, 그 외 나머지 단어들이 나올 확률은 0에 가깝도록 학습시켜야한다. 이를 위해서 적절한 수의 부정적 예시에 대해서도 학습을 진행한다. 이것을 네거티브 샘플링이라고 한다. 더 정확히는 정답 샘플에서의 손실과 오답 샘플에서의 손실을 더한 값을 최종 손실로 하여 학습한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-23-ch4_word2vec_improvement/fig9.JPG?raw=true)

<br>
### 네거티브 샘플링의 샘플링 기법

그렇다면 오답은 어떻게 샘플링하여야 할까? 단순히 무작위로 샘플링하는 것보다는 말뭉치의 통계 데이터를 기초로 샘플링하는 것이 좋다. 자주 등장하는 단어는 많이 추출하고, 드물게 등장하는 단어는 적게 추출하는 것이다. numpyp.random.choice() 함수를 이용하면 확률분포에 따라 샘플링을 할 수 있다.

```python
words = ['you', 'say', 'goodbye', 'I', 'hello', '.']
p = [0.5, 0.1, 0.05, 0.2, 0.05, 0.1] # 확률 분포
np.random.choice(words, p=p) # 'you'

```

<br>
조금 수정을 하자면 확률분포에서 한가지를 수정한다. 아래식과 같은 새로운 확률분포를 사용하는 것이다. 이렇게 변경하는 이유는 출현 확률이 낮은 단어를 버리지 않기 위해서이다. 0.75 제곱을 함으로서 원래 확률이 낮은 단어의 확률을 살짝 높일 수 있다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-23-ch4_word2vec_improvement/fig10.JPG?raw=true)

<br>
종합하자면 네거티브 샘플링은 말뭉치에서 단어의 확률분포를 만들고, 다시 0.75를 제곱한 다음, np.random.choice()를 사용해 부정적 예를 샘플링한다. 이 처리를 담당하는 클래스를 구현해보자.

```python
import collections

class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None
        
        counts = collections.Counter() # 각 단어의 개수를 셈
        for word_id in corpus:
            counts[word_id] += 1
            
        vocab_size = len(counts)
        self.vocab_size = vocab_size
        
        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]
            
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)
        
    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        
        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
        for i in range(batch_size):
            p = self.word_p.copy()
            target_idx = target[i]
            p[target_idx] = 0 # target이 뽑히지 않게 하기 위함
            p /= p.sum()
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
            
        return negative_sample

```

<br>
### 네거티브 샘플링 구현

```python
from common.layers import SigmoidWithLoss

class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
            
    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)
        
        # 긍정적 예 순전파
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)
        
        # 부정적 예 순전파
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)
            
        return loss
    
    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
            
        return dh

```

<br>
<br>

## 개선판 word2vec 학습

### CBOW 모델 구현

위에서 배운 Embedding 계층과 NegativeSamplingLoss계층을 이용해 아래 그림과 같은 신경망을 구현한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-11-23-ch4_word2vec_improvement/fig7.JPG?raw=true)

```python
class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size
        
        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')
        
        # 계층 생성
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)
        
        # 모든 가중치와 기울기를 배열에 모은다.
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
            
        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in
        
    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss
    
    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None

```

<br>
### CBOW 모델 학습 코드

마지막으로 CBOW 모델의 학습을 구현한다. 윈도우 크기를 5로, 은닉층의 뉴런 수를 100개로 설정했다. 보통 윈도우 크기는 2~10개, 은닉층의 뉴런수는 50 ~ 500개 정도면 좋은 결과를 얻는다. PTB 데이터셋은 말뭉치가 매우 커서 학습 시간이 상당히 오래 걸린다. 학습이 끝나면 가중치를 꺼내, 나중에 이용할 수 있도록 파일에 보관한다. 파일로 저장할 때는 파이썬의 피클 기능을 이용한다.

```python
import numpy as np
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from dataset import ptb
from common.util import create_contexts_target

# 하이퍼파라미터 설정
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

# 데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)

# 모델 등 생성
model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 학습 시작
trainer.fit(contexts, target, max_epoch, batch_size)
# trainer.plot()

# 나중에 사용할 수 있도록 필요한 데이터 저장
word_vecs = model.word_vecs
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)

#---------------------출력---------------------#
# | 에폭 1 |  반복 1 / 9295 | 시간 0[s] | 손실 4.16
# | 에폭 1 |  반복 21 / 9295 | 시간 2[s] | 손실 4.16
# | 에폭 1 |  반복 41 / 9295 | 시간 4[s] | 손실 4.15
# ...

```

<br>
### CBOW 모델 평가

앞 절에서 학습한 단어의 분산 표현을 평가해보자. [2장](https://sungbeomchoi.github.io/book_review/2021-11-21-ch2_NLP)에서 구현한 most_similar() 메서드를 이용하여, 단어 몇 개에 대해 거리가 가장 가까운 단어들을 뽑아보자.

```python
from common.util import most_similar

pkl_file = 'cbow_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']
    
querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

#---------------------출력---------------------#
# [query] you
#  we: 0.6103515625
#  someone: 0.59130859375
#  i: 0.55419921875
#  something: 0.48974609375
#  anyone: 0.47314453125

# [query] year
#  month: 0.71875
#  week: 0.65234375
#  spring: 0.62744140625
#  summer: 0.6259765625
#  decade: 0.603515625

# [query] car
#  luxury: 0.497314453125
#  arabia: 0.47802734375
#  auto: 0.47119140625
#  disk-drive: 0.450927734375
#  travel: 0.4091796875

# [query] toyota
#  ford: 0.55078125
#  instrumentation: 0.509765625
#  mazda: 0.49365234375
#  bethlehem: 0.47509765625
#  nissan: 0.474853515625

```

'you'에 대해서는 비슷한 단어로 인칭대명사 'I'와 'we' 등이 나왔다. 'year'에 대해서는 'month'와 'week' 같은 기간을 뜻하는 단어들이 나왔다. 'toyota'에 대해서는 'ford', 'nissan'같은 자동차 메이커가 나왔다. 이 결과를 보면 CBOW 모델로 획득한 단어의 분산 표현은 제법 괜찮은 특성을 지닌다고 볼 수 있다.

<br>
word2vec으로 얻은 단어의 분산 표현은 비슷한 단어를 가까이 모을 뿐 아니라, 더 복잡한 패턴을 파악하는 것으로 알려져 있다. 대표적인 예가 "king - man + woman = queen"으로 유명한 유추 문제이다. 더 정확히 말하면 word2vec의 단어의 분산 표현을 사용하면 유추 문제를 벡터의 덧셈과 뺄셈으로 풀 수 있다는 뜻이다. 단어 'man'의 분산 표현을 'vec("man")'이라고 표현해보자. 우리가 얻고 싶은 관계를 수식으로 나타내면 'vec("woman") - vec("man") = vec(?) - vec("king")'이 된다. 이 로직을 analogy()함수로 구현해보자. analogy()함수는 "a:b=c:?"에서 ?로 추정되는 상위 5개 단어를 리턴한다.

```python
def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    for word in (a, b, c):
        if word not in word_to_id:
            print(f'{word}(을)를 찾을 수 없습니다.')
            
    print(f'\\n[analogy] {a}:{b} = {c}:?')
    a_vec, b_vec, c_vec = word_matrix[word_to_id[a]], word_matrix[word_to_id[b]], word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec
    
    eps = 1e-8
    similarity = np.dot(word_matrix, query_vec)
    
    if answer is not None:
        print(f'==>{answer}:{str(np.dot(word_matrix[word_to_id[answer]], query_vec))}')
        
    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(f'{id_to_word[i]}: {similarity[i]}')
        
        count += 1
        if count >= top:
            return

```

<br>
실제로 몇 가지 유추 문제를 풀어보자. 첫 번째 문제 "king : man = queen : ?"의 대답은 "woman"이다. 두 번째 문제 "take : took = go : ?"의 답은 "went"이다. 어느정도 우리가 원하는 답을 내는것을 확인할 수 있다.

```python
analogy('king', 'man', 'queen', word_to_id, id_to_word, word_vecs)
analogy('take', 'took', 'go', word_to_id, id_to_word, word_vecs)
analogy('car', 'cars', 'child', word_to_id, id_to_word, word_vecs)
analogy('good', 'better', 'bad', word_to_id, id_to_word, word_vecs)

#---------------------출력---------------------#
# [analogy] king:man = queen:?
# woman: 75.125
# veto: 71.75
# ounce: 68.3125
# earthquake: 67.4375
# successor: 67.125

# [analogy] take:took = go:?
# went: 62.4375
# points: 58.34375
# began: 56.15625
# comes: 54.65625
# oct.: 53.59375

# [analogy] car:cars = child:?
# children: 76.1875
# average: 69.0
# yield: 61.40625
# cattle: 61.125
# priced: 61.0

# [analogy] good:better = bad:?
# more: 91.625
# less: 83.5625
# rather: 71.9375
# slower: 65.25
# greater: 64.375

```

<br>
<br>

## word2vec 남은 주제

### word2vec을 사용한 애플리케이션의 예

단어의 분산표현이 중요한 이유는 전이 학습에 있다. 전이 학습은 한 분야에서 배운 지식을 다른 분야에도 적용하는 기법이다. 자연어 문제를 풀 때 word2vec의 단어 분산 표현을 처음부터 학습하는 일은 거의 없다. 그 대신 먼저 큰 말뭉치로 학습을 끝난 후, 그 분산 표현을 각자의 작업에 이용하게 된다. 예를 들어 텍스트 분류, 문서 클러스터링, 품사 태그 갈기, 감정 분석 등 자연어 처리 작업이라면 가장 먼저 단어를 벡터로 변환하는 작업을 해야하는데, 이때 학습을 미리 끝낸 단어의 분산 표현을 이용할 수 있다.

단어의 분산 표현은 단어를 고정 길이 벡터로 변환해준다는 장점도 있다. 게다가 문장도 단어의 분산 표현을 사용하여 고정 길이 벡터로 변환할 수 있다. 문장을 고정 길이 벡터로 변환하는 방법은 활발하게 연구되고 있는데, 가장 간단한 방법은 문장의 각 단어를 분산 표현으로 변환하고 그 합을 구하는 것이다. 이를 bag-of-words라 하여, 단어의 순서를 고려하지 않는 모델이다. 또한 5장에서 설명하는 순환 신경망(RNN)을 사용하면 한 층 세련된 방법으로 문장을 고정 길이 벡터로 변환할 수 있다.

<br>
### 단어 벡터 평가 방법

word2vec을 통해 얻은 단어의 분산 표현이 좋은지는 어떻게 평가할까? 자주 사용되는 평가 척도가 단어의 '유사성'이나 '유추 문제'를 활용한 평가이다. 단어의 유사성 평가에서는 사람이 작성한 단어 유사도를 검증 세트를 사용해 평가하는 것이 일반적이다. 예를 들어 유사도를 0에서 10사이로 점수화한다면, "cat"과 "animal"의 유사도는 8점, "cat"과 "car"의 유사도는 2점과 같이, 사람이 단어 사이의 유사한 정도를 규정한다. 그리고 사람이 부여한 점수와 word2vec에 의한 코사인 유사도 점수를 비교해 그 상관성을 보는 것이다. 유추 문제를 활용한 평가는 "king : queen = man : ?"와 같은 유추 문제를 출제하고, 그 정답률로 단어의 분산 표현의 우수성을 측정한다. 유추 문제를 이용하면 '단어의 의미나 문법적인 문제를 제대로 이해하고 있는지'를 측정할 수 있다.