---
layout: post
title: Sequence to Sequence Learning with Neural Networks 리뷰
featured-img: 2021-03-30-Sequence_to_Sequence_Review/fig1
permalink: /paper_review/2021-03-30-Sequence_to_Sequence_Review
category: paper_review
use_math: true

---

이 포스트는 [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf) 논문에 대한 리뷰입니다. 부족한 지식으로 잘못된 내용이 있을 수 있습니다.

<br>

## Abstract

이 논문에서 저자는 여러층의 LSTM(encoder)을 사용하여 input sequence를 고정된 크기의 vector 변환한 이후, 또 다른 deep LSTM(decoder)을 통해 vector로 부터 target sequence를 만든다. WMT14 데이터셋을 사용했고, English to French 번역 문제를 해결했다. 여기서 34.8 BLEU score를 얻을 수 있었다. LSTM와 SMT 방식을 함께 사용했을 대는 36.5 BLEU score를 기록했다. 추가로 LSTM은 긴 문장에 대해서도 작업을 잘 수행했다. 또 source sentence에서 입력을 거꾸로 주었을 때 성능이 더 개선되었다. 이는 optimization 문제를 더 쉽게한 덕분인듯 하다.

<br>

## Introduction

DNN은 powerful machine learning model로서 여러 문제(speech recognition, object recognition)에서 좋은 성능을 낸다. 하지만 DNN은 input 과 target사이즈가 고정되어야한다는 단점이 있다. 사이즈가 고정되어 있다는 것은 speech revognition이나 machine translation과 같은 sequential problem에서는 치명적이다.

이를 해결하기 위해 이 논문에서는 LSTM을 통해 input sequence를 읽어 large fixed dimensional vecotor로 만들고, 이를 다른 LSTM에 넣어 output sequence를 뽑아낸다. 아래 그림은 대략적인 architecture이다.

성능으로는 5개의 deep LSTM을 앙상블함으로서 34.81 BLEU score를 기록했다. 이 성능은 neural network를 활용한 direct translation에서는 최고의 성능이다. 비교를 위해 SMT baseline은 33.30 BLEU score를 기록한다. 저자의 모델은 80k의 단어만으로 구성했기 때문에 모든 단어를 기록하기에는 부족했다. 이런 페널티에도 SMT 보다 좋은 성능을 낼 수 있었다. SMT를 통해 1000개의 best list를 뽑고, 이에 대해 LSTM을 사용했을 때는 36.5 BLEU score를 얻을 수 있었다.

이 논문에서 사용한 LSTM 모델로는 긴 문장에 대해서도 성능이 잘 나왔다. 이것은 source sentence를 뒤집어서 넣어줌으로서 가능했다. 뒤집어서 input을 넣어줌으로서 많은 short term dependencies를 통해 optmization을 더 쉽게 한 결과이다.

LSTM의 또다른 장점은 다양한 길이의 input sentence를 고정된 크기의 vector로 representation할 수 있다는 것이다. LSTM은 input sentence에 있는 의미를 잘 캐치하여 비슷한 의미는 가깝고, 다른 의미는 멀게 표현했다. 모델이 어순을 인지하고, 능동태와 수동태에 불변적임을 알 수 있었다.

<br>
<br>

## The model

RNN은 주어진 input sequence $\left(x_{1}, \ldots, x_{T}\right)$에 대해 sequence $\left(y_{1}, \ldots, y_{T}\right)$를 다음과 같은 식을 반복하여 계산하게 된다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-30-Sequence_to_Sequence_Review/fig1.JPG?raw=true)

하지만 RNN은 입력과 출력의 크기가 다를 경우에 대해서는 대처하지 못한다. 이를 해결하기 위한 간단한 방법은 하나의 RNN을 통해 고정된 크기의 vector를 생성하고, 이를 다른 RNN에 넣어 출력 sequence를 생성하는 것이다. 하지만 RNN은 긴 sequence에 대해서 잘 대처하지 못한다. 하지만 LSTM을 사용하여 long range temporal dependency도 학습시킬 수 있다. 따라서 여기서는 LSTM을 사용한다.

LSTM의 목적은 조건부확률 $p\left(y_{1}, \ldots, y_{T^{\prime}} \mid x_{1}, \ldots, x_{T}\right)$를 계산하는 것이다. encoder LSTM으로 input sequence 로부터 마지막 hidden state를 fixed dimensional representation $v$를 뽑고, decoder LSTM으로 output sequence를 뽑는다. 아래 식을 참고하라.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-30-Sequence_to_Sequence_Review/fig2.JPG?raw=true)

각 $p\left(y_{t} \mid v, y_{1}, \ldots, y_{t-1}\right)$는 모든 단어로부터 softmax를 계산하여 얻는다. 각 sentence의 끝에는 끝을 나타내는 심볼 <EOS>가 필요하다. 아래 그림을 참고하라. input sequence 'A, B, C, <EOS>'로 부터 representation vector $v$를 뽑아, output sequence 'W, X, Y, Z, <EOS>'가 될 확률을 계산한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-30-Sequence_to_Sequence_Review/fig3.JPG?raw=true)

실질적인 모델은 위의 그림과 아래와 같은 점에서 다르다.

-   두 개의 LSTM(encoder, decoder)을 사용한다.
-   깊은 LSTM을 사용한다.
-   input sentence의 입력을 뒤집어서 넣어준다. 위 그림에서와 달리 'C, B, A'를 입력으로 준다.

<br>
<br>

## Experiments

### Dataset details

WMT'14 English to French 데이터셋을 사용했다. 348M개의 French 단어와 304M개의 English로 구성되어 있는 12M개의 sentence를 학습시켰다. 임베딩을 위해서는 source 언어에서 160,000개의 가장 빈번한 단어, target 언어에서 80,000개의 빈번한 단어를 사용했다. 이 외의 단어는 'UNK'라는 심볼로 대체되었다.

<br>

### Decoding and Rescoring

학습의 목적은 source sentence S로부터 correct translation T에 대한 확률을 최대화하는 것이다. 아래 식 앞에있는 S는 학습 셋을 의미한다. 즉 이미 주어진 S와 T에 대해 학습을 진행하는 것이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-30-Sequence_to_Sequence_Review/fig4.JPG?raw=true)

학습이 끝난 이후에는 아래식처럼 가장 확률이 높은 T를 추출하는 방법으로 번역을 수행한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-30-Sequence_to_Sequence_Review/fig5.JPG?raw=true)

가장 그럴듯한 번역을 위해 beam search decoder를 사용한다. beam size가 2일때 가장 효과적으로 사용할 수 있었다. 또 LSTM을 통해 모든 단어들에 대한 확률을 계산하고 SMT으로 1000-best list를 뽑아 둘을 평균내는 방법도 사용해보았다.

<br>

### Reversing the Source Sentences

Source sentence의 순서를 뒤집어서 입력으로 넣어주었을 때 긴 문장에 대해서도 학습을 잘 수행할 수 있었다. 또 성능이 더 좋았다. LSTM을 그냥 사용했을 때는 25.9 BLEU score가 나왔지만 순서를 뒤집어서 입력으로 넣어주었을 때 30.6 BLEU score의 성능이 나왔다. 일반적으로 source sentence와 target sentence 사이의 거리가 멀다. 결과적으로 'minimal time lag' 문제를 일으키게된다. source sentence를 뒤집으면 평균적인 source sentence와 target sentence의 거리가 바뀌지는 않는다. 하지만 source sentence의 첫 번째 단어와 target sentence의 첫 번째 단어와의 거리는 매우 가까워진다. 이러면 minimal time lag문제가 현저히 줄어든다.

<br>

### Training details

-   Using 4layers LSTMs with 1000 cells at each layer
-   1000 dimensional word embeddings, with an input vocabulary of 160,000 and an output vocabulary of 80,000
-   8000 real numbers to represent a sentence
-   LSTM 파라미터는 -0.08~0.08로 uniform distribution으로 초기화
-   optimizer로는 SGD사용. learning rate 0.7, 5 epoch이후 절반 epoch마다 절반으로 줄임. 총 7.5 epochs training
-   batch of 128 sequence
-   exploding gradient 문제를 해결하기 위해 gradient에 hard constraint on the norm을 적용. 예시로 $s=\|g\|_{2}$(g는 gradient devided by 128)를 계산해보고, s>5이면 g=5g/s 로 적용
-   대부분의 문장이 짧고(20-30), 일부의 문장이 길다(over 100). 만약 128의 minibatch에 짧은 문장과 긴 문장을 섞어 넣으면 효율적이지 않다. 따라서 각 minibatch에는 비슷한 길이의 문장끼리 넣어주었다.

<br>

### Parellelization

하나의 GPU를 사용할 때는 초당 1700단어를 처리했다. 이것은 너무 느려서 총 8개의 GPU를 사용했다. 각 GPU는 하나의 LSTM layer를 담당한다(총 8개의 LSTM layer, encoder에 4 layer, decoder에 4 layer). 이 경우 초당 6300단어를 처리했다. 학습을 위해서 총 10일이 소요되었다.

<br>

### Experimental Results

가장 좋은 결과는 random initialization을 취한 5개의 LSTM을 ensemble했을 때 결과가 가장 좋았다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-30-Sequence_to_Sequence_Review/fig6.JPG?raw=true)

또 1000-best SMT와 5개의 LSTM ensemble을 결합했을 때 36.5BLEU score를 기록할 수 있었다. 이는 SOTA와도 0.5 점밖에 차이가 나지 않는다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-30-Sequence_to_Sequence_Review/fig7.JPG?raw=true)

<br>

### Performance on long sentence

우리의 모델은 긴 문장에 대해서도 잘 처리할 수 있다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-30-Sequence_to_Sequence_Review/fig8.JPG?raw=true)

<br>

### Model Analysis

source sentence로부터 변환한 vector를 살펴보면 단어의 순서에는 민감하지만 수동태와 능동태에 대해서는 insensitive하다는 것을 확인할 수 있다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-30-Sequence_to_Sequence_Review/fig9.JPG?raw=true)

<br>
<br>

## Related work

Machine translation을 해결하기 위한 여러가지 논문들에 대한 설명

<br>

## Conclusion

-   large deep LSTM을 통해 고전적인 방법인 standard SMT system보다 machine translation문제를 잘 해결할 수 있음을 보였다.
-   source sentence에서 단어의 순서를 바꾸면 더 성능이 좋다는 것을 찾아냈다.
-   LSTM을 통해 long sentence도 잘 해석할 수 있었다.
-   우리의 모델은 machine translation task뿐 아니라 sequence to sequence가 적용되는 다른 문제에도 잘 적용될 수 있을 것이다.