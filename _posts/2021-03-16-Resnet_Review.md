---
layout: post
title: Deep Residual Learning for Image Recognition(ResNet) 리뷰
featured-img: 2021-03-16-Resnet_Review/fig2
permalink: /paper_review/2021-03-16-Resnet_Review
category: paper_review

---

이 포스트는 ResNet으로 알려져있는 Deep Residual Learning for Image Recognition에 대한 리뷰입니다. [논문](https://arxiv.org/pdf/1512.03385.pdf)은 해당 링크를 참고하세요.

<br>

## Abstract

깊은 신경망일수록 훈련시키기가 힘들다. 저자는 residual network를 통해 해당 문제를 해결하고자 한다. Residual network를 사용하면 더 깊은 층의 학습을 잘 수행할 수 있어 더 높은 accuracy를 얻을 수 있다. ImageNet 데이터셋에서 모델로 152 layers(VGG보다 8배 더 깊은)를 쌓아 3.57%의 error를 달성했다. 이 결과로 ILSVRC 2015에서 우승을 했다.

<br>

<br>

## Introduction

최근(2015년)의 연구 결과들은 층을 깊게 쌓는것이 중요하다는 점을 시사하고, ImageNet의 대회에서 우수한 성적을 내는 모델들은 모두 층이 매우 깊다. 하지만 층을 깊게 쌓으면 여러 문제에 직면한다. 그 중 하나는 모델 weight의 수렴이 안되는 vanishing/exploding 문제이다. 이 문제를 해결하기 위해 normalrized initialization이나, batch normalization 등의 기법이 제시되었다.

층을 깊게 쌓으면 발생하는 또 다른 문제는 Degradation problem이다. 층을 깊게 쌓으면 오히려 accuracy가 떨어지는 문제이다. 이 문제의 원인은 오버피팅은 아니다. 오버피팅이라면 train accuracy가 높고, test accuracy가 낮아야하지만, 그래프를 보면 모든 accuracy가 낮은 것을 볼 수 있다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-16-Resnet_Review/fig1.JPG?raw=true)

저자는 이 문제를 해결하기 위해 deep residual learning framework를 제시한다. 기존의 mapping이 H(x)라면 본 논문에서는 F(x) = H(x) - x를 제시한다. 식 F(x) + x과 같이 하나 이상의 레이어를 skip하여 initial mapping을 더하는 것을 shortcut connections 라고한다. 만약 input으로 x가 들어가면 output으로 레이어 F를 거친 F(x)와 x과 더해진 F(x) + x가 나오고, 이것이 activation function인 relu를 거치게 된다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-16-Resnet_Review/fig2.JPG?raw=true)

여기서 우리는 2가지를 보이려고 한다.

1.  Deep residual net이 더 쉽게 optimize할 수 있고, 단순하게 레이어를 쌓은 plain net보다 더 좋은 성능을 낼 수 있다.
2.  Deep residual net을 이용하면 층을 높게 쌓을수록 accuracy를 높일 수 있다.

우리는 ILSVRC에서 우리의 net과 ensemble기법을 활용해 top-5 3.57% error를 달성하여 1등을 했다. 또 COCO 데이터셋을 활용한 여러 대회에서도 우수한 성적을 거두었다.

<br>

<br>

## Related Work

Residual Representations와 Shortcut Connections에 대한 선행 논문들에 관한 내용들(생략)

<br>

## Deep Residual Learning

### Residual Learning

여러개의 레이어를 거치고, 찾아야 하는 mapping을 H(x)라고 하자. 여러개의 비선형의 layer로 복잡한 함수를 근사할 수 있다고 가정하면 H(x) - x 도 근사할 수 있다. 여기서 x(identity mapping)를 더해줌으로서 깊은 모델이더라도 이미 거친 layer보다는 error가 더 커질 수 없다.

<br>

### Identity Mapping by Shortcuts

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-16-Resnet_Review/fig3.JPG?raw=true)

residual learning을 여러 레이어를 거칠때마다 적용할 것이다. 식은 위와 같다. x, y는 인풋과 아웃풋이고, F는 학습해야할 residual mapping이다. F는 여러개의 레이어로 구성되어 있다. F가 한 개의 레이어라면 y = Wx + x 형태를 띄는 linear layer와 동일하기 때문에 shortcut connection의 의미가 사라진다. F(x) 를 인풋 x와 더해준 이후에는 activation function을 한 번 더 적용해준다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-16-Resnet_Review/fig4.JPG?raw=true)

만약 F를 거치면서 shape가 달라질 수 있다. 이 때는 x에 linear projection w를 곱해준다. w를 통해 F(x)와 같은 shape로 맞춰준다.

<br>

### Network Architectures

본 논문에서는 plain net과 residual net을 각각 학습시킨다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-16-Resnet_Review/fig5.JPG?raw=true)

#### Plain Network

기본적으로 VGG net과 비슷한 형태를 띄고 있다. convolution layer는 3*3 filter로 구성되어 있고, 만약 feature map의 크기가 절반으로 줄어들면, filter의 수를 2배로 늘린다. 따라서 뒤의 층으로 갈수록 크기는 줄어들고, filter의 수는 늘어난다. 여기서는 pooling을 따로 사용하지 않고, convolution layer의 stride를 2로 조절하여 크기를 줄인다. 마지막에는 global average pooling layer를 사용하고, fully connected layer에서 softmax를 통해 class 수만큼 output을 낸다.(이미지넷의 경우 1000)

<br>

#### Residual Network

위의 Plain Network를 기본으로 하고, shortcut connections를 추가한다. input x와 F(x)가 같은 dimension size일 경우 그대로 더해준다. 만약 input x와 F(x)가 다른 dimension size일 경우 x에도 1*1filter, stride 2인 convolution layer를 사용하여 dimension size를 맞춰준다.

<br>

### Implementation

-   이미지의 짧은 변을 기준으로 (256, 480)의 크기로 scale augmentation을 진행
-   이미지를 224*224 사이즈로 random crop, horizontalal flip, color augmentation을 진행
-   이미지에 per-pixel mean subtracted을 적용
-   Batch normalization을 convolution layer와 activation function 사이 적용
-   He initialize
-   Batch size 256사용
-   Optimizer로 SGD를 사용하고, learning rate는 0.1 그리고, error의 변화가 없을때마다 10으로 나눠줌.
-   weight decay는 0.0001, momentum은 0.9 사용

<br>

<br>

## Experiments

### ImageNet Classification

ImageNet 2012 classification dataset을 통해 top-1, top-5 error rate를 측정했다. 아래 사진은 ImageNet에 사용된 네트워크의 구조이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-16-Resnet_Review/fig6.JPG?raw=true)

<br>

#### 결과1. Residual learning이 degration problem을 해결하는데 도움이 된다.

아래 사진에서도 알 수 있듯이 plain net에서는 층이 깊은 net이 얕은 net보다 error rate가 더 높다. 하지만 residual learning을 사용한 net 에서는 층이 깊은 net보다 얕은 net이 error rate가 더 높다. plain net에서는 degradation problem이 나타났지만 residual net에서는 나타나지 않았다. residual net에서는 더 깊은 net도 optmizing을 잘 한 것이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-16-Resnet_Review/fig7.JPG?raw=true)

<br>

#### 결과2. 층이 깊어질수록 ResNet의 효과가 드러난다.

18 layer에서는 plain net과 Residual net의 성능이 비슷하지만 34 layer에서는 residual net이 더 뛰어난 것을 볼 수 있다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-16-Resnet_Review/fig8.JPG?raw=true)

<br>

#### 결과3. shortcuts connection을 위한 x dimension size 변환 방식은 크게 영향이 없다.

이전에 F(x)와 x의 dimension size가 다르면 x에 변환이 필요하다고 했다. 변환 방식을 3가지 옵션을 통해 서로 비교해보았다.

-   A 옵션 : Dimension size 변경을 위해 zero-padding을 사용
-   B 옵션 : Dimension size 변경을 위해 projection shortcuts을 사용(dimension size 변경이 필요없는 경우 그대로 진행)
-   C 옵션 : 모든 shortcut에 projection shortcuts을 사용

<br>

그 결과 미세한 차이가 있긴하지만 degration problem을 해결하기위해 필수적이지는 않다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-16-Resnet_Review/fig9.JPG?raw=true)

<br>

#### Deeper Bottleneck Architecture

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-16-Resnet_Review/fig10.JPG?raw=true)

더 깊은 층의 학습을 위해 2 layer의 building block 대신 3층의 bottleneck을 사용했다. bottleneck은 1_1, 3_3, 1_1 convolution layer로 구성되어있다. 1_1 layer를 통해 dimension의 사이즈를 줄였다가 키우는 작용을 한다. ResNet-34에서는 기본 building block을 사용하고, ResNet-50,101,152에서는 bottleneck building block을 사용하여 위 표와 같은 성과를 얻을 수 있었다.

<br>

<br>

### CIFAR-10

CIFAR-10은 50,000개의 학습 데이터와 10,000개의 테스트 데이터로 구성된 이미지 데이터셋이다. 총 10개 class로 분류되어있다. 네트워크에 인풋으로 per-pixel mean subtracted가 적용된 32_32 이미지가 들어간다. 첫 번째 layer는 3_3 covolution layer이고, 이후 filter의 수가 (32, 16, 8)인 3*3 convolution layer가 이어진다. subsampling을 위해 stride 2를 사용한다. net의 끝에는 global average pooling을 거쳐 10개의 output을 내는 fully connected layer를 사용한다. 그래서 총 6n+2의 weighted layer가 사용된다. 여기서는 n={3, 5, 7, 9, 18}의 net을 시험했다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-16-Resnet_Review/fig11.JPG?raw=true)

<br>

#### implementation

-   weight decay: 0.0001, momentum: 0.9를 사용
-   He weight initialization
-   batch normalization
-   batch size: 128
-   learning rate는 0.1로 시작하고, 32k와 48k iteration에 각각 10으로 나눠준다.
-   augmentation : 이미지에 4 padding을 하고, 32*32로 random crop을 진행, 여기서 horizontal flip 을 random으로 진행한다.

학습 결과는 아래와 같다. 확실히 층이 깊어질수록 error rate가 내려간다. 1202 layer에서 error rate가 오히려 올라간 것은 overfitting의 영향이지 degradation problem으로 인한것이 아니다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-16-Resnet_Review/fig12.JPG?raw=true)

<br>

<br>

### Object Detection on PASCAL and MS COCO

본 논문에서는 Faster R-CNN에서 VGG-16을 ResNet 101로 바꾸어서 학습하였다. 그 결과 VGG-16보다 더 높은 mAP를 얻을 수 있었다. 아래 표는 각각 PASCAL과 COCO 데이터셋에 적용한 결과이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-16-Resnet_Review/fig13.JPG?raw=true)