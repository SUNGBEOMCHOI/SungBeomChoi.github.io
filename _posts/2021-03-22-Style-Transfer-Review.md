---
layout: post
title: Image Style Transfer Using Convolutional Neural Networks 리뷰
featured-img: 2021-03-22-Style-Transfer-Review/fig8
permalink: /paper_review/2021-03-22-Style-Transfer-Review
category: paper_implementation
use_math: true

---



이 포스트는 [Image Style Transfer Using Convolutional Neural Networks](https://rn-unison.github.io/articulos/style_transfer.pdf) 논문에 대한 리뷰입니다.

<br>

## Abstract

서로 다른 스타일의 이미지에서 의미있는 정보를 뽑는것은 어려운 image processing 문제이다. 저자는 CNN을 통해서 이미지에서 high level 정보를 뽑아낸다. 그리고 이를 통해 content image와 style image를 결합하는 과정을 통해 Artistic style 알고리즘을 만들었다.

<br>

## Introduction

하나의 이미지의 스타일을 다른 이미지에 옮기는 것은 texture transfer하는 문제이다. Texture transfer의 목표는 source image로 부터 texture를 추출해서 target image의 semantic content는 보존하면서 texture를 합치는 것이다. 이를 위한 여러 non-parametric 알고리즘들이 개발되었다. 이 방법은 주어진 source texture에서 일부를 resampling하는 방법을 사용한다. 하지만 이들의 문제는 low-level image feature들만 사용한다는 것이다.

최근의 Deep convolutional neural network는 이미지로부터 high-level semantic 정보를 뽑는데 좋은 성능을 보여주었다. 또한 object detection, texture recognition, artistic style classification 등의 여러 vision task에 대해서도 generalize하게 사용되었다. 저자는 CNN에서 학습한 이미지의 representation을 통해 content image와 style image를 독립적으로 처리하고 조작하는 parametric한 방법을 보여준다.

<br>

<br>

## Deep image representations

저자의 CNN은 object recognition에 사용된 VGG모델을 basis로 한다. VGG모델에서 16개의 convolution layer와 5개의 pooling layer를 사용한다. pooling layer는 VGG에서 사용했던 max pooling대신 global average pooling을 사용했다.

<br>

### Content representation

여기서 하나의 레이어는 각 사이즈가 M인 N개의 feature map으로 구성되어 있다. M은 feature map의 가로와 세로 사이즈의 곱이다. 레이어 $l$에 대한 activation은 $F_{i j}^{l}$에 저장된다. $F_{i j}^{l}$은 i 번째 필터, 위치 j에서의 activation이다.

서로 다른 레이어에서 추출된 Image 정보를 시각화하기 위해 white noise 이미지에 original image에 대한 gradient descent를 사용한다. $\vec{p}$를 original image, $\vec{x}$를 generated image, $P^{l}$ 와 $F^{l}$를 레이어$l$에서 그들의 feaure representation이라고 하자. 그러면 두 feature representation의 squared-error loss를 다음과 같이 정의할 수 있다.

$$\mathcal{L}_{\text {content }}(\vec{p}, \vec{x}, l)=\frac{1}{2} \sum_{i, j}\left(F_{i j}^{l}-P_{i j}^{l}\right)^{2}$$

이후 위의 loss에 대해서 generated image $\vec{x}$를 갱신하는 방향으로 back-propagation을 진행한다. 시간이 지나면서 generated image는 점점 original image와 가까워지는 방향으로 진행된다. 다음 그림은 레이어에 따른 representation의 정도이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-22-Style-Transfer-Review/fig1.JPG?raw=true)

뒤의 사진일수록 더 깊은 레이어에서 추출한 representation이다. 뒤로갈수록 detail한 픽셀의 정보는 지워지고 high level content만 보존되는 것을 볼 수 있다. 따라서 우리는 뒤의 레이어를 사용하여 content image에 대해서는 전체적인 느낌만 살렸다.

<br>

### Style representation

Style representation을 위해 서로 다른 filter에 대한 correlation 정보를 뽑았다. 이 feature correlation은 Gram matrix $G_{i j}^{l}=\sum_{k} F_{i k}^{l} F_{j k}^{l}$로 주어진다. 다음 그림은 레이어에 따른 representation의 정보이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-22-Style-Transfer-Review/fig2.JPG?raw=true)

content representation의 경우와 마찬가지로 앞쪽의 레이어에서는 detail한 pixel값을 살리고, 뒤의 레이어로 갈 수록 전체적인 느낌을 추출하는 것을 볼 수 있다. original image와 noise image의 gram matrices를 최소화하는 방향으로 gradient descent를 취해준다.

$\vec{a}$를 original image, $\vec{x}$를 generated image라 하고, $A^{l}$와 $G^{l}$를 그들의 style representation 이라 하면 레이어 $l$에서의 total loss는 다음과 같이 정의한다.

$$E_{l}=\frac{1}{4 N_{l}^{2} M_{l}^{2}} \sum_{i, j}\left(G_{i j}^{l}-A_{i j}^{l}\right)^{2}$$

그리고 total style loss는 다음과 같다.

$$\mathcal{L}_{\text {style }}(\vec{a}, \vec{x})=\sum_{l=0}^{L} w_{l} E_{l}$$

여기서 $w_{l}$은 각 레이어의 error에 얼마나 가중치를 줄지를 의미한다. 위의 loss에 대해서 generated image $\vec{x}$를 갱신하는 방향으로 back-propagation을 진행한다.

<br>

### Style transfer

style of an artwork $\vec{a}$를 사진 $\vec{p}$에 옮기기 위해서 새로 생성하는 이미지는 $\vec{a}$의 style representation과 $\vec{p}$의 content representation을 모두 가지고 있어야한다. 아래 그림처럼 style representation은 여러 이미지에서 추출하고, content representation은 하나의 깊은 레이어로 부터 추출하게 된다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-22-Style-Transfer-Review/fig3.JPG?raw=true)

그리고 아래의 loss를 최소화하는 방향으로 generated image를 갱신한다.

$$\mathcal{L}_{\text {total }}(\vec{p}, \vec{a}, \vec{x})=\alpha \mathcal{L}_{\text {content }}(\vec{p}, \vec{x})+\beta \mathcal{L}_{\text {style }}(\vec{a}, \vec{x})$$

$\alpha$와 $\beta$는 얼마나 content와 style representation을 반영할지에 대한 weight factor이다. 저자는 L-BFGS를 optmisation을 위해 사용했다.

<br>

<br>

## Result

아래 그림은 content representation을 위해서는 conv4_2 layer를 사용하고, style representation을 위해서는 conv1_1, 2_1, 3_1, 4_1, 5_1을 사용하여 나온 결과이다. 또 각 레이어의 $w_{l}$는 모두 동일하게 1/5를 사용해주었다. $\alpha / \beta$의 경우 B에서는 $1_10^{-5}$, C에서는 $8_10^{-4}$, D에서는 $5_10^{-3}$, E에서는 $5_10^{-4}$을 사용해주었다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-22-Style-Transfer-Review/fig4.JPG?raw=true)

<br>

### Trade-off between content and style matching

아래 그림은 $\alpha / \beta$에 따른 변화이다. $\alpha / \beta$를 크게 설정할 수 록 content가 살아나는것을 볼 수 있다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-22-Style-Transfer-Review/fig9.JPG?raw=true)

<br>

### Effect of different layers of the Convolutional Neural Network

Style representation에서는 higher layer로 갈수록 더 넓은 범위의 local area를 살리는 것을 볼 수 있었다. 가장 비주얼적으로 끌리는 이미지는 style layer에서는 conv 1_1, 2_1, 3_1, 4_1, 5_1을 사용할 때였다.

content representation에서는 higher layer로 갈수록 디테일한 pixel값은 죽이고, 더 넓은 범위의 특징을 살렸다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-22-Style-Transfer-Review/fig5.JPG?raw=true)

<br>

### Initialisation of gradient descent

맨 처음의 이미지를 white noise를 사용하는지, content image에서 시작하는지, style image를 사용하는지에 따른 결과이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-22-Style-Transfer-Review/fig6.JPG?raw=true)

<br>

### Photorealistic style transfer

조금 더 나아가서 artistic style transfer에 대해 생각해보자. 아래 그림은 뉴욕의 밤의 image를 style 로, 런던의 낮시간의 image를 content로 하여 생성한 이미지이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-22-Style-Transfer-Review/fig7.JPG?raw=true)

<br>

<br>

## Discussion

이 논문에서는 cnn을 사용한 feature representation을 style transfer에서 어떻게 사용하는지 알아보았다. 이것은 좋은 성능을 보여주었으나 몇 가지 한계점이 존재한다. 첫 번째 한계점은 이미지의 resolution이 높아지면 속도가 많이 느려진다는 것이다. 두 번째 한계점은 이미지의 어떤 점을 style이라고 나타낼지 모른다는 것이다. 이 논문에서는 gram matrix를 사용하였으나, 이것이 style의 올바른 표현인지는 확실하지 않다.