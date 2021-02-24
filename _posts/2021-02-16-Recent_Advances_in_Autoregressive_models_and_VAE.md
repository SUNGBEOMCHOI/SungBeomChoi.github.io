---
layout: post
title: PRML 2021 1일차 : Recent Advances in Autoregressive models and VAE
featured-img: 2021-02-16-Recent_Advances_in_Autoregressive_models_and_VAE/title
permalink: /things/Recent_Advances_in_Autoregressive_models_and_VAE
category: things

---

#### 이 글은 정보과학회에서 진행하는 PRML 2021의 첫 번째 세션인 김세훈님의 Recent Advances in Autoregressive models and VAE에 대한 포스트이다.

## introduction
모든 Generative model의 목적은 자신에게 주어진 여러 정보를 통해 모르는 정보를 예측하는 것이다. Generative model 중 하나인 Autoregressive Model은 자신의 과거의 예측이 이후 예측에 영향을 준다는 것이다.

이 모델을 이미지에 적용하면 다음그림과 같이 표현할 수 있다. 과거 데이터인 왼쪽, 위 픽셀을 가지고, 현재 픽셀을 예측하는 것이다. 
![Generative model](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-16-Recent_Advances_in_Autoregressive_models_and_VAE/fig1.JPG?raw=true)

Autoregressive Model은 다음 그림처럼 순서대로 이뤄진다. 맨 처음 1*1 픽셀이 주어지면 오른쪽 아래로 진행하면서 전체 이미지를 생성하게 된다. 
![Create sequantially](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-16-Recent_Advances_in_Autoregressive_models_and_VAE/fig2.JPG?raw=true)

### 모델의 분류
분류를 나누자면 CNN-based Model과 Self-Attention-based Model로 나눌 수 있다. 초기에는 CNN을 가지고 하였으나 long range의 예측에서의 오류가 있었고, 이런 점을 self-attention을 통해서 보완할 수 있었다.

<strong>CNN-based Models</strong>
- PixelCNN
- Gated PixelCNN
- PixelCNN++
- PixelCNN with Auxiliary Variables
- Locally Masked CONV. for AR

<strong>Self-Attention-based Models</strong>
- PixelSNAIL
- Image Transformer
- Sparse Transformer
- Distributional Augmentation
- Image GPT

### Measurement
Measurement 는 bpd(bits/dim)을 사용하게 된다. bpd는 얼마나 적은 정보를 통해서 이미지를 생성할 수 있는지에 대한 지표이다. 현재에는 32px*32px, 8bit의 이미지에서 3.0 정도면 경쟁력있는 알고리즘으로 생각할 수 있다.

## CNN-based Models
### 1. PixelCNN (bpd : 3.14)
픽셀 CNN에서 loss function은 다음과 같다.
![loss function](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-16-Recent_Advances_in_Autoregressive_models_and_VAE/fig13.JPG?raw=true)
위에서 봤듯이 이전 픽셀들의 정보들이 주어지면 현재 픽셀을 잘 유추하도록 학습하는 것이다. 

PixelCNN에서는 과거의 정보만 사용하고, 미래의 데이터의 영향을 없애기 위해 Masked 3*3 conv를 사용하였다. 
![Masked conv](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-16-Recent_Advances_in_Autoregressive_models_and_VAE/fig3.JPG?raw=true)

r, g, b 각각의 픽셀값을 모두 예측하게 되고, 예측은 r, g, b의 순서로 이루어진다. 따라서 r을 이전픽셀로 예측한 이후, g를 예측할 때는 이전 픽셀의 정보 + 현재 픽셀의 r값을 활용한다. 비슷하게 b를 예측할 때는 이전 픽셀의 정보 + 현재 픽셀의 r, g값을 활용하게 된다.
![masked color](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-16-Recent_Advances_in_Autoregressive_models_and_VAE/fig4.JPG?raw=true)

#### Architecture
Masked conv와 weight normalization + elu의 조합을 여러개 거친이후 최종으로 256개 softmax layer를 통과하여 확률값이 나오게 된다. 256인 이유는 8bit 이미지이기 때문이다.
![architecture](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-16-Recent_Advances_in_Autoregressive_models_and_VAE/fig5.JPG?raw=true)

###  2. Gated PixelCNN (bpd : 3.03)
이후 PixelCNN을 발전시키려는 노력이 계속되었다. PixelCNN의 문제점 중 하나는 masked 3*3 conv 를 사용시 blind spot이 생긴다는 것이다. Gated PixelCNN에서는 1*3 conv로 해당 픽셀의 수직 부분의 값을 가져오는 vertical stack과 1*1 conv로 해당 픽셀의 왼쪽 부분의 값을 가져오는 horizontal stack을 나누어 문제를 해결했다.
![Create sequantially](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-16-Recent_Advances_in_Autoregressive_models_and_VAE/fig6.JPG?raw=true)

또 현재 픽셀을 예측할 때 이전 픽셀 정보들의 중요도가 서로다르다.  이점에 착안해서 gated conv layer를 통해 중요도도 학습시켜 반영하였다.

conditional conv layer에서는 픽셀 값 뿐만이 아니라 특정 조건을 주고, 이미지를 생성하도록 한다. 예를 들어 고양이라는 조건과 1*1픽셀의 정보를 넣어주면 sequential하게 픽셀을 생성하여 고양이 이미지를 뽑아내게 된다.
![conditional conv](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-16-Recent_Advances_in_Autoregressive_models_and_VAE/fig7.JPG?raw=true)

### 3. PixelCNN++ (bpd : 2.92)
PixelCNN++은 Gated PixelCNN과 전체적인 구성은 비슷하지만 약간의 구성요소만 변경하여 더 높은 성능을 이끌어냈다. 

첫 번째로 loss function을 변경하였다. 이전에는 출력을 256개 중에 하나를 뽑아내는 방식이었다. 하지만 intensity는 원래 연속적이라고 생각할 수 있기 때문에 여기서는 logistic을 사용했다.
![loss function](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-16-Recent_Advances_in_Autoregressive_models_and_VAE/fig8.JPG?raw=true)

두 번째로 픽셀 이미지를 예측하는 구조를 변경하였다. 이전에는 r, g, b의 값을 따로따로 구해야 하므로 모델을 3번 돌려야했다. 하지만 본 논문에서는 r, g, b는 연관성이 있다고 생각할 수 있고, 이를 활용했다. r 값은 신경망을 돌린 데이터로 구하지만 g를 예측할 때는 r값을 구하기 위해 신경망을 돌린 정보 + 예측한 r값의 연관성을 linear하게 합하는 방법을 사용했다. b를 예측할 때는 추가로 예측한 g의 값도 사용하게 된다.
![create r,g,b by in one time](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-16-Recent_Advances_in_Autoregressive_models_and_VAE/fig9.JPG?raw=true)

세 번째로 모델의 architecture를 변경하였다. 이전의 모델들은 값을 sequential하게 뽑기 때문에 뒤로 갈수록 오차가 커진다는 단점이 있었다. 이를 해결하기위해 U-Net처럼 strided conv를 활용하고, skip connection, dropout을 활용하여 이전 값을 고루 고려하여 값을 뽑을 수 있도록 하엿다.
![architecture](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-16-Recent_Advances_in_Autoregressive_models_and_VAE/fig10.JPG?raw=true)

## Self-Attention-based Models
여러 노력들에도 불구하고 CNN 모델은 long range dependency를 극복하기에는 한계가 있었다. 이런 부분을 해결하기 위해 최근에는 Self-Attention 모듈을 활용한 모델들이 나오고 있다.

### 1. PixelSNAIL (bpd : 2.85)
PixelSNAIL은 CNN과 self-attention을 함께 활용한 하이브리드 모델이다.

PixelSNAIL architecture의 구성요소는 residual block과 attention block으로 나눠진다. 
- residual block: conv layer를 통해 인풋에서 적절한 representation을 추출
-  attention block은 long range의 오차를 보완해주게 된다.
![architecture elements](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-16-Recent_Advances_in_Autoregressive_models_and_VAE/fig11.JPG?raw=true)

아래 사진은 CNN 기반 모델과 Self-Attention 기반 모델을 잘 보여준다. 사진에서 노란색 점이 현재 예측하는 픽셀이고, 보라색 영역은 그 픽셀을 생성하기 위해 실질적으로 참고하는 정보이다. CNN 기반 모델인 Gated PixelCNN과 PixelCNN++는 어느정도 근처의 정보만 참고하고 있지만, Self-Attention을 도입한 PixelSNAIL에서는 이전 픽셀을 전부 참고하고 있다. 
![diffrence btw cnn based and self attention based](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-16-Recent_Advances_in_Autoregressive_models_and_VAE/fig12.JPG?raw=true)
