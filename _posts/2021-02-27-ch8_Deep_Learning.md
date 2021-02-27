---
layout: post
title: (밑바닥부터 시작하는 딥러닝) 8장. 딥러닝
featured-img: 2021-02-20-ch8_Deep_Learning/fig1
permalink: /book_review/2021-02-20-ch8_Deep_Learning
category: book_review

---

## 조금 더 깊은 신경망

다음 그림과 같은 조금 더 깊은 CNN을 만들어보고, MNIST데이터셋에 적용해본다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-27-ch8_Deep_Learning/fig1.jpg?raw=true)

<br>

### 네트워크 특징
-   3*3의 작은 필터를 사용한 합성곱 계층
-   층이 깊어질수록 채널 수가 늘어남
-   활성화 함수는 ReLU
-   완전연결 계층 뒤에 드롭아웃 계층 사용
-   Adam을 사용해 최적화
-   가중치 초깃값은 He의 초깃값

<br>

### 네트워크 코드
~~~
class DeepConvNet:
    """정확도 99% 이상의 고정밀 합성곱 신경망

    네트워크 구성은 아래와 같음
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
    """
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_4 = {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
                 conv_param_5 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_6 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 hidden_size=50, output_size=10):
        # 가중치 초기화===========
        # 각 층의 뉴런 하나당 앞 층의 몇 개 뉴런과 연결되는가（TODO: 자동 계산되게 바꿀 것）
        pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLU를 사용할 때의 권장 초깃값
        
        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(conv_param['filter_num'], 
                                                        pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        self.params['W7'] = weight_init_scales[6] * np.random.randn(64*4*4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = weight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        # 계층 생성===========
        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'], 
                           conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W2'], self.params['b2'], 
                           conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W3'], self.params['b3'], 
                           conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                           conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                           conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W6'], self.params['b6'],
                           conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W8'], self.params['b8']))
        self.layers.append(Dropout(0.5))
        
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]
~~~

<br>

### 훈련코드
~~~
from dataset.mnist import load_mnist
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='Adam', optimizer_param={'lr':0.001},
                 evaluate_sample_num_per_epoch=1000)
trainer.train()
~~~

<br>

### Accuracy 측정
놀랍게도 99.6%라는 놀라운 성능을 보여준다.
~~~
index = np.random.choice(x_test.shape[0], 1000)
acc = network.accuracy(x_test[index], t_test[index])
print(acc) 0.996
~~~

<br>
<br>

## 정확도를 더 높이려면

###  Data augmentation
회전, 이동, crop, flip 등을 통해서 데이터의 양을 늘리는 기법을 의미한다.
    
![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-27-ch8_Deep_Learning/fig2.jpg?raw=true)
    
<br>

### 층을 깊게 한다.

### 학습의 효율성이 좋아진다.
예를 들어 5*5 receptive field를 분석하기 위해서는 한층을 사용할 경우에는 25개의 파라미터가 필요하다. 하지만 두 층을 사용하면 18개의 파라미터만 학습시키면 된다. 

#### 1층을 사용한 경우

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-27-ch8_Deep_Learning/fig3.jpg?raw=true)

#### 2층을 사용한 경우
![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-27-ch8_Deep_Learning/fig4.jpg?raw=true)

<br>

### 계층적으로 분석할 수 있다.
예를들어 이미지를 보고, 클래스를 예측하는 분류문제에서 한층으로 한다고 하면 이미지에서 바로 어떤 클래스인지 맞춰야한다. 하지만 여러 층으로 나누게 되면 첫 번째 층에는 엣지나 blob 같은 간단한 특징을 뽑고, 층이 깊어질수록 조금 더 복잡한 특징을 추출하는 등 계층적으로 분석을 할 수 있다.

<br>
<br>

## 딥러닝의 초기역사

### 이미지넷

이미지넷은 100만장이 넘는 이미지를 담고 있는 데이터셋이다. 이 데이터를 이용한 시험 중 하나가 분류대회이다. 분류대회에서는 1000개의 클래스를 제대로 분류하는지를 겨룬다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-27-ch8_Deep_Learning/fig5.jpg?raw=true)

이 중 세 가지 신경망을 소개하도록 하겠다.

<br>

### VGG

16층으로 층 수를 이전보다 늘렸고, 3*3의 작은 필터를 사용한 합성곱 계층을 연속으로 거친다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-27-ch8_Deep_Learning/fig6.jpg?raw=true)

<br>

### GoogLeNet

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-27-ch8_Deep_Learning/fig7.jpg?raw=true)

<br>

GoogLeNet은 가로 방향에 폭이 있고, 이를 인셉션 구조라한다. 인셉션 구조는 크기가 다른 필터를 여러 개 적용하여 그 결과를 결합한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-27-ch8_Deep_Learning/fig8.jpg?raw=true)

<br>

### ResNet

딥러닝 학습에서는 층이 지나치게 깊으면 학습이 잘 되지 않고, 오히려 성능이 떨어지는 경우도 있다. ResNet에서는 스킵연결로 층의 깊이에 비례해 성능을 향상하도록 했다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-27-ch8_Deep_Learning/fig9.jpg?raw=true)

<br>

스킵연결은 입력데이터를 합성곱 계층을 건너뛰어 출력에 바로 더하는 구조이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-27-ch8_Deep_Learning/fig10.jpg?raw=true)

<br>
<br>

## 딥러닝 고속화

GPU를 활용해 대량의 연산을 고속으로 처리할 수 있다. 최근 프레임워크에서는 학습을 복수의 GPU와 여러 기기로 분산 수행한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-02-27-ch8_Deep_Learning/fig11.jpg?raw=true)

알렉스넷의 학습시간은 CPU에서는 40일이나 걸리지만 GPU로는 6일까지 단축된다. 또 cuDNN이라는 딥러닝에 최적화된 라이브러리를 사용하면 더 빨라진다.