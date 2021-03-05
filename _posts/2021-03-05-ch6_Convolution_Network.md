---
layout: post
title: (시작하세요! 텐서플로 2.0 프로그래밍) 6장. 컨볼루션 신경망(CNN)
featured-img: 2021-03-05-ch6_Convolution_Network/fig2
permalink: /book_review/2021-03-05-ch6_Convolution_Network
category: book_review

---


## 특징 추출

앞장에서 배운 보스턴 주택 가격 데이터셋에는 주택의 가격을 예측하기 위한 주택당 방의 수, 재산세율, 범죄율 같은 특징(feature)들이 있었다.

Fashion MNIST 같은 이미지 데이터에서는 특징을 직접 찾아야 한다. 과거의 비전 연구에서는 특징을 연구하기 위한 다양한 방법이 개발되었다. 예를 들어 SIFT 알고리즘은 이미지의 회전과 크기에 대해 변하지 않는 특징을 추출해서 두 개의 이미지에서 서로 대응하는 부분을 찾아낸다.

이런 특징 추출(Feature Extraction) 기법 중 하나인 컨볼루션 연산은 각 필셀을 본래 픽셀과 그 주변 픽셀의 조합으로 대체하는 동작이다. 아래 그림에서는 3*3 크기의 작은 필터는 왼쪽의 원본 이미지를 각각 새로운 이미지로 변환한다. 각 필터의 생김새에 따라 수직선/수평선, 흐림, 날카로움 등의 특징을 추출할 수 있다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-05-ch6_Convolution_Network/fig1.JPG?raw=true)

컨볼루션 신경망(CNN)은 사람이 직접 설계하는 것이 아닌 네트워크가 특징을 추출하는 필터를 자동으로 생성한다. 학습을 계속하면 네트워크를 구성하는 각 뉴런들은 입력한 데이터에 대해 특정 패턴을 잘 추출할 수 있도록 적응한다.

<br>
<br>

## 주요 레이어 정리

### 컨볼루션 레이어(convolution Layer)

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-05-ch6_Convolution_Network/fig2.JPG?raw=true)

컨볼루션 레이어를 통해 데이터의 특징을 추출한다.

kernel_size는 필터 행렬의 크기이다. 이것을 receptive field라고도 부른다.

strides는 필터가 계산 과정에서 한 스텝마다 이동하는 크기이다.

padding은 컨볼루션 연산 전에 입력 이미지 주변에 빈값을 넣을지 지어하는 옵션으로 'valid'와 'same'이라는 2가지 옵션 중 하나를 사용한다. 'valid'는 빈값을 사용하지 않는다. 'same'은 빈값을 넣어서 출력 이미지의 크기를 입력과 같도록 보존한다.

filters는 필터의 갯수이다. 필터의 개수는 네트워크가 얼마나 많은 특징을 추출할 수 있는지 결정하기 때문에 많을수록 좋지만, 너무 많을 경우 학습 속도가 느려질 수 있고, 과적합이 발생할 수 있다.

```python
tf.keras.layers.Conv2D(kernel_size=(3,3),strides=(2,2),padding='valid',filters=16)

```

<br>

### 풀링 레이어

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-05-ch6_Convolution_Network/fig3.JPG?raw=true)

이미지를 구성하는 픽셀 중 인접한 픽셀들은 비슷한 정보를 갖고 잇는 경우가 많다. 이런 이유로 이미지의 크기를 줄이면서 중요한 정보만 남기기 위해 서브샘플링이라는 기법을 사용한다. 보통 Max 풀링 레이어를 사용한다.

```python
tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))

```

<br>

### 드롭아웃 레이어

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-05-ch6_Convolution_Network/fig4.JPG?raw=true)

드롭아웃 레이어는 학습 과정에서는 확률적으로 일부 뉴런에 대한 연결을 끊고, 테스트할 대는 정상적으로 모든 값을 포함해서 계산한다. 이는 오버피팅을 막는 효과가 있다.

```python
tf.keras.layers.Dropout(rate=0.3)

```

<br>

<br>

## Fashion MNIST 데이터세트에 적용하기

Fashion MNIST 데이터를 합성곱 신경망을 이용하여 학습해본다.

<br>

### 데이터셋 불러오기 및 정규화

```python
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

train_X = train_X / 255.0
test_X = test_X / 255.0

```

<br>

Conv2D 레이어는 채널을 가진 형태의 데이터를 받도록 기본적으로 설정돼있기 대문에 채널을 갖도록 데이터의 shape를 바꾼다.

```python
# reshape 이전
print(train_X.shape, test_X.shape)

train_X = train_X.reshape(-1, 28, 28, 1)
test_X= test_X.reshape(-1, 28, 28, 1)

# reshape 이후
print(train_X.shape, test_X.shape)
#---------------------출력---------------------#
(60000, 28, 28) (10000, 28, 28)
(60000, 28, 28, 1) (10000, 28, 28, 1)

```

<br>

### 데이터 확인

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
for c in range(16):
  plt.subplot(4,4,c+1)
  plt.imshow(train_X[c].reshape(28,28), cmap='gray')

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-05-ch6_Convolution_Network/fig5.JPG?raw=true)

<br>

### 모델 생성

Conv - Pool - Conv - Pool - Conv - Flatten - Dense - Dropout - Dense 레이어로 모델을 구성한다.

```python
model = tf.keras.Sequential([
              tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), input_shape=(28,28,1)),
              tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
              tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3)),
              tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
              tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3)),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(units=128, activation='relu'),
              tf.keras.layers.Dropout(rate=0.3),
              tf.keras.layers.Dense(units=10, activation='softmax')
 ])

 model.compile(optimizer=tf.keras.optimizers.Adam(),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
 
 model.summary()
#---------------------출력---------------------#
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 3, 3, 128)         73856     
_________________________________________________________________
flatten (Flatten)            (None, 1152)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               147584    
_________________________________________________________________
dropout (Dropout)            (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 241,546
Trainable params: 241,546
Non-trainable params: 0
_________________________________________________________________

```

<br>

### 모델학습

출력된 loss, accuracy 그래프를 보면 과대적합이 의심된다.

```python
history = model.fit(train_X, train_Y, epochs=25, validation_split=0.25)

import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r--', label='val_accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-05-ch6_Convolution_Network/fig6.JPG?raw=true)

<br>

### 평가

정확도는 88% 정도 나온다. 이는 학습 데이터셋의 정확도인 95% 보다 아래로, 과대적합 되었음을 알 수 있다.

```python
model.evaluate(test_X, test_Y, verbose=0)  # verbose 0은 progress bar 없이, 결과만 출력
#---------------------출력---------------------#
[0.5032432079315186, 0.8867999911308289]

```

<br>

<br>

## 퍼포먼스 높이기

### 더 깊은 레이어 쌓기

더 깊은 레이어를 쌓으면 성능이 더 올라갈 수 있다. 여기서는 VGGNet의 스타일로 구성한 컨볼루션 신경망을 사용해 Fashion MNIST 데이터셋을 학습해본다.

<br>

#### 모델 생성

모델은 Conv - Conv - Pool - Dropout - Conv - Conv - Pool - Dropout - Flatten - Dense - Dropout - Dense - Dropout - Dense 레이어로 구성한다.

```python
model = tf.keras.Sequential([
              tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same',
                                     activation='relu', input_shape=(28, 28, 1)),
              tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'),
              tf.keras.layers.MaxPool2D(pool_size=(2,2)),
              tf.keras.layers.Dropout(rate=0.5),
              tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'),
              tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='valid', activation='relu'),
              tf.keras.layers.MaxPool2D(pool_size=(2,2)),
              tf.keras.layers.Dropout(rate=0.5),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(units=512, activation='relu'),
              tf.keras.layers.Dropout(rate=0.5),
              tf.keras.layers.Dense(units=256, activation='relu'),
              tf.keras.layers.Dropout(rate=0.5),
              tf.keras.layers.Dense(units=10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#---------------------출력---------------------#
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_3 (Conv2D)            (None, 28, 28, 32)        320       
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 28, 28, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 14, 14, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 14, 14, 64)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 14, 14, 128)       73856     
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 12, 12, 256)       295168    
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 6, 6, 256)         0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 6, 6, 256)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 9216)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               4719104   
_________________________________________________________________
dropout_3 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 256)               131328    
_________________________________________________________________
dropout_4 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 10)                2570      
=================================================================
Total params: 5,240,842
Trainable params: 5,240,842
Non-trainable params: 0
_________________________________________________________________

```

<br>

#### 모델 학습

정확도는 92% 정도 나온다. loss와 accuracy 그래프를 보면 오버피팅이 되지 않았을 것으로 예상된다.

```python
history = model.fit(train_X, train_Y, epochs=25, validation_split=0.25)

import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r--', label='val_accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-05-ch6_Convolution_Network/fig7.JPG?raw=true)

<br>

#### 평가

정확도가 92% 정도 나왔다.

```python
model.evaluate(test_X, test_Y, verbose=0)
#---------------------출력---------------------#
[0.22130292654037476, 0.9197999835014343]

```

<br>

<br>

### Data Augmentation

Data Augmentation은 훈련 데이터에 없는 이미지를 새롭게 만들어내서 훈련 데이터를 보강하는 것이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-05-ch6_Convolution_Network/fig8.JPG?raw=true)

<br>

#### 코드 구현

tf.keras에는 Data Augmentation을 쉽게 해주는 ImageDataGenerator가 있다.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

image_generator = ImageDataGenerator(
            rotation_range=10, # 회전
            zoom_range=0.10, # 확대
            shear_range=0.5, # 기울임
            width_shift_range=0.10, # 가로방향 평행 이동
            height_shift_range=0.10, # 세로방향 평행 이동
            horizontal_flip=True, # 좌우반전
            vertical_flip=False # 상하반전
)

augment_size = 100

x_augmented = \\
image_generator.flow(np.tile(train_X[0].reshape(28*28),100).reshape(-1,28,28,1),
                      np.zeros(augment_size), batch_size=augment_size, 
                      shuffle=False).next()[0]
#np.tile(A, reps)은 A를 reps에 정해진 형식만큼 반복한 값을 반환한다.

```

```python
plt.figure(figsize=(10,10))
for c in range(100):
  plt.subplot(10,10,c+1)
  plt.axis('off')
  plt.imshow(x_augmented[c].reshape(28,28), cmap='gray')
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-05-ch6_Convolution_Network/fig9.JPG?raw=true)