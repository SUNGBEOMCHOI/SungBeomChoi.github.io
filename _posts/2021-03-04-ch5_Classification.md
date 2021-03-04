---
layout: post
title: (시작하세요! 텐서플로 2.0 프로그래밍) 5장. 분류(Classification)
featured-img: 2021-03-04-ch5_Classification/fig5
permalink: /book_review/2021-03-04-ch5_Classification
category: book_review

---

## 이항 분류

이항분류는 정답의 범주가 두 개인 분류 문제이다.
여기서는 와인의 당도, 산도, 알코올 도수 등의 데이터를 통해 레드인지 화이트와인인지 구분하도록 하겠다.


### 데이터셋 불러오기

캘리포니아 어바인 대학에서 제공하는 와인 데이터셋을 불러온다.

```python
import pandas as pd

red = pd.read_csv('<http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv>', sep=';')
white = pd.read_csv('<http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv>', sep=';')
print(red.head())
print(white.head())

#---------------------출력---------------------#
fixed acidity  volatile acidity  citric acid  ...  sulphates  alcohol  quality
0            7.4              0.70         0.00  ...       0.56      9.4        5
1            7.8              0.88         0.00  ...       0.68      9.8        5
2            7.8              0.76         0.04  ...       0.65      9.8        5
3           11.2              0.28         0.56  ...       0.58      9.8        6
4            7.4              0.70         0.00  ...       0.56      9.4        5

[5 rows x 12 columns]
   fixed acidity  volatile acidity  citric acid  ...  sulphates  alcohol  quality
0            7.0              0.27         0.36  ...       0.45      8.8        6
1            6.3              0.30         0.34  ...       0.49      9.5        6
2            8.1              0.28         0.40  ...       0.44     10.1        6
3            7.2              0.23         0.32  ...       0.40      9.9        6
4            7.2              0.23         0.32  ...       0.40      9.9        6

[5 rows x 12 columns]

```

<br>

### 레드와인인지 화이트화인인지 표시하는 속성추가 및 두 데이터 프레임 합치기

아래 결과를 보면 type의 평균이 0.75가 나오는 것을 보면 0에 해당하는 값보다 1에 해당하는 값이 더 많을 것으로 짐작할 수 있다.

```python
red['type'] = 0
white['type'] = 1

wine = pd.concat([red, white])
print(wine.describe())

#---------------------출력---------------------#
fixed acidity  volatile acidity  ...      quality         type
count    6497.000000       6497.000000  ...  6497.000000  6497.000000
mean        7.215307          0.339666  ...     5.818378     0.753886
std         1.296434          0.164636  ...     0.873255     0.430779
min         3.800000          0.080000  ...     3.000000     0.000000
25%         6.400000          0.230000  ...     5.000000     1.000000
50%         7.000000          0.290000  ...     6.000000     1.000000
75%         7.700000          0.400000  ...     6.000000     1.000000
max        15.900000          1.580000  ...     9.000000     1.000000

[8 rows x 13 columns]

```

<br>

### type 히스토그램

아래 그래프를 통해 확실하게 1에 해당하는 값이 더 많다는 것을 확인할 수 있다.

```python
import matplotlib.pyplot as plt

plt.hist(wine['type'])
plt.xticks([0, 1])
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-04-ch5_Classification/fig1.JPG?raw=true)

<br>

### 데이터 정규화

또 데이터 정규화를 해주어야한다. 정규화 전에 데이터가 어떤 값으로 구성되어 있는지 알아본다.

```python
print(wine.info())
#---------------------출력---------------------#
<class 'pandas.core.frame.DataFrame'>
Int64Index: 6497 entries, 0 to 4897
Data columns (total 13 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   fixed acidity         6497 non-null   float64
 1   volatile acidity      6497 non-null   float64
 2   citric acid           6497 non-null   float64
 3   residual sugar        6497 non-null   float64
 4   chlorides             6497 non-null   float64
 5   free sulfur dioxide   6497 non-null   float64
 6   total sulfur dioxide  6497 non-null   float64
 7   density               6497 non-null   float64
 8   pH                    6497 non-null   float64
 9   sulphates             6497 non-null   float64
 10  alcohol               6497 non-null   float64
 11  quality               6497 non-null   int64  
 12  type                  6497 non-null   int64  
dtypes: float64(11), int64(2)
memory usage: 710.6 KB
None

```

<br>

데이터의 type이 모두 숫자값이기 때문에 정규화를 진행할 수 있다. 최댓값이 1, 최솟값이 0이 되도록 정규화를 진행한다.

결과를 보면 max와 min값이 1과 0으로 바뀐것을 볼 수 있다.

```python
wine_norm = (wine - wine.min()) / (wine.max() - wine.min())
print(wine_norm.describe())

#---------------------출력---------------------#
fixed acidity  volatile acidity  ...      quality         type
count    6497.000000       6497.000000  ...  6497.000000  6497.000000
mean        0.282257          0.173111  ...     0.469730     0.753886
std         0.107143          0.109758  ...     0.145543     0.430779
min         0.000000          0.000000  ...     0.000000     0.000000
25%         0.214876          0.100000  ...     0.333333     1.000000
50%         0.264463          0.140000  ...     0.500000     1.000000
75%         0.322314          0.213333  ...     0.500000     1.000000
max         1.000000          1.000000  ...     1.000000     1.000000

[8 rows x 13 columns]

```

<br>

### 데이터 섞기

딥러닝 학습을 위해 데이터를 훈련 데이터와 테스트 데이터로 나누기 전에 레드 와인과 화이트 와인이 비슷한 비율로 들어가도록 데이터를 한 번 랜덤하게 섞어야한다.

판다스의 sample() 함수는 전체 데이터프레임에서 frac 인수로 지정된 비율만큼의 행을 랜덤하게 뽑아서 새로운 데이터프레임을 만든다.

```python
import numpy as np

wine_shuffle = wine_norm.sample(frac=1)
wine_np = wine_shuffle.to_numpy() # 넘파이 array로 변환

```

<br>

### 훈련 데이터와 테스트 데이터 분리

학습을 위해 훈련 데이터와 테스트 데이터를 분리한다. 비율은 8:2로 나누었다. 나눈후 Y데이터를 원-핫 인코딩 형식으로 변환한다. tf.utils의 to_categorical 함수는 num_calasses 만큼의 인덱스 수로 원핫인코딩을 진행해준다.

아래 코드의 출력을 보면 train_X[0]의 경우 특성 12개를 가지고 있고, train_Y[0]는 원-핫 인코딩된 모습을 볼 수 있다.

```python
import tensorflow as tf

train_idx = int(len(wine_np) * 0.8)
train_X, train_Y = wine_np[:train_idx, :-1], wine_np[:train_idx, -1] 
test_X, test_Y = wine_np[train_idx:, :-1], wine_np[train_idx:, -1]
train_Y = tf.keras.utils.to_categorical(train_Y, num_classes=2)  # 원-핫 인코딩
test_Y = tf.keras.utils.to_categorical(test_Y, num_classes=2)
print(train_X[0])
print(train_Y[0])
#---------------------출력---------------------#
[0.24793388 0.08       0.1686747  0.18404908 0.06478405 0.18402778
 0.29953917 0.16290727 0.36434109 0.08426966 0.39130435 0.5       ]
[0. 1.]

```

<br>

### 모델 생성

모델은 총 4개의 층을 갖는다. 첫 layer의 input_shape는 특성이 12개임을 반영했다. 마지막 layer의 경우 activation을 분류 문제에 맞게 확률을 출력하는 softmax를 사용하였다.

optimizer는 Adam을 사용했고, loss는 분류에 맞게 cross entropy로 해주었다. metrics에는 accuracy를 넣어서 학습시 loss 뿐만 아니라 accuracy도 기록하도록 했다.

```python
model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=48, activation='relu', input_shape=(12,)),
            tf.keras.layers.Dense(units=24, activation='relu'),
            tf.keras.layers.Dense(units=12, activation='relu'),
            tf.keras.layers.Dense(units=2, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
#---------------------출력---------------------#
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 48)                624       
_________________________________________________________________
dense_1 (Dense)              (None, 24)                1176      
_________________________________________________________________
dense_2 (Dense)              (None, 12)                300       
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 26        
=================================================================
Total params: 2,126
Trainable params: 2,126
Non-trainable params: 0
_________________________________________________________________

```

<br>

### 모델 학습

트레이닝 결과 모두 정확도가 100%에 가까운 결과를 보인다..

```python
history = model.fit(train_X, train_Y, epochs=25, batch_size=32, validation_split=0.25)
#---------------------출력---------------------#
Epoch 1/25
122/122 [==============================] - 1s 4ms/step - loss: 0.2468 - accuracy: 0.9056 - val_loss: 0.1578 - val_accuracy: 0.9631
Epoch 2/25
122/122 [==============================] - 0s 2ms/step - loss: 0.0946 - accuracy: 0.9707 - val_loss: 0.0386 - val_accuracy: 0.9908
...
(중간생략)
...
Epoch 24/25
122/122 [==============================] - 0s 2ms/step - loss: 0.0530 - accuracy: 0.9880 - val_loss: 0.0320 - val_accuracy: 0.9908
Epoch 25/25
122/122 [==============================] - 0s 2ms/step - loss: 0.0359 - accuracy: 0.9899 - val_loss: 0.0434 - val_accuracy: 0.9915

```

<br>

### 그래프 그리기

```python
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legent()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r--', label='val_accuracy')
plt.xlabel('Epoch')
plt.legent()

plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-04-ch5_Classification/fig2.JPG?raw=true)

<br>

### 평가

테스트 셋으로 성능을 평가한다. 아래줄의 출력은 loss와 accuracy 값이다. 거의 100%에 가까운 정확도를 보인다.

```python
model.evaluate(test_X, test_Y)
#---------------------출력---------------------#
41/41 [==============================] - 0s 833us/step - loss: 0.0527 - accuracy: 0.9938
[0.052699703723192215, 0.9938461780548096]

```

<br>

<br>

## 다항 분류

다항 분류란 category의 수가 2개를 초과하는 경우이다. 와인 데이터셋에서 앞처럼 type의 값을 예측하려는 값으로 사용하는 대신 0~10까지 숫자로 분류되어 있는 quality를 예측해본다.


### quality 데이터 확인

품질 데이터의 정보와 각 카테고리의 수를 확인해본다.

확인해보니 min값이 3, max 값이 9이다. 또 value_counts함수로 확인해보니 각 항목의 수의 차이가 있다.

```python
print(wine['quality'].describe())
print(wine['quality'].value_counts())
#---------------------출력---------------------#
count    6497.000000
mean        5.818378
std         0.873255
min         3.000000
25%         5.000000
50%         6.000000
75%         6.000000
max         9.000000
Name: quality, dtype: float64
6    2836
5    2138
7    1079
4     216
8     193
3      30
9       5
Name: quality, dtype: int64

```

<br>

#### 그래프 출력

```python
plt.hist(wine['quality'], bins=7)
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-04-ch5_Classification/fig3.JPG?raw=true)

<br>

### qulaity 범주를 재구성

데이터의 수 차이가 크고, 세세한 분류가 어려울 듯하다. 따라서 quality 3~5는 나쁨, 6은 보통, 7~9는 좋음으로 재구성 한다.

```python
wine.loc[wine['quality'] <= 5, 'new_quality'] = 0
wine.loc[wine['quality'] == 6, 'new_quality'] = 1
wine.loc[wine['quality'] >= 7, 'new_quality'] = 2
print(wine['new_quality'].value_counts())
#---------------------출력---------------------#
1.0    2836
0.0    2384
2.0    1277
Name: new_quality, dtype: int64

```

<br>

### 데이터 정규화 및 훈련 데이터, 테스트 데이터 분류

```python
del wine['quality'] # quality 데이터 삭제
wine_norm = (wine - wine.min()) / (wine.max() - wine.min())
wine_shuffle = wine_norm.sample(frac=1)
wine_np = wine_shuffle.to_numpy()

train_idx = int(len(wine_np) * 0.8)
train_X, train_Y = wine_np[:train_idx, :-1], wine_np[:train_idx, -1]
test_X, test_Y = wine_np[train_idx:, :-1], wine_np[train_idx:, -1]
train_Y = tf.keras.utils.to_categorical(train_Y, num_classes=3)
test_Y = tf.keras.utils.to_categorical(test_Y, num_classes=3)

```

<br>

### 모델 생성 및 학습

학습 데이터의 accuracy를 보면 80%정도이다.

```python
model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=48, activation='relu', input_shape=(12, )),
            tf.keras.layers.Dense(units=24, activation='relu'),
            tf.keras.layers.Dense(units=12, activation='relu'),
            tf.keras.layers.Dense(units=3, activation='softmax'),
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_X, train_Y, epochs=25, batch_size=32, validation_split=0.25)
#---------------------출력---------------------#
Epoch 1/25
122/122 [==============================] - 1s 3ms/step - loss: 0.5348 - accuracy: 0.8014 - val_loss: 0.4162 - val_accuracy: 0.8162
Epoch 2/25
122/122 [==============================] - 0s 2ms/step - loss: 0.4291 - accuracy: 0.7940 - val_loss: 0.3976 - val_accuracy: 0.8008
...
(중간 생략)
...
Epoch 24/25
122/122 [==============================] - 0s 2ms/step - loss: 0.3944 - accuracy: 0.8351 - val_loss: 0.3837 - val_accuracy: 0.8262
Epoch 25/25
122/122 [==============================] - 0s 2ms/step - loss: 0.4017 - accuracy: 0.8266 - val_loss: 0.3872 - val_accuracy: 0.8323

```

<br>

### 학습 기록 그래프

```python
plt.figure(figsize=(12, 4))

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

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-04-ch5_Classification/fig4.JPG?raw=true)

<br>

### 평가

테스트 셋으로 학습된 모델을 평가해보겠습니다. 정확도가 81% 정도 나옵니다.

```python
model.evaluate(test_X, test_Y)
#---------------------출력---------------------#
41/41 [==============================] - 0s 917us/step - loss: 0.3953 - accuracy: 0.8162
[0.3953031599521637, 0.8161538243293762]

```

<br>

<br>

## Fashion MNIST

Fashion MNIST는 옷, 신발, 가방의 이미지들을 모아놓았다. 범주가 10개라는 점과 각 이미지의 크기가 28*28 픽셀이라는점은 MNIST와 동일하지만 좀 더 어려운 문제로 평가된다.


### 데이터 가져오기

학습 데이터의 갯수는 60000개, 시험 데이터의 갯수는 10000개이다.

```python
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

print(len(train_X), len(test_X))
#---------------------출력---------------------#
60000 10000

```

<br>

### 데이터 확인

plt의 imshow를 이용해 이미지를 보겠다.

```python
import matplotlib.pyplot as plt

plt.imshow(train_X[0], cmap='gray')
plt.colorbar()
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-04-ch5_Classification/fig5.JPG?raw=true)

<br>

### 정규화

위의 그림의 colorbar를 확인하면 데이터의 이미지가 0~255 까지의 값을 가진다는 것을 확인할 수 있다. 또 축의 값을 보면 28*28 픽셀의 이미지임을 확인할 수 있다. 따라서 0~255의 값을 0~1로 정규화해주겠다.

```python
train_X = train_X / 255
test_X = test_X / 255

```

<br>

### 모델 생성

이전에는 분류하려고 하는 값을 원-핫 인코딩으로 바꾸는 부분이 있었다. 하지만 표현을 원하는 라벨은 1개인데, 이를 표현하기 위해서는 10개의 숫자가 필요하다. 이런 대부분의 값이 0인 행렬을 희소 행렬(sparse matrix)라고 한다. 희소 행렬에서 모두 0으로 표현하는 것은 메모리의 낭비이다. 따라서 별도의 변환은 하지 않는다.

원-핫 인코딩이 아닌 데이터를 받아서 계산하기 위해 모델에서 수정이 필요하다. loss를 sparse_categorical_crossentropy로 설정해주는 것이다.

여기서 모델은 간단하게 3층으로 구성한다. 첫 번째 layer에서는 flatten에서는 28*28 픽셀의 이미지를 쭉 펴준다. optimizer는 Adam을 사용했는데 학습률은 기본값인 0.001을 적용했다.

```python
model = tf.keras.Sequential([
          tf.keras.layers.Flatten(input_shape=(28,28)),
          tf.keras.layers.Dense(units=128, activation='relu'),
          tf.keras.layers.Dense(units=10, activation='softmax'),
])

model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
#---------------------출력---------------------#
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
dense_12 (Dense)             (None, 128)               100480    
_________________________________________________________________
dense_13 (Dense)             (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________

```

<br>

### 학습

최종적인 정확도는 훈련 데이터에서는 94%, validation 데이터에서는 88%가 나온다. 또 validation의 정확도는 어느순간 더 증가하지 않는다. 오버피팅이 의심된다.

```python
history = model.fit(train_X, train_Y, epochs=25, validation_split=0.25)
#---------------------출력---------------------#
Epoch 1/25
1407/1407 [==============================] - 3s 2ms/step - loss: 0.6620 - accuracy: 0.7722 - val_loss: 0.4110 - val_accuracy: 0.8533
Epoch 2/25
1407/1407 [==============================] - 3s 2ms/step - loss: 0.3993 - accuracy: 0.8586 - val_loss: 0.3831 - val_accuracy: 0.8621
...
(중간 생략)
...
Epoch 24/25
1407/1407 [==============================] - 3s 2ms/step - loss: 0.1612 - accuracy: 0.9400 - val_loss: 0.3463 - val_accuracy: 0.8925
Epoch 25/25
1407/1407 [==============================] - 3s 2ms/step - loss: 0.1529 - accuracy: 0.9427 - val_loss: 0.3642 - val_accuracy: 0.8877

```

<br>

### 학습 결과 시각화

아래 그래프를 보면 확실히 오버피팅 되었음을 확인할 수 있다. 4장에 사용했던 Early Stopping을 사용하면 좋을 듯 하다.

```python
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r--', label='val_accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-04-ch5_Classification/fig6.JPG?raw=true)

<br>

### 평가

정확도가 88% 가 나온다. 확실히 훈련 데이터의 정확도인 94% 보다는 작다.

```python
model.evaluate(test_X, test_Y)
#---------------------출력---------------------#
313/313 [==============================] - 0s 1ms/step - loss: 0.4070 - accuracy: 0.8836
[0.40704625844955444, 0.8835999965667725]

```