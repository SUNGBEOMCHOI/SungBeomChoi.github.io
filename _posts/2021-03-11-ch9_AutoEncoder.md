---
layout: post
title: (시작하세요! 텐서플로 2.0 프로그래밍) 9장. 오토인코더(AutoEncoder)
featured-img: 2021-03-11-ch9_AutoEncoder/fig1
permalink: /book_review/2021-03-11-ch9_AutoEncoder
category: book_review

---



## 인코더와 디코더, 잠재변수

오토인코더(AutoEncoder)는 자기 자신을 재생성하는 네트워크이다. 오토인코더는 크게 인코더, 잠재변수, 디코더로 나눌 수 있다. 인코더는 지금까지 컨볼루션 신경망에서 봐왔던 특징 추출기와 같은 역할을 한다. 특징 추출기는 입력 이미지에서 특징을 추출해서 일차원의 벡터로 만들었다. 이 일차원의 벡터가 바로 잠재 변수이다. 잠재 변수에는 입력 데이터가 압축돼 있다. 이 압축된 데이터를 다시 해석해서 출력 데이터, 즉 동일한 입력 데이터로 만들어주는 것이 디코더의 역할이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-11-ch9_AutoEncoder/fig1.JPG?raw=true)

<br>

## MNIST 데이터셋에 적용하기

### 데이터셋 불러오기

```python
(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.mnist.load_data()
print(train_X.shape, train_Y.shape)

train_X = train_X / 255.0
test_X = test_X / 255.0
#---------------------출력---------------------#
(60000, 28, 28) (60000,)

```

<br>

### Dense 오토인코더 모델 정의

입력과 출력의 형태가 같아야 한다. 네트워크의 첫 번째 Dense 레이어와 세번째 Dense 레이어는 뉴런의 수가 같아서 대칭을 이룬다. 이 둘은 각각 인코더와 디코더의 역할을 한다.

```python
train_X = train_X.reshape(-1, 28 * 28)
test_X = test_X.reshape(-1, 28 * 28)
print(train_X.shape, train_Y.shape)

model = tf.keras.Sequential([
            tf.keras.layers.Dense(784, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(784, activation='sigmoid')
])

model.compile(optimizer=tf.optimizers.Adam(), loss='mse')
model.summary()
#---------------------출력---------------------#
(60000, 784) (60000,)
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 784)               615440    
_________________________________________________________________
dense_1 (Dense)              (None, 64)                50240     
_________________________________________________________________
dense_2 (Dense)              (None, 784)               50960     
=================================================================
Total params: 716,640
Trainable params: 716,640
Non-trainable params: 0
_________________________________________________________________

```

<br>

### Dense 오토인코더 모델 학습

loss가 0.006정도로 작게 나왔지만 이 수치로는 제대로 학습이 된지 알기 힘들다.

```python
model.fit(train_X, train_X, epochs=10, batch_size=256)

```

<br>

### 테스트 데이터로 Dense 오토인코더의 이미지 재생성

왼쪽이 입력, 오른쪽이 재생성된 출력이다. 출력된 결과가 대체로 입력과 비슷하게 나왔다.

```python
import random
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(4,8))
for c in range(4):
  plt.subplot(4, 2, c*2+1)
  rand_index = random.randint(0, test_X.shape[0])
  plt.imshow(test_X[rand_index].reshape(28,28), cmap='gray')
  plt.axis('off')

  plt.subplot(4, 2, c*2+2)
  img = model.predict(np.expand_dims(test_X[rand_index], axis=0))
  plt.imshow(img.reshape(28,28), cmap='gray')
  plt.axis('off')

plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-11-ch9_AutoEncoder/fig2.JPG?raw=true)

<br>

### 컨볼루션 오토인코더 모델 정의

이전에는 이미지에 대한 문제에서는 컨볼루션 레이어가 효과적이라는 것을 보았다. 그렇다면 Dense대신 컨볼루션 레이어를 사용해 오토인코더를 만들어보자.

```python
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

model = tf.keras.Sequential([
          tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=(2,2), 
                          activation='relu', input_shape=(28, 28, 1)),
          tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=(2,2), 
                          activation='relu'),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dense(7*7*64, activation='relu'),
          tf.keras.layers.Reshape(target_shape=(7,7,64)),
          tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=(2,2), 
                          padding='same', activation='relu'),
          tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=2, strides=(2,2), 
                          padding='same', activation='sigmoid'),
])

model.compile(optimizer=tf.optimizers.Adam(), loss='mse')
model.summary()
#---------------------출력---------------------#
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_3 (Conv2D)            (None, 14, 14, 32)        160       
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 7, 7, 64)          8256      
_________________________________________________________________
flatten_1 (Flatten)          (None, 3136)              0         
_________________________________________________________________
dense_5 (Dense)              (None, 64)                200768    
_________________________________________________________________
dense_6 (Dense)              (None, 3136)              203840    
_________________________________________________________________
reshape_1 (Reshape)          (None, 7, 7, 64)          0         
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 14, 14, 32)        8224      
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 28, 28, 1)         129       
=================================================================
Total params: 421,377
Trainable params: 421,377
Non-trainable params: 0
_________________________________________________________________

```

<br>

### 컨볼루션 오토인코더 모델 학습

```python
model.fit(train_X, train_X, epochs=20, batch_size=256)

```

<br>

### 테스트 데이터로 컨볼루션 오토인코더의 이미지 재생성

```python
import random
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(4,8))
for c in range(4):
  plt.subplot(4, 2, c*2+1)
  rand_index = random.randint(0, test_X.shape[0])
  plt.imshow(test_X[rand_index].reshape(28,28), cmap='gray')
  plt.axis('off')

  plt.subplot(4, 2, c*2+2)
  img = model.predict(np.expand_dims(test_X[rand_index], axis=0))
  plt.imshow(img.reshape(28,28), cmap='gray')
  plt.axis('off')

plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-11-ch9_AutoEncoder/fig3.JPG?raw=true)

오른쪽의 출력 결과는 왼쪽의 원본에 비해 중간중간 각진 모습이 보인다. 이것은 네트워크의 일부분에서 값이 아예 사라지는 것을 의미한다. 마지막 레이어를 제외하면 활성화함수로는 relu를 사용했다. relu는 양수는 그대로 반환하고, 0이나 음수가 들어오면 0을 반환한다. relu를 통해 나온 출력이 0이면 다음 레이어에서는 가중치의 효과를 모두 0으로 만들어 버린다. 그리고 디코더에서는 출력의 사이즈가 점점커진다. 즉 0인 부분이 확대된다는 의미이다.

이런 문제를 해결하기 위해 Relu와 비슷하지만 음수를 받았을 때 0보다 조금 작은 음수를 출력하는 elu가 고안되었다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-11-ch9_AutoEncoder/fig4.JPG?raw=true)

<br>

### 활성화함수를 elu로 바꾼 컨볼루션 오토인코더 모델의 정의 및 학습

```python
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

model = tf.keras.Sequential([
          tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=(2,2), 
                          activation='elu', input_shape=(28, 28, 1)),
          tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=(2,2), 
                          activation='elu'),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(64, activation='elu'),
          tf.keras.layers.Dense(7*7*64, activation='elu'),
          tf.keras.layers.Reshape(target_shape=(7,7,64)),
          tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=(2,2), 
                          padding='same', activation='elu'),
          tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=2, strides=(2,2), 
                          padding='same', activation='sigmoid'),
])

model.compile(optimizer=tf.optimizers.Adam(), loss='mse')

model.fit(train_X, train_X, epochs=20, batch_size=256)

```

<br>

### 활성화함수가 elu인 컨볼루션 오토인코더의 이미지 재생성

이전과 같은 각진 모습을 찾아볼 수 없다. 활성화함수 하나일 뿐인데 변화가 크다. 하이퍼 파라미터의 튜닝을 잘 해야겠다.

```python
import random
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(4,8))
for c in range(4):
  plt.subplot(4, 2, c*2+1)
  rand_index = random.randint(0, test_X.shape[0])
  plt.imshow(test_X[rand_index].reshape(28,28), cmap='gray')
  plt.axis('off')

  plt.subplot(4, 2, c*2+2)
  img = model.predict(np.expand_dims(test_X[rand_index], axis=0))
  plt.imshow(img.reshape(28,28), cmap='gray')
  plt.axis('off')

plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-11-ch9_AutoEncoder/fig5.JPG?raw=true)

<br>

<br>

## 클러스터링

클러스터링은 데이터를 여러개의 군집으로 묶는 것으로, 비지도학습의 한 종류이다.

<br>

## K-평균 클러스터링

K-평균 클러스터링은 주어진 입력 중 K개의 클러스터 중심을 임의로 정한 다음에 각 데이터와 K개의 중심과의 거리를 비교해서 가장 가까운 클러스터로 배당하고, K개의 중심의 위치를 해당 클러스터로 옮긴 후, 이를 반복하는 알고리즘이다.

<br>

### 모델 정의 및 실행

MNIST에서도 K-평균 클러스터링을 적용할 수 있다. 28*28의 이미지에서 크기가 64인 잠재변수를 뽑아온다.

```python
latent_vector_model = tf.keras.Sequential([
          tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=(2,2), 
                          activation='elu', input_shape=(28, 28, 1)),
          tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=(2,2), 
                          activation='elu'),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(64, activation='elu'),
])
latent_vector = latent_vector_model.predict(train_X) # 잠재변수 분리
print(latent_vector.shape)
#---------------------출력---------------------#
(60000, 64)

```

<br>

### K-평균 클러스터링 알고리즘 사용

뽑아온 잠재변수에 K-평균 클러스터링 알고리즘을 적용한다. 사이킷런 라이브러리의 KMeans를 활용한다. 총 10개의 라벨로 나누고, random_state를 통해 항상 같은 결과가 나오도록 한다. n_init은 알고리즘의 실행횟수로, 여기서는 10을 입력했기 때문에 중심의 위치를 다르게 선택해서 10번 테스트한 뒤 가장 좋은 결과를 저장한다.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, n_init=10, random_state=42)
kmeans.fit(latent_vector)

print(kmeans.labels_) # 데이터가 어떤 라벨로 분류되었는지
print(kmeans.cluster_centers_.shape) # cluster 중심들의 shape
#---------------------출력---------------------#
[9 7 6 ... 9 5 6]
(10, 64)

```

<br>

### 클러스터링 결과 출력

각 행은 0번 클러스터, 1번 클러스터, ..., 9번 클러스터를 나타낸다. 각 숫자를 잘 분류한 결과도 보이지만, 그렇지 않은 결과도 존재한다. 잠재변수의 차원 수를 늘리거나 KMeans()의 n_init을 늘려서 좀 더 분류가 잘 되도록 시도해볼 수 있다.

```python
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12,12))

for i in range(10):
  images = train_X[kmeans.labels_ == i]
  for c in range(10):
    plt.subplot(10, 10, i*10+c+1)
    plt.imshow(np.array(images[c]).reshape(28,28), cmap='gray')
    plt.axis('off')
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-11-ch9_AutoEncoder/fig6.JPG?raw=true)

<br>

<br>

## t-SNE

클러스터링 결과를 출력하려면 결국 2차원, 혹은 3차원으로 잠재변수가 가진 차원을 축소해야한다. t-SNE는 고차원의 데이터를 저차원의 시각화를 위한 데이터로 변환한다. t-SNE는 각 데이터의 유사도를 정의하고, 원래 공간에서의 유사도와 저차원 공간에서의 유사도가 비슷해지도록 학습시킨다.

t-분포는 정규분포와 비슷하게 생겼지만 중심이 좀 더 낮고 꼬리가 좀 더 두꺼운 분포이다. 거리를 확률로 표현한다는 것은 데이터 하나를 중심으로 다른 데이터를 거리에 대한 t-분포의 확률로 치환시키는 것이다. 가까운 거리의 데이터는 확률값이 높아지고, 먼 거리의 데이터는 확률값이 낮아진다. 고차원과 저차원에서 확률값을 각각 구한 다음, 저차원의 확률값이 고차원에 가까워지도록 학습시키는 것이 t-SNE 알고리즘의 주요 내용이다.

<br>

### 사이킷런의 t-SNE 사용

TNSE의 인수 중 처음에 나오는 n_components는 저차원의 수를 의미한다. learning_rate는 학습률로 10에서 1000사이의 큰 숫자를 넣는다. perplexity는 알고리즘 계산에서 고려할 최근접 이웃의 숫자이다. 보통 5에서 50 사이의 숫자를 넣는다.

결과값이 잘 뭉쳐있으나 이것으로는 숫자별로 잘 분리되었는지는 알기힘들다.

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, learning_rate=100, perplexity=15, random_state=0)
tsne_vector = tsne.fit_transform(latent_vector[:5000])

cmap = plt.get_cmap('rainbow', 10)
fig = plt.scatter(tsne_vector[:,0], tsne_vector[:,1], marker='.', c=train_Y[:5000], cmap=cmap)
cb = plt.colorbar(fig, ticks=range(10))
n_clusters=10
tick_locs=(np.arange(n_clusters)+0.5)*(n_clusters-1)/n_clusters

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-11-ch9_AutoEncoder/fig7.JPG?raw=true)

<br>

### t-SNE 클러스터 위에 MNIST 이미지 표시

더 직관적으로 결과를 살펴보기위해 결과값위에 MNIST이미지를 표시해본다. AnnotationBox는 이미지나 텍스트 등의 주석을 그래프 위에 표시하기 위한 주석상자를 그리는 함수이다. 출력 이미지를 보면 각 숫자는 대부분 자신이 속한 클러스터에 표시되고 있다. t-SNE가 데이터를 효율적으로 압축하여 시각화함을 알 수 있다.

```python
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

tsne = TSNE(n_components=2, perplexity=15, random_state=0)
tsne_vector = tsne.fit_transform(latent_vector[:5000])

plt.figure(figsize=(16,16))
ax = plt.subplot(1,1,1)
ax.scatter(tsne_vector[:,0], tsne_vector[:,1], marker='.', c=train_Y[:5000], cmap='rainbow')
for i in range(200):
  imagebox = OffsetImage(np.array(train_X[i]).reshape(28,28))
  ab = AnnotationBbox(imagebox, (tsne_vector[i,0], tsne_vector[i,1]), frameon=False, pad=0.0)
  ax.add_artist(ab)

ax.set_xticks([])
ax.set_yticks([])
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-11-ch9_AutoEncoder/fig8.JPG?raw=true)

<br>

<br>

## 초해상도 이미지 얻기

픽셀로 구성된 이미지는 고해상도로 변환하면 이미지에서 사각형이 두드러져 보이는, 소위 픽셀이 깨지는 현상이 일어나게 된다. 이를 자연스러운 고해상도 이미지로 만들어주는 것이 바로 초해상도(super Resolution) 작업이다.

오토인코더를 활용해서 초해상도 작업을 하는 네트워크를 학습시킬 수 있다. 여기서는 REDNet이라는 네트워크를 사용한다. REDNet은 Residual Encoder-Decoder Network의 약자로, 인코더와 디코더는 오토인코더에서 봤고, Residual은 ResNet 등에서 사용하는 skip-connection이다.

<br>

### 데이터셋 불러오기

먼저 학습을 할 이미지를 불러와야한다. 여기서는 BSD(Berkeley Segmentation Dataset)을 사용한다. 압축이 해제된 파일들은 /content/images 폴더 아래에 저장된다.

```python
tf.keras.utils.get_file('/content/bsd_images.zip', '<http://bit.ly/35pHZlC>', extract=True)

!unzip /content/bsd_images.zip

```

<br>

### 이미지 경로 저장 및 확인

```python
import pathlib

image_root = pathlib.Path('/content/images')

all_image_paths = list(image_root.glob('*/*'))
print(all_image_paths[:10])
#---------------------출력---------------------#
[PosixPath('/content/images/train/173036.jpg')
PosixPath('/content/images/train/8049.jpg')
PosixPath('/content/images/train/117054.jpg')
PosixPath('/content/images/train/202012.jpg')
PosixPath('/content/images/train/326038.jpg')
PosixPath('/content/images/train/198054.jpg')
PosixPath('/content/images/train/113009.jpg')
PosixPath('/content/images/train/246053.jpg')
PosixPath('/content/images/train/176019.jpg')
PosixPath('/content/images/train/246016.jpg')]

```

<br>

### 데이터를 [tf.data](http://tf.data).Dataset로 변환

BSD500은 200장의 훈련 데이터, 100장의 검증 데이터, 200장의 테스트 데이터로 구성돼 있다. 각 데이터셋 집합을 처리하기 위한 [tf.data](http://tf.data).Dataset를 각 데이터셋마다 만든다. 이를 위해서는 몇 가지 작업이 필요하다.

<br>

#### 이미지 경로 분리 저장

```python
train_path, valid_path, tets_path = [], [], []

for image_path in all_image_paths:
  if str(image_path).split('.')[-1] != 'jpg':
    continue

  if str(image_path).split('/')[-2] == 'train':
    train_path.append(str(image_path))
  elif str(image_path).split('/')[-2] == 'val':
    valid_path.append(str(image_path))
  else:
    test_path.append(str(image_path))

```

<br>

#### 원본 이미지에서 조각을 추출하고 입력, 출력 데이터를 반환하는 함수 정의

현재 가지고 있는 이미지는 고해상도이기 때문에 이미지의 해상도를 일부러 낮추고 원본과 함께 반환하는 함수를 만든다. 또 학습 효율을 높이기 위해 이미지를 받아온 다음 이미지에서 가로 * 세로 50픽셀의 작은 조각(patch)을 잘라서 학습에 사용한다. 저해상도의 이미지는 입력에 사용되고, 고해상도의 이미지는 출력에 사용된다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-11-ch9_AutoEncoder/fig9.JPG?raw=true)

```python
def get_hr_and_lr(image_path):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  hr = tf.image.random_crop(img, [50, 50, 3])
  lr = tf.image.resize(hr, [50, 50])
  lr = tf.image.resize(lr, [50, 50])
  return lr, hr

```

<br>

#### train, valid Dataset 정의

```python
train_dataset = tf.data.Dataset.list_files(train_path)
train_dataset = train_dataset.map(get_hr_and_lr)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(16)

valid_dataset = tf.data.Dataset.list_files(valid_path)
valid_dataset = valid_dataset.map(get_hr_and_lr)
valid_dataset = valid_dataset.repeat()
valid_dataset = valid_dataset.batch(16)

```

<br>

### REDNet 네트워크 정의

모델의 구조가 직선형이 아니기 때문에 시퀀셜 모델이 아닌 함수형 API를 사용해서 네트워크를 정의해야한다. 레이어의 수를 인수로 받는 REDNet()함수를 만들어서 다양한 층을 갖는 네트워크를 함수 호출 한 번으로 만들 수 있게 한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-11-ch9_AutoEncoder/fig14.JPG?raw=true)

num_layers는 컨볼루션 레이어와 디컨볼루션 레이어의 수이다. 같은 수의 컨볼루션 레이어와 디컨볼루션 레이어가 존재하기 때문에 REDNet-30이라면 num_layer에 15를 넣으면 된다. 입력 레이어의 shape에서는 이미지의 높이와 너비를 None으로 지정해서 어떤 크기의 이미지라도 입력으로 받을 수 있게 한다.

이후 인코더에서는 짝수번째 컨볼루션 레이어를 지날 때마다 x를 잔류 레이어 리스트에도 저장한다. 디코더에서는 홀수 번째의 디컨볼루션 레이어를 통과할 때마다 잔류 레이어 리스트에 저장돼 있던 값을 residual_layers.pop()으로 뒤에서부터 하나씩 가져온다.

```python
def REDNet(num_layers):
  conv_layers = []
  deconv_layers = []
  residual_layers = []

  inputs = tf.keras.layers.Input(shape=(None, None, 3))
  conv_layers.append(tf.keras.layers.Conv2D(3, kernel_size=3, padding='same', activation='relu'))

  for i in range(num_layers-1):
    conv_layers.append(tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    deconv_layers.append(tf.keras.layers.Conv2DTranspose(64, kernel_size=3, padding='same', activation='relu'))

  deconv_layers.append(tf.keras.layers.Conv2DTranspose(3, kernel_size=3, padding='same'))

  # 인코더 시작
  x = conv_layers[0](inputs)

  for i in range(num_layers-1):
    x = conv_layers[i+1](x)
    if i % 2 == 0:
      residual_layers.append(x)

  # 디코더 시작
  for i in range(num_layers-1):
    if i % 2 ==1:
      x = tf.keras.layers.Add()([x, residual_layers.pop()])
      x = tf.keras.layers.Activation('relu')(x)
    x = deconv_layers[i](x)

  x = deconv_layers[-1](x)

  model = tf.keras.Model(inputs=inputs, outputs=x)
  return model

```

<br>

### PSNR 정의 및 모델 컴파일

이제 모델이 정의되었고, 평가방법을 생각해야한다. 보통 PSNR(Peak Signeal-to-Noise Ratio), 즉 "신호 대 잡음비"를 사용한다. 원본 이미지와 재구성된 이미지의 PSNR을 계산하면 이미지의 품질이 얼마나 좋은지를 측정할 수 있다. 동일한 2개의 이미지를 PSNR로 계산하면 무한대의 값이 나오고, 보통 30이상이면 좋은 값이다. 평균 제곱 오차가 분모에 있기 때문에 평균 제곱 오차가 낮을수록 큰 값을 갖게된다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-11-ch9_AutoEncoder/fig10.JPG?raw=true)

```python
def psnr_metric(y_true, y_pred):
  return tf.image.psnr(y_true, y_pred, max_val=1.0)

model = REDNet(15)
model.compile(optimizer=tf.optimizers.Adam(0.0001), loss='mse', metrics=[psnr_metric])

```

<br>

### REDNet-30 모델 학습

학습데이터를 총 500번 돌며 학습한다. steps_per_epoch을 통해 batch를 몇번 거쳐야 1epoch인지 넣어준다.

```python
history = model.fit(train_dataset,
                    epochs=500,
                    steps_per_epoch=len(train_path)//16,
                    validation_data=valid_dataset,
                    validation_steps=len(valid_path),
                    verbose=2)

```

<br>

### 학습 그래프 확인

학습결과 psnr이 약 33정도 나오고, 오버피팅은 없는듯하다.

```python
import matplotlib.pyplot as plt
plt.plot(history.history['psnr_metric'], 'b-', label='psnr')
plt.plot(history.history['val_psnr_metric'], 'r--', label='val_psnr')
plt.xlabel('Epoch')

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-11-ch9_AutoEncoder/fig11.JPG?raw=true)

<br>

### 평가

학습된 네트워크가 테스트 이미지를 어떻게 복원하는지 확인해본다. predict-hr의 psnr과 lr-hr의 psnr이 비슷하게 나온다. 이것만 보면 학습이 안된 것 같으나 실제 이미지를 출력해보면 학습이 되었다는 사실을 알 수 있다.

```python
img = tf.io.read_file(test_path[0])
img = tf.image.decode_jpeg(img, channels=3)
hr = tf.image.convert_image_dtype(img, tf.float32)

lr = tf.image.resize(hr, [hr.shape[0]//2, hr.shape[1]//2])
lr = tf.image.resize(lr, [hr.shape[0], hr.shape[1]])
predict_hr = model.predict(np.expand_dims(lr, axis=0))

print(tf.image.psnr(np.squeeze(predict_hr, axis=0), hr, max_val=1.0))
print(tf.image.psnr(lr, hr, max_val=1.0))

plt.figure(figsize=(16,4))

plt.subplot(1, 3, 1)
plt.imshow(hr)
plt.title('original - hr')

plt.subplot(1, 3, 2)
plt.imshow(lr)
plt.title('lr')

plt.subplot(1, 3, 3)
plt.imshow(np.squeeze(predict_hr, axis=0))
plt.title('sr')

plt.show()
#---------------------출력---------------------#
tf.Tensor(25.893595, shape=(), dtype=float32)
tf.Tensor(25.849056, shape=(), dtype=float32)

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-11-ch9_AutoEncoder/fig12.JPG?raw=true)

<br>

#### Set5 데이터셋을 통한 평가

Super Resolution 기술의 성능을 비교하기 위한 벤치마크 중 유명한 것으로 set5라는 데이터셋이있다. 이중에서도 자주 쓰이는 나비의 사진을 통해 평가해본다. 여기서는 확실히 psnr의 차이가 있고, 이미지에서도 차이를 보인다.

```python
image_path = tf.keras.utils.get_file('butterfly.png', '<http://bit.ly/2oAOxgH>')
img = tf.io.read_file(image_path)
img = tf.image.decode_jpeg(img, channels=3)
hr = tf.image.convert_image_dtype(img, tf.float32)

lr = tf.image.resize(hr, [hr.shape[0]//2, hr.shape[1]//2])
lr = tf.image.resize(lr, [hr.shape[0], hr.shape[1]])
predict_hr = model.predict(np.expand_dims(lr, axis=0))

print(tf.image.psnr(np.squeeze(predict_hr, axis=0), hr, max_val=1.0))
print(tf.image.psnr(lr, hr, max_val=1.0))

plt.figure(figsize=(16,6))
plt.subplot(1, 3, 1)
plt.imshow(hr)
plt.title('original - hr')

plt.subplot(1, 3, 2)
plt.imshow(lr)
plt.title('lr')

plt.subplot(1, 3, 3)
plt.imshow(np.squeeze(predict_hr, axis=0))
plt.title('sr')

tf.Tensor(30.224031, shape=(), dtype=float32)
tf.Tensor(24.783773, shape=(), dtype=float32)
#---------------------출력---------------------#
tf.Tensor(30.224031, shape=(), dtype=float32)
tf.Tensor(24.783773, shape=(), dtype=float32)

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-11-ch9_AutoEncoder/fig13.JPG?raw=true)