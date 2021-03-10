---
layout: post
title: (시작하세요! 텐서플로 2.0 프로그래밍) 8장. 사전 훈련된 모델 다루기(전이 학습)
featured-img: 2021-03-10-ch8_Pre_Trained_Model/fig3
permalink: /book_review/2021-03-10-ch8_Pre_Trained_Model
category: book_review

---


좋은 성능을 보이는 네트워크는 수백개의 레이어를 쌓은 경우가 대부분이고, 레이어가 늘어남에 따라 네트워크를 훈련시키는 데 걸리는 시간도 증가한다. 다행히 연구자들은 자신이 만든 사전 훈련된 모델(pre-trained model)을 인터넷에 올려놓는다. 이렇게 얻은 모델을 그대로 사용할 수도 있고, 전이 학습(Transfer Learning)이나 신경 스타일 전이(Neural Style Transfer)처럼 다른 과제를 위해 재가공해서 사용할 수도 있다.

<br>

## 텐서플로 허브

텐서플로에서 제공하는 텐서플로 허브(TensorFlow Hub)는 재사용 가능한 모델을 쉽게 이용할 수 있는 라이브러리다. 텐서플로 허브 홈페이지([https://tfhub.dv/](https://tfhub.dv/))에서는 이미지, 텍스트, 비디오 등의 분야에서 사전 훈련된 모델들을 검색해볼 수 있다.

<br>

### 텐서플로 허브에서 사전 훈련된 MobileNet 모델 불러오기

텐서플로 허브에 올라와 있는 모델은 hub.KerasLayer() 명령으로 tf.Keras에서 사용 가능한 레이어로 변환할 수 있다.

output shape의 1001은 클래스의 갯수를 의미한다. 1000종류의 이미지와 이 가운데 어떤 것에도 속하지 않는다고 판달될때는 background를 반환한다.

```python
import tensorflow_hub as hub

mobile_net_url = "<https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2>"
model = tf.keras.Sequential([
            hub.KerasLayer(handle=mobile_net_url, input_shape=(224,224,3), trainable=False)
])

model.summary()

#---------------------출력---------------------#
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
keras_layer (KerasLayer)     (None, 1001)              3540265   
=================================================================
Total params: 3,540,265
Trainable params: 0
Non-trainable params: 3,540,265
_________________________________________________________________

```

<br>

### 데이터셋 불러오기

성능을 평가하기 위해 ImageNet의 데이터 중 일부만 모아놓은 ImageNetV2를 사용한다. 또 이중에서도 각 클래스에서 가장 많은 선택을 받은 이미지 10장씩을 모아놓은 10,000장의 이미지가 포함된 TopImages 데이터를 사용한다.

tf.keras.utils.get_file() 함수로 ImageNetV2 데이터를 불러올 수 있다. 함수 인수 중 extract=True로 지정했기 때문에 tar.gz 형식의 압축 파일이 자동으로 해제되어 구글 코랩 가상 머신에 저장된다.

```python
import os
import pathlib
content_data_url = '/content/sample_data'
data_root_orig = tf.keras.utils.get_file(
    'imagenetV2', 
    '<https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-top-images.tar.gz>', 
    cache_dir=content_data_url, 
    extract=True)
data_root = pathlib.Path(content_data_url + '/datasets/imagenetv2-topimages')
print(data_root)

```

데이터 디렉터리 밑에 각 라벨에 대한 숫자 이름으로 하위 디렉터리가 만들어져있다. 하위 디렉터리는 0~999까지 총 1,000개이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-10-ch8_Pre_Trained_Model/fig1.JPG?raw=true)

<br>

### 라벨 텍스트 불러오기

라벨에 대한 숫자가 어떤 데이터를 뜻하는지에 대한 정보는 tf.keras.utils.get_file()을 통해따로 불러와야한다.

총 1001개의 라벨이 있다. 이 중 0번째 라벨은 아무것도 없다는 background이고, 1~1000까지 실제 클래스의 라벨이다.

```python
label_file = tf.keras.utils.get_file('label',
  '<https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt>')
label_text = None
with open(label_file, 'r') as f:
  label_text = f.read().split('\\n')[:-1]
print(len(label_text))
print(label_text[:10])

#---------------------출력---------------------#
1001
['background', 'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead',
 'electric ray', 'stingray', 'cock', 'hen']

```

<br>

### 이미지 출력

총 이미지는 10000장이다.

```python
import matplotlib.pyplot as plt
import random

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
# 이미지를 랜덤하게 섞는다.
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print('image_count:', image_count)

plt.figure(figsize=(12,12))
for c in range(9):
  image_path = random.choice(all_image_paths)
  plt.subplot(3,3,c+1)
  plt.imshow(plt.imread(image_path))
  idx = int(image_path.split('/')[-2]) + 1 # background의 인덱스가 0이므로 1을 더함
  plt.title(str(idx) + ', '+label_text[idx])
  plt.axis('off')
plt.show()
#---------------------출력---------------------#
image_count: 10000

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-10-ch8_Pre_Trained_Model/fig2.JPG?raw=true)

<br>

### 성능 평가

Top-5 accuracy는 83%, Top-1 accuracy는 60%정도 나왔다.

```python
import cv2
import numpy as np

top_1 = 0
top_5 = 0
for image_path in all_image_paths:
  img = cv2.imread(image_path)
  img = cv2.resize(img, dsize=(224, 224))
  img = img / 255.0
  img = np.expand_dims(img, axis=0)
  top_5_predict = model.predict(img)[0].argsort()[::-1][:5]
  idx = int(image_path.split('/')[-2]) + 1
  if idx in top_5_predict:
    top_5 += 1
    if top_5_predict[0] == idx:
      top_1 += 1

print('Top-5 correctness:', top_5 / len(all_image_paths) * 100, '%')
print('Top-1 correctness:', top_1 / len(all_image_paths) * 100, '%')


```

<br>

<br>

## 전이 학습

전이 학습(Transfer Learning)은 미리 훈련된 모델을 다른 작업에 사용하기 위해 추가적인 학습을 시키는 것이다.

<br>

### 모델의 일부를 재학습시키기

전이 학습의 예시로 CNN의 전이 학습을 보자. 미리 훈련된 CNN을 불러올 때 가장 마지막의 Dense 레이어는 제외한다. 그리고 새로운 기능을 수행할 Dense 레이어를 추가한다. 이후 네트워크를 훈련한다. 이때 새로 추가된 레이어의 가중치만 훈련시킬 수도 있고, 미리 훈련된 모델의 일부 레이어를 훈련시킬 수도 있다. 이 때 훈련하지 않는 레이어를 '얼린다(freeze)' 라고 표현한다. 새로운 작업을 위한 데이터의 양이 많을수록 기존에 훈련된 데이터와 차이가 많아져서 다시 학습할 필요가 생기기 때문에 얼리는 레이어의 양을 줄이게 된다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-10-ch8_Pre_Trained_Model/fig3.JPG?raw=true)

<br>

### 캐글 다운로드

스탠퍼드 대학의 Dogs Dataset을 이용한다. 이 데이터는 2만여 개의 사진으로 구성돼 있으며 120가지 견종에 대한 라벨이 붙어있다. 여기서는 원본 데이터보다 접근이 쉬운 캐글 데이터셋을 사용한다. 캐글에서 제공하는 파이썬 API를 사용하면 손쉽게 데이터를 받고 학습 결과를 캐글에 올릴 수 있다.

```python
!pip install kaggle

```

<br>

### 데이터셋 다운로드

현재는 캐글의 파일구조가 바뀌어서 아래 코드로 파일을 받아온다. 하지만 캐글에 사용법을 알아두면 좋다.

```python
import os
# os.environ['KAGGLE_USERNAME'] = 'user_name' # 토큰을 받아 이름과 키 입력
# os.environ['KAGGLE_KEY'] = 'user_key'
# !kaggle competitions download -c dog-breed-identification

tf.keras.utils.get_file('/content/labels.csv', '<http://bit.ly/2GDxsYS>')
tf.keras.utils.get_file('/content/sample_submission.csv', '<http://bit.ly/2GGnMNd>')
tf.keras.utils.get_file('/content/train.zip', '<http://bit.ly/31nIyel>')
tf.keras.utils.get_file('/content/test.zip', '<http://bit.ly/2GHEsnO>')

!unzip train.zip

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-10-ch8_Pre_Trained_Model/fig4.JPG?raw=true)

<br>

### labels.csv 파일의 내용확인

파일의 이름인 id와 breed(견종)으로 구성되어 있다. 총 데이터는 10,222장의 사진이 훈련 데이터에 포함되어 있다.

```python
import pandas as pd
label_text = pd.read_csv('labels.csv')
print(label_text.head())

label_text.info()

#---------------------출력---------------------#
																id             breed
0  000bec180eb18c7604dcecc8fe0dba07       boston_bull
1  001513dfcb2ffafc82cccf4d8bbaba97             dingo
2  001cdf01b096e06d78e9e5112d419397          pekinese
3  00214f311d5d2247d5dfe4fe24b2303d          bluetick
4  0021f9ceb3235effd7fcde7f7538ed62  golden_retriever

RangeIndex: 10222 entries, 0 to 10221
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   id      10222 non-null  object
 1   breed   10222 non-null  object
dtypes: object(2)
memory usage: 159.8+ KB

```

<br>

### 이미지 확인

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12,12))
for c in range(9):
  image_id = label_text.loc[c, 'id']
  plt.subplot(3,3,c+1)
  plt.imshow(plt.imread('./train/' + image_id + '.jpg'))
  plt.title(str(c) + ', ' + label_text.loc[c, 'breed'])
  plt.axis('off')
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-10-ch8_Pre_Trained_Model/fig5.JPG?raw=true)

<br>

### 훈련 데이터를 메모리에 로드

코랩에서 메모리가 부족하여 훈련 데이터 중 5000장만 불러왔다.

```python
import cv2

train_X = []
for i in range(5000):
  img = cv2.imread('./train/' + label_text['id'][i] + '.jpg')
  img = cv2.resize(img, dsize=(224,224))
  img = img / 255.0
  train_X.append(img)
train_X = np.array(train_X)
print(train_X.shape)
#---------------------출력---------------------#
(5000, 224, 224, 3)

```

<br>

### train 라벨 데이터를 메모리에 로드

```python
unique_Y = label_text['breed'].unique().tolist()
train_Y = [unique_Y.index(breed) for breed in label_text['breed']]
train_Y = np.array(train_Y)
print(train_Y)
#---------------------출력---------------------#
(5000,)

```

<br>

### 전이 학습 모델 정의

```python
from tensorflow.keras.applications import MobileNetV2
mobilev2 = MobileNetV2()
x = mobilev2.layers[-2].output
predictions = tf.keras.layers.Dense(120, activation='softmax')(x)
model = tf.keras.Model(inputs=mobilev2.input, outputs=predictions)

# 뒤에서 20개까지의 레이어는 훈련 가능, 나머지는 가중치 고정
for layer in model.layers[:-20]:
  layer.trainable = False
for layer in model.layers[-20:]:
  layer.trainable = True

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

#---------------------출력---------------------#
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 224, 224, 3) 0                                            
__________________________________________________________________________________________________
Conv1 (Conv2D)                  (None, 112, 112, 32) 864         input_1[0][0]                    
__________________________________________________________________________________________________
bn_Conv1 (BatchNormalization)   (None, 112, 112, 32) 128         Conv1[0][0]                      
__________________________________________________________________________________________________
Conv1_relu (ReLU)               (None, 112, 112, 32) 0           bn_Conv1[0][0]                   
__________________________________________________________________________________________________
...
(중간 생략)
...
Conv_1 (Conv2D)                 (None, 7, 7, 1280)   409600      block_16_project_BN[0][0]        
__________________________________________________________________________________________________
Conv_1_bn (BatchNormalization)  (None, 7, 7, 1280)   5120        Conv_1[0][0]                     
__________________________________________________________________________________________________
out_relu (ReLU)                 (None, 7, 7, 1280)   0           Conv_1_bn[0][0]                  
__________________________________________________________________________________________________
global_average_pooling2d (Globa (None, 1280)         0           out_relu[0][0]                   
__________________________________________________________________________________________________
dense (Dense)                   (None, 120)          153720      global_average_pooling2d[0][0]   
==================================================================================================
Total params: 2,411,704
Trainable params: 1,204,280
Non-trainable params: 1,207,424
__________________________________________________________________________________________________

```

<br>

### 모델 학습

과적합 된 듯한 모습을 보이고, 검증 데이터의 정확도는 50%정도 나온다.

```python
history = model.fit(train_X, train_Y, epochs=10, validation_split=0.25, batch_size=32)

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

plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-10-ch8_Pre_Trained_Model/fig6.JPG?raw=true)

<br>

<br>

## 신경 스타일 전이

신경 스타일 전이(Neural Style Transfer) 논문은 반 고흐의 <별이 빛나는 밤에> 라는 그림과 풍경 사진을 합성해서 반 고흐가 그린 것 같은 스타일의 풍경 이미지를 만들어냈다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-10-ch8_Pre_Trained_Model/fig7.JPG?raw=true)

<br>

## 컨볼루션 신경망을 사용한 텍스처 합성

텍스처(Texture)는 컴퓨터 비전에서 쓰이는 의미로는 지역적으로는 비교적 다양한 값을 가지면서 전체적으로는 비슷한 모습을 보이는 이미지를 뜻한다. 텍스처 합성(Texture Synthesis)은 한 장의 이미지를 원본으로 삼아 해당 텍스처를 재생성하는 작업이다. 이 때 합성되는 이미지는 원본과 비슷해야 하지만 똑같아서는 안되며, 어색한 부분과 반복적인 부분이 없어야 하며 원하는 크기로 생성 가능해야한다.

기존의 텍스처 합성 방법 중 가장 효과적이었던 방법은 크게 두 가지로 분류할 수 있다. 첫 번째는 픽셀이나 이미지를 잘게 쪼갠 단위인 Patch를 재배열하는 방법이다. 두 번째는 파라미터에 의한 텍스처 모델링이다. 먼저 원본 텍스처의 공간적인 통곗값(spatial statistics)을 사람이 정교하게 만든 여러 개의 필터로 구한다. 필터를 통과한 결과물이 같다면 같은 텍스처라고 가정한 후 이 결과물이 같아질 때까지 타깃 텍스처를 변형시킨다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-10-ch8_Pre_Trained_Model/fig8.JPG?raw=true)

여기서 필터 부분을 딥러닝을 활용할 수 있다는 생각이 든다. 필터를 컨볼루션 신경망이 자동으로 만들게 해서 네트워크가 이미지의 특징을 자동으로 추출할 수 있게 하는 것이다. 여기서는 네트워크로 VGG-19를 사용한다.

필터의 각 레이어에서 뽑아낸 특징값을 종합하여 Gram matrix라는 값을 계산한다. Gram matrix는 특징 추출값을 1차원의 벡터로 변환한 다음, 벡터를 쌓아올린 행렬을 자신의 전치 행렬과 행렬곱해서 얻는 값이다. 이렇게 얻은 Gram matrix는 특징 추출값의 상관관계를 나타낸다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-10-ch8_Pre_Trained_Model/fig9.JPG?raw=true)

Gram matrix를 원본 텍스처와 타깃 텍스처에 대해서 모두 구한 다음, 두 Gram matrix의 평균 제곱 오차를 구한다. 이후 이 오차가 작아지도록 타깃 텍스처를 변형하면 원본 텍스처를 닮은 타깃 텍스처가 생성된다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-10-ch8_Pre_Trained_Model/fig10.JPG?raw=true)

<br>

### 원본 텍스처 이미지 불러오기

```python
import matplotlib.pyplot as plt
import cv2

style_path = tf.keras.utils.get_file('style.jpg', '<http://bit.ly/2mGfZIq>')

style_image = plt.imread(style_path)
style_image = cv2.resize(style_image, dsize=(224,224))
style_image = style_image / 255.0
plt.imshow(style_image)

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-10-ch8_Pre_Trained_Model/fig11.JPG?raw=true)

<br>

### 타깃 텍스처 만들기

타깃 텍스처는 랜덤 노이즈 이미지에서 시작한다.

```python
target_image = tf.random.uniform(style_image.shape)
plt.imshow(target_image)

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-10-ch8_Pre_Trained_Model/fig12.JPG?raw=true)

<br>

### VGG-19 네트워크 불러오기

텍스처 합성에 사용할 VGG-19 네트워크를 불러온다. 불러올 때 include_top 인수를 False로 지정해서 마지막의 Dense 레이어를 제외한 나머지 레이어를 불러온다.

```python
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

vgg = VGG19(include_top=False, weights='imagenet')

for layer in vgg.layers:
  print(layer.name)
#---------------------출력---------------------#
input_1
block1_conv1
block1_conv2
block1_pool
block2_conv1
block2_conv2
block2_pool
block3_conv1
block3_conv2
block3_conv3
block3_conv4
block3_pool
block4_conv1
block4_conv2
block4_conv3
block4_conv4
block4_pool
block5_conv1
block5_conv2
block5_conv3
block5_conv4
block5_pool

```

<br>

### 특징 추출 모델 만들기

이미지를 입력하면 지정한 레이어에서 출력을 내는 새로운 모델을 만들었다.

```python
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

vgg.trainable = False
outputs = [vgg.get_layer(name).output for name in style_layers]
model = tf.keras.Model([vgg.input], outputs)

```

<br>

### Gram matrix 계산 함수 정의

입력된 특징 추출값의 형태를 변환한다. 예시로 첫 번째 레이어인 block1_conv1을 통과한 특징 추출값의 차원수는 (224, 224, 64)이다. 이것을 (50176, 64)로 변환하여 a에 저장한다. a와 a의 전치 행렬의 곱을 계산하여 (64, 64)의 행렬을 반환하고, 50176으로 나눠서 최종 값을 반환한다.

```python
def gram_matrix(input_tensor):
  channels = int(input_tensor.shape[-1])
  a = tf.reshape(input_tensor, [-1, channels])
  n = tf.shape(a)[0]
  gram = tf.matmul(a, a, transpose_a=True)
  return gram / tf.cast(n, tf.float32)

```

<br>

### 원본 텍스처의 Gram matrix 계산

결과를 확인해보면 Gram matrix 값은 레이어마다 다르게 나오고 최댓값도 크게 차이가 난다. 각 레이어에서 계산되는 Gram matrix 값에 가중치를 곱해주는 방법으로 특정한 레이어가 너무 큰 영향을 끼치지 못하도록 제어 할 수도 있으나 여기서는 가중치 없이 계산해본다.

```python
style_batch = style_image.astype('float32')
style_batch = tf.expand_dims(style_batch, axis=0) # 입력넣을때 차원수 + 1
style_output = model(preprocess_input(style_batch * 255.0)) # 특징 추출

style_outputs = [gram_matrix(out) for out in style_output]

plt.figure(figsize=(12,10))
for c in range(5):
  plt.subplot(3,2,c+1)
  array = sorted(style_outputs[c].numpy()[0].tolist())
  array = array[::-1]
  plt.bar(range(style_outputs[c].shape[0]), array)
  plt.title(style_layers[c])
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-10-ch8_Pre_Trained_Model/fig13.JPG?raw=true)

<br>

### 타깃 텍스처를 업데이트하기 위한 함수 정의

타깃 텍스처를 업데이트하기 위해서는 몇 가지 함수를 설정해야 한다. 먼저 타깃 텍스처에서 Gram matrix를 구하는 함수가 필요하다. 그리고 원본 텍스처의 Gram matrix값과 타깃 텍스처의 Gram matrix값 사이의 MSE를 구하는 함수가 필요하다. 또 나오는 값이 0.0 에서 1.0 사이의 컬러 값이어야 하기 때문에 그 이하나 이상으로 값이 넘어가지 않게 해주는 함수가 필요하다.

```python
def get_outputs(image):
  image_batch = tf.expand_dims(image, axis=0)
  output = model(preprocess_input(image_batch * 255.0))
  outputs = [gram_matrix(out) for out in output]
  return outputs

def get_loss(outputs, style_outputs):
  return tf.reduce_sum([tf.reduce_mean((o-s)**2) for o,s in zip(outputs, style_outputs)])

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

```

<br>

### 이미지 업데이트 함수 정의

지금까지 배워온 딥러닝 네트워크의 학습은 tf.keras를 이요해 모델을 정의하고 fit() 함수를 이용해 가중치가 주어진 과제를 잘 수행하도록 학습시키는 것이었다. 그러나 여기에는 학습해야할 가중치가 존재하지 않고, 인풋인 이미지를 업데이트 해야한다. 텐서플로의 Gradient Tape은 이런부분을 해결해준다. 어떤 식이 들어가더라도 자동 미분을 통해 입력에 대한 손실을 구한 뒤 다른 변수에 대한 Gradient를 계산한다. 여기서 다른 변수는 입력이 될 수도 있고, 가중치가 될 수도 있다.

여기서는 tf.function()이 등장한다. 이 함수는 train_step() 함수를 인수로 받아서 Autograph라는 기능을 추가해준다. Autograph는 파이썬 문법으로 그래프를 컨트롤 할 수 있게 한다.

```python
opt = tf.optimizers.Adam(learning_rate=0.2, beta_1=0.99, epsilon=1e-1)

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = get_outputs(image)
    loss = get_loss(outputs, style_outputs)

  grad = tape.gradient(loss, image) # 이미지에 대한 gradient 계산
  opt.apply_gradients([(grad, image)]) # image 업데이트
  image.assign(clip_0_1(image)) # image의 값을 0~1사이로 유지

```

<br>

### 텍스처 합성 알고리즘 실행

결과 원본과 비슷한 느낌이 들기도 하지만 매끄러운 원본과 달리 자글자글한 노이즈가 보인다.

```python
import IPython.display as display

image = tf.Variable(target_image)

epochs = 50
steps_per_epoch = 100

step =0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
  display.clear_output(wait=True)
  plt.imshow(image.read_value())
  plt.title("Train_step: {}".format(step))
  plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-10-ch8_Pre_Trained_Model/fig14.JPG?raw=true)

<br>

### variation loss 함수 정의

자글자글한 노이즈를 개선하기 위해 전체 손실에 variation loss라는 것을 추가한다. variation loss란 어떤 픽셀과 바로 옆에 인접한 픽셀의 차이이다. 이 차이가 작을수록 이미지는 매끄럽게 보일것이다.

```python
def high_pass_x_y(image):
  x_var = image[:,1:,:] - image[:,:-1,:]
  y_var = image[1:,:,:] - image[:-1,:,:]
  return x_var, y_var

def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)

```

<br>

### variation loss를 손실 계산식에 추가, 각 손실의 가중치 추가

기존의 손실 함수에 variation loss를 추가한다. 기존의 loss는 0.1만큼의 가중치, 그리고 variation loss는 1e-9의 가중치를 준다.

```python
total_variation_weight = 1e9
style_weight = 1e-1

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = get_outputs(image)
    loss = style_weight * get_loss(outputs, style_outputs) 
    loss += total_variation_weight * total_variation_loss(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

```

<br>

### variation loss를 추가한 텍스처 합성 알고리즘 실행

확실히 결과 이미지가 이전보다는 매끄러워진것을 확인할 수 있다.

```python
import IPython.display as display

target_image = tf.random.uniform(style_image.shape)
image = tf.Variable(target_image)

epochs = 50
steps_per_epoch = 100

step =0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
  display.clear_output(wait=True)
  plt.imshow(image.read_value())
  plt.title("Train_step: {}".format(step))
  plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-10-ch8_Pre_Trained_Model/fig15.JPG?raw=true)

<br>

<br>

## 컨볼루션 신경망을 사용한 신경 스타일 전이

신경 스타일 전이는 위에서 배운 Gram matrix를 이용한 텍스처 합성에 한가지를 더 추가한 것이다. 바로 content 텍스처이다. 타깃 텍스처를 만들기 위해서 style 텍스처와 Gram matrix의 MSE를 구하고, content 텍스처와는 픽셀 값의 차이인 MSE를 구한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-10-ch8_Pre_Trained_Model/fig16.JPG?raw=true)

<br>

### content 텍스처 불러오기

content 텍스처와 타깃 텍스처의 크기가 같아야한다. 이 둘은 서로 특징 추출값의 픽셀을 MSE로 비교하기 때문에 크기가 다르면 안된다. 반면 style 텍스처는 타깃 첵스처와 크기가 달라도 상관없다. Gram matrix 계산값은 각 레이어의 [채널수] * [채널수]만큼의 값을 서로 비교하기 때문이다.

```python
import matplotlib.pyplot as plt
import cv2

content_path = tf.keras.utils.get_file('content.jpg', '<http://bit.ly/2mAfUX1>')

content_image = plt.imread(content_path)
max_dim = 512
long_dim = max(content_image.shape[:-1])
scale = max_dim / long_dim
new_height = int(content_image.shape[0] * scale)
new_width = int(content_image.shape[1] * scale)

content_image = cv2.resize(content_image, dsize=(new_width, new_height))
content_image = content_image / 255.0
plt.figure(figsize=(8,8))
plt.imshow(content_image)

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-10-ch8_Pre_Trained_Model/fig17.JPG?raw=true)

<br>

### content 특징 추출 모델 만들기

```python
content_batch = content_image.astype('float32')
content_batch = tf.expand_dims(content_batch, axis=0)

content_layers = ['block5_conv2']

vgg.trainable = False
outputs = [vgg.get_layer(name).output for name in content_layers]
model_content = tf.keras.Model([vgg.input], outputs)
content_output = model_content(preprocess_input(content_batch * 255.0))

```

<br>

### content, output, loss 함수 정의

```python
def get_content_output(image):
  image_batch = tf.expand_dims(image, axis=0)
  output = model_content(preprocess_input(image_batch * 255.0))
  return output

def get_content_loss(image, content_output):
  return tf.reduce_sum(tf.reduce_mean(image-content_output)**2)

```

<br>

### content loss를 손실 계산식에 추가

```python
opt = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.99, epsilon=1e-1)

total_variation_weight = 1e9
style_weight = 1e-2
content_weight = 1e4

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = get_outputs(image)
    output2 = get_content_output(image)
    loss = style_weight * get_loss(outputs, style_outputs)
    loss += content_weight * get_content_loss(output2, content_output)
    loss += total_variation_weight * total_variation_loss(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

```

<br>

### 신경 스타일 전이 실행

학습 시간을 절약하기 위해서 타깃 텍스처를 랜덤 노이즈가 아닌 content 텍스처에서 시작하게 했다.

```python
import IPython.display as display

# target_image = tf.random.uniform(style_image.shape)
image = tf.Variable(content_image.astype('float32'))

epochs = 20
steps_per_epoch = 100

step =0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
  display.clear_output(wait=True)
  plt.imshow(image.read_value())
  plt.title("Train_step: {}".format(step))
  plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-10-ch8_Pre_Trained_Model/fig18.JPG?raw=true)