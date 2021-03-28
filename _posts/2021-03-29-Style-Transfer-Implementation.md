---
layout: post
title: Style Transfer 구현
featured-img: 2021-03-29-Style-Transfer-Implementation/fig2
permalink: /paper_implementation/2021-03-29-Style-Transfer-Implementation
category: paper_implementation

---

해당 포스트는 style transfer로 알려져 있는 Image Style Transfer Using Convolutional Neural Networks에 대한 구현입니다. 해당 코드는 tensorflow 홈페이지와 나동빈님의 github을 참고하여 작성하였습니다.

자세한 논문의 내용은 [Style transfer 리뷰](https://sungbeomchoi.github.io/paper_review/2021-03-22-Style-Transfer-Review)을 참고하세요

<br>

<p  align="center">
<a  href="https://colab.research.google.com/github/SUNGBEOMCHOI/Paper_implementation/blob/main/Style%20Transfer/Style_Transfer(Keras).ipynb">
<img  src="https://colab.research.google.com/assets/colab-badge.svg"  alt="Open In Colab"  height="50"  width="180"/>
</a>
</p>

<br>


### 라이브러리 불러오기

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import IPython.display as display
import PIL.Image

```
<br>
<br>

## 이미지 준비

style transfer를 위한 content image와 style image를 준비한다.

```python
dir = '<https://github.com/SUNGBEOMCHOI/Paper_implementation/blob/main/Style%20Transfer/image>'
content_img_path = dir + '/content_img_1.jpg?raw=true' # content 이미지의 url을 입력
style_img_path = dir + '/style_img_1.jpg?raw=true'
content_img_path = tf.keras.utils.get_file(os.getcwd()+'/image/content_img1.jpg', content_img_path)
style_img_path = tf.keras.utils.get_file(os.getcwd()+'/image/style_img1.jpg', style_img_path)

```
<br>

```python
def load_img(path_to_img): # 파일의 경로로부터 이미지 tensor를 반환
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.io.decode_image(img, channels=3, dtype=tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :] # axis=0의 차원 증가
    return img

def show_image(image, title=None): # 이미지를 시각화
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0) # axis-0의 차원 축소
    
    plt.imshow(image)
    if title:
        plt.title(title)

def tensor_to_image(tensor): # tensor를 PIL image로 반환
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

```
<br>

```python
content_image = load_img(content_img_path)
style_image = load_img(style_img_path)

```
<br>

```python
plt.subplot(1, 2, 1)
show_image(content_image, 'content_image')
plt.subplot(1, 2, 2)
show_image(style_image, 'style_image')

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-29-Style-Transfer-Implementation/fig1.JPG?raw=true)

<br>
<br>

## 모델 준비

학습된 VGG19 네트워크를 fully connected layer를 제외한 feature extrator 부분만 가져옵니다.

```python
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

def vgg_layers(layer_names):
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

```
<br>

생성하는 이미지와 content, style 이미지와 비교할 layer를 정의합니다. content layer의 경우 detail한 pixel은 죽이고, 전체적인 이미지를 위해 block5_conv2를 사용합니다. 또 style layer의 경우 1, 2, 3, 4, 5의 block 레이어를 가져옵니다. block 레이어는 각 2개의 conv layer로 구성되어있습니다.

```python
content_layers = ['block5_conv2'] 
style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']

```

<br>

gram matrix 추출하기 위한 함수를 제작합니다. gram matrix는 style image에서 style을 뽑아내기 위해 사용합니다.

```python
def gram_matrix(input_tensor):
    result = tf.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

```

<br>

```python
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
      super(StyleContentModel, self).__init__()
      self.vgg =  vgg_layers(style_layers + content_layers)
      self.style_layers = style_layers
      self.content_layers = content_layers
      self.num_style_layers = len(style_layers)
      self.vgg.trainable = False

    def call(self, inputs):
      "[0,1] 사이의 실수 값을 입력으로 받습니다"
      inputs = inputs*255.0
      preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
      outputs = self.vgg(preprocessed_input)
      style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                        outputs[self.num_style_layers:])

      style_outputs = [gram_matrix(style_output)
                       for style_output in style_outputs]

      content_dict = {content_name:value 
                      for content_name, value 
                      in zip(self.content_layers, content_outputs)}

      style_dict = {style_name:value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}

      return {'content':content_dict, 'style':style_dict}

```

<br>

```python
extractor = StyleContentModel(style_layers, content_layers)

```

<br>
<br>

## 경사하강법 실행

```python
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']
image = tf.Variable(content_image) # content_image와 똑같은 tensor 생성, 변화시킬 이미지

```

<br>

```python
style_weight=1e-2
content_weight=1e4

def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / len(style_outputs)

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / len(content_outputs)
    loss = style_loss + content_loss
    return loss

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

```
<br>

```python
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

```
<br>

```python
epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print('.', end='')
    display.clear_output(wait=True)
    display.display(tensor_to_image(image))
    print(f'훈련스텝: {step}')

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-29-Style-Transfer-Implementation/fig2.JPG?raw=true)