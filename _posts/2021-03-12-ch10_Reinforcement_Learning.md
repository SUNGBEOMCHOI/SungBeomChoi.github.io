---
layout: post
title: (시작하세요! 텐서플로 2.0 프로그래밍) 10장. 강화학습(Reinforcement Learning)
featured-img: 2021-03-12-ch10_Reinforcement_Learning/fig15
permalink: /book_review/2021-03-12-ch10_Reinforcement_Learning
category: book_review

---
강화학습은 실수와 보상을 통해 배우는 알고리즘이다. 신경망이 정답과 예측 사이의 오차를 역전파해서 의미있는 가중치와 편향을 학습하는 것처럼 강화학습은 좋은 선택과 나쁜 선택에서 배운다.

<br>

## 신경망으로 경험 학습하기

강화학습의 환경으로 사용할 Gym에 대해 알아본다. Gym에는 전통적인 알고리즘 흉내 내기, Box2D를 사용한 간단한 물리 조작계, 아타리 게임 등 다양한 환경이 포함되어있다.

Gym의 구조는 강화학습에서 요구하는 표준적인 구조이다. 일단 문제가 주어진 환경(environment)이 있고, 강화학습 문제를 풀기 위한 에이전트(agent)가 존재한다. 에이전트는 행동(action)으로 환경에 영향을 주고, 그 결과에 따라 보상(reward)을 받는다. 좋은 보상을 받으면 에이전트는 그 행동을 더 많이 하게 되고, 나쁜 보상을 받으면 그 행동을 덜 하도록 학습하는 것이 강화학습의 기본이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig1.JPG?raw=true)

여기서 풀어볼 문제는 MountainCar-v0이다. 두 바퀴가 달린 차(에이전트)로 언덕(환경)을 올라가는 문제인데, 이 때 보상은 각 시간 단위(time step)마다 -1이 주어지고, 오른쪽의 깃발에 도달하면 하나의 에피소드(episode)가 끝난다. 즉, 보상은 음수로 주어지며 에피소드를 빨리 끝낼수록 한 에피소드에 얻는 보상의 총합이 커지기 때문에 가장 짧은 시간 내에 언덕을 올라가야한다.

<br>

에이전트가 취할 수 있는 행동은 "왼쪽으로 이동", "정지", "오른쪽으로 이동"의 3가지 이다. 만약 오른쪽으로 이동하는 행동은 가파른 언덕을 올라갈 만큼의 충분한 힘을 받지 못하기 때문에 이 행동만으로는 언덕을 올라갈 수 없다. 왼쪽과 오른쪽으로 반복해서 움직이며 가속도를 붙인 다음에야 언덕을 올라갈 수 있다.

<br>

### MountainCar-v0 환경 만들기

```python
import gym
import random
env = gym.make('MountainCar-v0')

```

<br>

### 환경의 관찰 공간, 행동 공간 변수 확인

관찰공간은 2개의 숫자로 이뤄지며, 각각 최댓값과 최솟값을 가진다. 관찰공간은 x 위치와 속도이다. x 위치의 범위는 -1.2~0.6이고, 속도 범위는 -0.07에서 0.07입니다. 종료 조건은 X 위치가 0.5, 즉 깃발에 도달했을 때이다. 각 에피소드가 시작될 때 차는 -0.6에서 -0.4 사이의 랜덤한 위치에서 시작한다. 에이전트가 취할 수 있는 행동의 경우의 수인 행동 공간(action space)은 이산적인 3가지이다. 에피소드는 200 time step이 지나면 멈춘다.

```python
print(env.observation_space)
print(env.observation_space.low)
print(env.observation_space.high, '\\n')
print(env.action_space, '\\n')
print(env._max_episode_steps)
#---------------------출력---------------------#
Box(-1.2000000476837158, 0.6000000238418579, (2,), float32)
[-1.2  -0.07]
[0.6  0.07] 

Discrete(3) 

200

```

<br>

### 랜덤한 행동을 하는 에이전트 실행

랜덤한 행동을 env에 전달하면 obs(관측값), reward(보상), done(종료여부), info(기타정보)를 받는다.

```python
step = 0
score = 0
env.reset()

while True:
  action = env.action_space.sample() # action_space에서 속도를 랜덤하게 뽑음
  obs, reward, done, info = env.step(action) 
  print(score)
  score += reward
  step += 1

  if done:
    break

print('score:', score)
print('step:', step)
#---------------------출력---------------------#
(초기생략)
...
-196.0
-197.0
-198.0
-199.0
score: -200.0
step: 200

```

<br>

### 성공적인 에피소드 저장

문제를 풀기 위해서는 여러 번의 에피소드 중 성공적인 에피소드를 저장한 다음, 그때 행동했던 데이터를 신경망에 학습시키는 방법을 사용해볼 수 있다. 따라서 신경망을 학습시킬 데이터를 확보하는 코드가 필요하다. 랜덤한 행동을 하는 에이전트로 10000 에피소드를 실행하고 그 중 성공적인 에피소드의 데이터를 저장한다.

관찰 상태의 첫 번째인 X좌표가 -0.2보다 클 경우, 보상을 -1 대신 +1로 바꾼다. 그리고 이렇게 계산된 에피소드의 누적 보상이 -198보다 클 경우, 즉 3번 이상 -0.2보다 큰 X좌표 값을 기록했을 경우 성공적인 에피소드로 판단해서 training_data에 game_memory 의 값을 저장한다. training_data에는 obs값과 action이 들어간다.

결과를 확인해보면 모든 행동이 실패했음을 확인할 수 있다. 누적 보상이 -198보다 큰 경우는 50번 정도인데, 200스텝이므로 총 훈련데이터는 10000개이다. 이 정도면 신경망을 학습시키기에는 충분한 데이터이다.

```python
env = gym.make('MountainCar-v0')

scores = []
training_data = []
accepted_scores = []
required_score = -198

for i in range(10000):
  if i % 100 == 0:
    print(i)
  env.reset()
  score = 0
  game_memory = []
  previous_obs = []

  while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    if len(previous_obs) > 0:
      game_memory.append([previous_obs, action])

    previous_obs = obs
    if obs[0] > -0.2:
      reward = 1

    score += reward

    if done:
      break

  scores.append(score)
  if score > required_score:
    accepted_scores.append(score)
    for data in game_memory:
      training_data.append(data)

scores = np.array(scores)
print(accepted_scores)

import seaborn as sns
sns.distplot(scores, rug=True)
#---------------------출력---------------------#
[-194.0, -164.0, -190.0, -186.0 ... ]

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig2.JPG?raw=true)

<br>

### 훈련 데이터 만들기

training_data에 있었던 obs값과 action값을 분리하여 각각 X, Y로 넣는다.

```python
train_X = np.array([i[0] for i in training_data]).reshape(-1, 2)
train_Y = np.array([i[1] for i in training_data]).reshape(-1, 1)
print(train_X.shape)
print(train_Y.shape)
#---------------------출력---------------------#
(10149, 2)
(10149, 1)

```

<br>

### 분류 신경망 정의

```python
model = tf.keras.Sequential([
          tf.keras.layers.Dense(128, input_shape=(2,), activation='relu'),
          tf.keras.layers.Dense(32, activation='relu'),
          tf.keras.layers.Dense(3, activation='softmax'),
])

model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

```

<br>

### 학습

학습 결과, 약 40%의 정확도를 보인다. 행동이 3가지이기 때문에 랜덤한 행동을 할 때 33.3%의 정확도를 보일 것이라고 가정하면 뭔가 의미있는 행동을 얻은 것 같다.

```python
history = model.fit(train_X, train_Y, epochs=30, batch_size=16, validation_split=0.25)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r--', label='val_accuracy')
plt.legend()
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig3.JPG?raw=true)

<br>

### 평가

신경망을 통해 이전 상태를 관측하고, 출력으로 내놓은 action을 env에 넣어서 진행한다.

```python
scores = []
steps = []
actions = []

for i in range(500):
  if i % 100 == 0:
    print(i)
  score = 0
  step = 0
  previous_obs = []
  env.reset()

  while True:
    if len(previous_obs) == 0:
      action = env.action_space.sample()
    else:
      logit = model.predict(np.expand_dims(previous_obs, axis=0))[0]
      action = np.argmax(logit)
      actions.append(action)

    obs, reward, done, info = env.step(action)
    previous_obs = obs
    score += reward
    step += 1

    if done:
      break
  scores.append(score)
  steps.append(step)

```

<br>

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].set_title('score')
ax[1].set_title('step')
sns.distplot(scores, rug=True, ax=ax[0])
sns.distplot(steps, rug=True, ax=ax[1])

print(np.mean(scores))
#---------------------출력---------------------#
-141.684

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig4.JPG?raw=true)

<br>

```python
sns.distplot(actions)

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig5.JPG?raw=true)

<br>

<br>

## 큐러닝(Q-Leanrning)

앞 절에서 시도한 방법은 신경망 네트워크를 사용하긴 했지만 강화학습의 이론을 사용한 것은 아니다. 이 절에서는 강화학습의 대표적인 방법론인 큐러닝을 사용한다.

<br>

### MountainCarContinuous-v0 환경만들기

앞 절에서는 행동 공간이 이산적인 MountainCar-v0에서 학습을 했다. 여기서는 MountainCarContinuous-v0라는 행동 공간이 연속된 환경에서 학습한다.

```python
env = gym.make('MountainCarContinuous-v0')

print(env.observation_space)
print(env.observation_space.low)
print(env.observation_space.high, '\\n')
print(env.action_space, '\\n')
print(env._max_episode_steps)
#---------------------출력---------------------#
Box(-1.2000000476837158, 0.6000000238418579, (2,), float32)
[-1.2  -0.07]
[0.6  0.07]

Box(-1.0, 1.0, (1,), float32) 

999

```

<br>

### 랜덤 행동 에이전트의 환경 실행 결과 확인(200스텝 확인)

각 스텝마다 얻는 보상은 행동의 제곱에 0.1을 곱한 값의 음수이다. 그리고 깃발에 도달하면 +100을 얻는다. 깃발에 도달하기 전에는 움직이지 않으면 0의 보상을 받고, 큰 힘으로 움직일수록 음의 보상을 더 많이 받는다.

```python
env.reset()
score = 0
step = 0
for i in range(200):
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)

  previous_obs = obs
  score += reward
  step += 1

  if done:
    break

print(score, step)
#---------------------출력---------------------#
-5.9852823370057235 200

```

<br>

### 랜덤 행동 에이전트의 환경 실행 결과 확인

평균점수는 -32점 정도이다.

```python
env.reset()
score = 0
step = 0

while True:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)

  previous_obs = obs
  score += reward
  step += 1
  if done:
    break

print(score, step)
#---------------------출력---------------------#
-32.65987104544475 999

```

<br>

### 성공적인 에피소드 저장

앞 절과 동일하게 스텝당 -1, X좌표로 -0.2이상일 때 +1을 주는 방식을 사용했다.

```python
scores = []
training_data = []
accepted_scores = []
required_score = -198

for i in range(10000):
  if i % 100 == 0:
    print(i)
  env.reset()
  score = 0
  game_memory = []
  previous_obs = []
  
  for i in range(200):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    if len(previous_obs) > 0:
      game_memory.append([previous_obs, action])

    previous_obs = obs
    if obs[0] > -0.2:
      reward = 1
    else:
      reward = -1

    score += reward

    if done:
      break

  scores.append(score)
  if score > required_score:
    accepted_scores.append(score)
    for data in game_memory:
      training_data.append(data)

scores = np.array(scores)
print(scores.mean())
print(accepted_scores)

import seaborn as sns
sns.distplot(scores, rug=True)
#---------------------출력---------------------#
199.8688
[-176, -168, -194, -184, -174, -186,...

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig6.JPG?raw=true)

<br>

### 데이터 준비

```python
model = tf.keras.Sequential([
              tf.keras.layers.Dense(128, input_shape=(2,), activation='elu'),
              tf.keras.layers.Dense(32, activation='elu'),
              tf.keras.layers.Dense(1, activation='linear')
])
model.compile(optimizer=tf.optimizers.Adam(), loss='mse')

```

<br>

### 회귀 모델 정의

```python
history = model.fit(train_X, train_Y, epochs=10, validation_split=0.25)

```

<br>

### 회귀 신경망으로 에이전트 행동을 확인

```python
scores = []
steps = []
actions = []

for i in range(500):
  if i % 100 == 99:
    print(i, 'mean score: {}, mean step: {}'.format(np.mean(scores[-100:]), np.mean(steps[-100:])))

    score = 0
    step = 0
    previous_obs = []
    env.reset()

    while True:
      if len(previous_obs) == 0:
        action = env.action_space.sample()
      else:
        action = model.predict(np.expand_dims(previous_obs, axis=0))[0]
        actions.append(action)

      obs, reward, done, info = env.step(action)
      previous_obs = obs
      score += reward
      step += 1

      if done:
        break

    scores.append(score)
    steps.append(step)

```

<br>

### score, step 분포 확인

평균 점수는 79점이 나온다.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].set_title('score')
ax[1].set_title('step')
sns.distplot(scores, rug=True, ax=ax[0])
sns.distplot(steps, rug=True, ax=ax[1])

print(np.mean(scores))
#---------------------출력---------------------#
79.30275009257642

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig7.JPG?raw=true)

<br>

### 행동 분포 확인

최댓값과 최솟값은 1과 -1이지만 실제 행동은 그보다 작은 범위에 분포하고 있음을 확인할 수 있다. 특히 0 근처의 값이 가장 많기 때문에 보상도 그만큼 작은 음수값을 가지게 된다.

```python
sns.distplot(actions)

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig8.JPG?raw=true)

회귀 신경망 대신 이번에는 큐러닝을 이용해 문제를 풀어본다. 큐러닝은 관찰 상태에서 취할 수 있는 모든 행동의 q값을 학습하는 방법이다. 특정 상태에서 어떤 행동의 Q값이 다른 행동보다 높다면 Q값이 높은 행동을 우선적으로 선택하게 된다. 가장 높은 Q값을 가진 행동을 선택할 수도 있고, 소프트맥스 함수로 각 Q값을 입력으로 삼아 확률을 기반으로 한 행동을 선택할 수 있다.

모든 상태에 대한 모든 행동의 Q값을 구하면 테이블 형태의 데이터를 얻게 된다. 이를 큐테이블(Q-Table)이라고 한다. 이 큐테이블을 학습시키는 과정이 큐러닝이다. 학습된 큐테이블을 참조하면 큐러닝 에이전트를 움직일 수 있다.

그런데 MountainCarContinuous-v0에서는 행동 공간이 연속적이기 때문에 모든 값에 대한 Q값을 계산하는 것은 불가능합니다. 이럴 때는 행동 공간을 일정한 간격의 격자(grid)로 나눠서 해당 격자에 대한 Q값을 계산할 수 있다. 또 관찰 공간도 연속적이기 때문에 마찬가지로 격자로 나눠서 해당 격자에 대한 Q값을 저장한다.

<br>

### 관찰 공간과 행동 공간을 격자화

처음에는 각 행동의 Q값이 어떤지 알 수 없기 때문에 초기화 과정을 거쳐야 한다. 기본값으로는 매우 작은 값인 0.0001을 넣어준다. 모두 같은 값을 넣어주기 때문에 처음에는 랜덤한 행동이 선택된다. 그리고 해당 행동이 효과적이지 않다고 판단되어 Q값이 조금 떨어지면 나머지 Q값이 그대로 유지되고 있기 때문에 다른 행동으로 탐색할 확률이 증가한다. 초기 Q값을 높게 줄수록 처음에 시도했던 행동이 효과적이지 않을 때 다른 행동을 탐색해볼 확률이 높아진다.

행동은 -1과 1사이의 6개 값이 출력된다.

```python
state_grid_count = 10
action_grid_count = 6

q_table = []
for i in range(state_grid_count):
  q_table.append([])
  for j in range(state_grid_count):
    q_table[i].append([])
    for k in range(action_grid_count):
      q_table[i][j].append(1e-4)

actions = range(action_grid_count)
actions = np.array(actions).astype(float)
actions *= ((env.action_space.high - env.action_space.low) / (action_grid_count - 1))
actions += env.action_space.low

print(actions)
#---------------------출력---------------------#
[-1.         -0.59999999 -0.19999999  0.20000002  0.60000002  1.00000003]

```

<br>

### obs_to_state, softmax 함수 정의

obs_to_state() 함수는 관찰 상태를 각 격자에 배당한다.

```python
import random
def obs_to_state(env, obs):
  obs = obs.flatten()
  low = env.observation_space.low
  high = env.observation_space.high
  idx = (obs - low) / (high - low) * state_grid_count
  idx = [int(x) for x in idx]
  return idx

def softmax(logits):
  exp_logits = np.exp(logits - np.max(logits))
  sum_exp_logits = np.sum(exp_logits)
  return exp_logits / sum_exp_logits

```

```python
sample = env.observation_space.sample()
grid = obs_to_state(env, sample)

print(sample)
print(grid)
#---------------------출력---------------------#
[-0.29474872 -0.0123932 ]
[5, 4]

```

<br>

### 큐러닝 에이전트 학습

행동을 선택할 때 입실론-그리디라는 정책을 사용했다. 입실론-그리디 정책은 입실론이라는 값보다 난수가 작을 때는 랜덤한 행동을 사용하고, 그렇지 않으면 지금까지 찾은 것 중에서 가장 좋은 방법을 선택하는 것이다. 즉, 처음에는 다양한 행동을 시도하는 탐색(exploration)을 하고, 나중에는 지금까지 찾아낸 행동 중 최적의 행동을 이용(exploitation)하는 방법이다. 입실론은 처음에는 큰 값으로 설정하고 학습을 지속함에 따라 점점 작아지도록 설정한다.

큐러닝이 빠르게 정담을 찾을 수 있도록 스텝마다 -0.05의 보상을 더한다. 작은 값이지만 이 보상은 가만히 있는 것보다 에이전트가 움직이도록 자극하는 역할을 한다.

학습률을 사용한 큐함수의 식은 다음과 같다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig11.JPG?raw=true)

큐함수를 업데이트하는 식은 아래와 같다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig12.JPG?raw=true)

```python
max_episodes = 10000
scores = []
steps = []
select_actions = []

learning_rate = 0.05
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01

for i in range(max_episodes):
    epsilon *= 0.9
    epsilon = max(epsilon_min, epsilon)
    
    if i % 100 == 0 and i != 0:
        print(i, 'mean score: {}, mean step: {}, epsilon: {}'.format(np.mean(scores[-100:]), np.mean(steps[-100:]), epsilon))
        
    previous_obs = env.reset()
    score = 0
    step = 0
    
    while True:
        state_idx = obs_to_state(env, previous_obs)
        if random.random() < epsilon:
            action_idx = random.randint(0, action_grid_count-1)
            action = actions[action_idx]
        else:
            logits = q_table[state_idx[0]][state_idx[1]]
            action_idx = np.argmax(softmax(logits))
            action = actions[action_idx]
        
        obs, reward, done, info = env.step([action])
        previous_obs = obs
        score += reward
        reward -= 0.05
        step += 1
        
        select_actions.append(action)
        
        new_state_idx = obs_to_state(env, obs)
        
        q_table[state_idx[0]][state_idx[1]][action_idx] = \\
            q_table[state_idx[0]][state_idx[1]][action_idx] + \\
            learning_rate * (reward + gamma * np.amax(q_table[new_state_idx[0]][new_state_idx[1]]) - q_table[state_idx[0]][state_idx[1]][action_idx])
        
        if done:
            break
    
    scores.append(score)   
    steps.append(step)
    
    if np.mean(scores[-100:]) >= 90:
        print('Solved on episode {}!'.format(i))
        break

#---------------------출력---------------------#
100 mean score: -18.372680860919907, mean step: 970.61, epsilon: 0.01
200 mean score: 60.831919478397396, mean step: 592.18, epsilon: 0.01
Solved on episode 276!

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig9.JPG?raw=true)

score의 분포에서 문제를 푼 경우와 풀지 못한 경우로 나뉘는 것을 확인할 수 있다. 아마도 초반에는 문제를 못 푸는 비율이 많다가 나중에는 문제를 푸는 비율이 많아진 것 같다.

<br>

#### 선택된 행동의 비율

0의 근처에 많은 것으로 보아 속도가 적을수록 페널티를 적게받는 것이 적용된 것을 알 수 있다. 또 오른쪽으로 움직이려는 행동이 왼쪽으로 움직이려는 행동보다 많다.

```python
sns.distplot(select_actions)

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig10.JPG?raw=true)

<br>

#### 학습이 진행됨에따라 score 그래프

에피소드가 150이 넘어가면서는 거의 성공하는 모습을 볼 수 있다.

```python
plt.plot(scores)
plt.xlabel('episodes')
plt.ylabel('score')
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig13.JPG?raw=true)

<br>

### 큐테이블 시각화

x 위치가 0 이상일 때는 오른쪽으로 이동하려는 움직임이 많고, 반대로 0 이하일때는 왼쪽으로 이동해서 가속도를 올리려는 움직임이 보인다.

```python
q_values = []
for i in range(state_grid_count):
  q_values.append([])
  for j in range(state_grid_count):
    action_idx = np.argmax(q_table[i][j])
    q_values[i].append(actions[action_idx])

plt.figure(figsize=(8, 6))
ax = sns.heatmap(q_values, annot=True, cmap='BrBG')
ax.set_xlabel('position')
ax.set_ylabel('velocity')
xticks = env.observation_space.low[0] + range(state_grid_count+1) * \\
          abs((env.observation_space.high[0] - env.observation_space.low[0]) / state_grid_count)
xticks = [int((xticks[idx] + xticks[idx+1]) / 2 * 100) / 100 for idx, xtick in enumerate(xticks[:-1])]
ax.set_xticklabels(xticks)
yticks = env.observation_space.low[1] + range(state_grid_count+1) * \\
          abs((env.observation_space.high[1] - env.observation_space.low[1]) / state_grid_count)
yticks = [int((yticks[idx] + yticks[idx+1]) / 2 * 100) / 100 for idx, xtick in enumerate(yticks[:-1])]
ax.set_yticklabels(yticks)
ax.invert_yaxis()
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig14.JPG?raw=true)

<br>

## 딥 큐러닝 네트워크

큐러닝에 딥러닝 신경망을 적용한 것이 바로 딥 큐러닝 네트워크이다. 큐러닝은 관찰 상태와 행동에 큐테이블의 크기가 영향을 받는다는 문제점이 있었다. 연속된 문제를 풀어야하지만 grid를 나누어서 state를 정의하고, action을 취해야했다. 또 관찰상태의 차원수가 아주 많은 경우에는 그만큼 큰 저장공간이 필요하여 성능이 떨어진다.

여기서 DQN으로 풀어볼 문제는 인터넷에서 쉽게 접할 수 있는 게임 중 하나인 2048이다. 이 게임은 16개의 타일 공간(관찰 상태)에 있는 숫자들을 상하좌우로 움직여서, 2048이라는 숫자를 만드는 것이 목표인 게임이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig15.JPG?raw=true)

큐러닝은 환경을 실행하며 얻어지는 데이터로 즉시 학습했지만 DQN은 별도의 메모리에 데이터를 저장한 뒤에 어느 정도의 데이터가 쌓이면 랜덤한 샘플을 뽑아서 학습시킨다. 이렇게 하는 이유는 일반적으로 딥러닝을 포함하는 머신러닝에서 관찰의 대상인 데이터는 '상호 독립적이고 동일한 분포에 속한다'고 가정하기 때문이다. 그런데 2048같은 게임에서 앞의 상황과 뒤의 상황은 독립적이라고 하기 힘들다. 이렇게 상관관계가 있는 데이터를 사용하면 학습에도 편향이 생길 수 있다.

<br>

그리고 DQN은 하나가 아닌 2개의 큐네트워크를 사용한다. 앞절에서 유도한 DQN 계산식은 다음과 같다. 여기서 파란색으로 표시한 현재 상태의 큐함수 값 Q(s, a)를 구하기 위해 빨간색으로 표시한 다음 상태의 큐함수값인 Q(s', a') 중 최댓값을 구해야 한다. 그런데 여기서 같은 큐네트워크를 사용하면 Q(s, a)값이 업데이트될 때 가중치가 변하게 되어 Q(s', a')에도 영향을 준다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig16.JPG?raw=true)

이를 방지하기 위해 안쪽의 빨간색 네트워크를 타깃 네트워크로 분리한다. 가중치를 세타로 표시한다면 다음 식처럼 가중치만 다른 두 개의 네트워크를 사용하게 된다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig17.JPG?raw=true)

타깃 네트워크의 값은 고정시켜서 원할하게 학습되게 하고, 전체 네트워크의 학습과 너무 동떨어지지 않도록 일정 주기마다 큐네트워크의 가중치로 덮어씌운다.

<br>

### gym_2048 설치

—no-dependencies를 추가하여 depend하는 라이브러리를 설치하지 않는다. 이를 설치하면 낮은 버전의 numpy, gym 등이 설치된다.

```python
!pip install gym_2048 --no-dependencies

```

<br>

### gym_2048 환경 확인

관찰 공간은 가로 4, 세로 4 크기의 비어있는 Box 값이다. 값은 최소 2에서 최대 4294967296까지이다. 행동공간은 0~3까지 총 4가지의 행동이 가능하다. 각 숫자는 0=왼쪽, 1=위쪽, 2=오른쪽, 3=아래쪽으로 타일들을 움직이는 행동을 나타낸다.

```python
import gym_2048
import gym

env = gym.make('2048-v0')
obs = env.reset()

print(obs)
print(env.observation_space)
print(env.action_space)
#---------------------출력---------------------#
[[2 0 0 0]
 [0 0 0 0]
 [0 2 0 0]
 [0 0 0 0]]
Box(2, 4294967296, (4, 4), int64)
Discrete(4)

```

<br>

### 랜덤 행동 에이전트의 실행 결과 확인

총 점수는 668점, step은 86까지 진행되었다.

```python
score = 0
step = 0
obs = env.reset()

while True:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)

  score += reward
  step += 1
  if done:
    break

print('score:', score, 'step:', step)
print(obs)
#---------------------출력---------------------#
score: 668 step: 86
[[ 2  4  8  2]
 [32  8  2  4]
 [ 2 32 64 16]
 [ 4  2 16  2]]

```

<br>

### 2048 게임 보드의 원-핫 인코딩 함수

2048과 비슷하게 생긴 게임 보드를 인코딩하는 방법으로 딥마인드의 알파고 등에서 사용한 원-핫 인코딩을 써볼 수 있다. 여기서는 하나의 보드판을 12개의 레이어(0, 2, 4, ... , 1024, 2048)로 나눈다. 그리고 각각 해당 타일이 있으며 1, 없으며 0으로 저장하는 것이다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig18.JPG?raw=true)

```python
layer_count = 12
table = {2**i:i for i in range(layer_count)}
print(table)

def preprocess(obs):
  x = np.zeros((4, 4, layer_count))
  for i in range(4):
    for j in range(4):
      if obs[i, j] > 0:
        v = min(obs[i, j], 2**(layer_count-1))
        x[i,j,table[v]] = 1
      else:
        x[i, j, 0] = 1
  return x
#---------------------출력---------------------#
{1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7, 256: 8, 512: 9, 1024: 10, 2048: 11}

```

<br>

### 모델 정의 함수

딥 큐러닝 네트워크를 정의할 때 게임 보드가 가진 공간 정보를 활용하기 위해 컨볼루션 신경망을 사용한다. 원-핫 인코딩을 거친 (4,4,12)의 크기를 가진 데이터는 각각 (1,2) 크기의 커널과 (2,1) 크기의 커널을 가진 컨볼루션 레이어를 통과한다. 그리고 두 번째로 각각 (1,2), (2,1) 크기의 커널을 가진 컨볼루션 레이어를 통과한다.

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig19.JPG?raw=true)

```python
layer_count = 12
def build_model():
  dense1 = 128
  dense2 = 128

  x = tf.keras.Input(shape=(4, 4, layer_count))

  conv_a = tf.keras.layers.Conv2D(dense1, kernel_size=(2,1), activation='relu')(x)
  conv_b = tf.keras.layers.Conv2D(dense1, kernel_size=(1,2), activation='relu')(x)
  conv_aa = tf.keras.layers.Conv2D(dense2, kernel_size=(2,1), activation='relu')(conv_a)
  conv_ab = tf.keras.layers.Conv2D(dense2, kernel_size=(1,2), activation='relu')(conv_a)
  conv_ba = tf.keras.layers.Conv2D(dense2, kernel_size=(2,1), activation='relu')(conv_b)
  conv_bb = tf.keras.layers.Conv2D(dense2, kernel_size=(1,2), activation='relu')(conv_b)

  flat = [tf.keras.layers.Flatten()(a) for a in [conv_a, conv_b, conv_aa, conv_ab,
                                                 conv_ba, conv_bb]]
  
  concat = tf.keras.layers.Concatenate()(flat)
  dense1 = tf.keras.layers.Dense(256, activation='relu')(concat)
  out = tf.keras.layers.Dense(4, activation='linear')(dense1)

  model=  tf.keras.Model(inputs=x, outputs=out)
  model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=0.0005), loss='mse')
  model.summary()
  return model

model = build_model()
target_model = build_model()
#---------------------출력---------------------#
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            [(None, 4, 4, 12)]   0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 3, 4, 128)    3200        input_2[0][0]                    
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 4, 3, 128)    3200        input_2[0][0]                    
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 2, 4, 128)    32896       conv2d[0][0]                     
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 3, 3, 128)    32896       conv2d[0][0]                     
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 3, 3, 128)    32896       conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 4, 2, 128)    32896       conv2d_1[0][0]                   
__________________________________________________________________________________________________
flatten (Flatten)               (None, 1536)         0           conv2d[0][0]                     
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 1536)         0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
flatten_2 (Flatten)             (None, 1024)         0           conv2d_2[0][0]                   
__________________________________________________________________________________________________
flatten_3 (Flatten)             (None, 1152)         0           conv2d_3[0][0]                   
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 1152)         0           conv2d_4[0][0]                   
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 1024)         0           conv2d_5[0][0]                   
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 7424)         0           flatten[0][0]                    
                                                                 flatten_1[0][0]                  
                                                                 flatten_2[0][0]                  
                                                                 flatten_3[0][0]                  
                                                                 flatten_4[0][0]                  
                                                                 flatten_5[0][0]                  
__________________________________________________________________________________________________
dense (Dense)                   (None, 256)          1900800     concatenate[0][0]                
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 4)            1028        dense[0][0]                      
==================================================================================================
Total params: 2,039,812
Trainable params: 2,039,812
Non-trainable params: 0

```

<br>

### 학습 함수 정의

```python
gamma = 0.9
batch_size = 512
max_memory = batch_size*8
memory = []

def append_sample(state, action, reward, next_state, done):
  memory.append([state, action, reward, next_state, done])

def train_model():
  np.random.shuffle(memory)

  len = max_memory // batch_size
  for k in range(len):
    mini_batch = memory[k*batch_size:(k+1)*batch_size]

    states = np.zeros((batch_size, 4, 4, layer_count))
    next_states = np.zeros((batch_size, 4, 4, layer_count))
    actions, rewards, done = [], [], []

    for i in range(batch_size):
      states[i] = mini_batch[i][0]
      actions.append(mini_batch[i][1])
      rewards.append(mini_batch[i][2])
      next_states[i] = mini_batch[i][3]
      dones.append(mini_batch[i][4])

    target = model.predict(states)
    next_target = target_model.predicT(next_states)

    for i in range(batch_size):
      if dones[i]:
        target[i][actions[i]] = rewards[i]
      else:
        target[i][actions[i]] = rewards[i] + gamma * np.amax(next_target[i])

    model.fit(states, target, batch_size=batch_size, epochs=1, verbose=0)

```

<br>

### DQN 학습

네트워크를 10001 에피소드 동안 학습시킨다. epsilon은 시작값을 0.9로, 최솟값을 0.01로 두고 그 사이에서 변화하도록 조절한다.

보상부분은 이전 상태의 최대 타일과 현재 상태의 최대 타일을 비교해서 현재의 최대 타일이 더 클 경우 새로운 보상을 부여한다. 그렇지않을 경우 보상은 일괄적으로 0이 된다. 그 다음에는 보드에 깔린 타일의 숫자를 얼마나 줄였는지, 즉 타일을 얼마나 합쳤는지에 대한 보상을 구해서 더한다.

새롭게 구한 보상과 이전 상태, 현재 상태, 행동, 게임 종료 여부를 append_sample()함수를 사용해 메모리에 저장한다. 메모리의 크기가 max_memory보다 크거나 같으면 학습을 실행하고 메모리는 비운다.

```python
import math

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    sum_exp_logits = np.sum(exp_logits)
    return exp_logits / sum_exp_logits

max_episodes = 10001
epsilon = 0.9
epsilon_min = 0.1

scores = []
steps = []
iteration = 0

train_count = 0

for i in range(max_episodes):
    if i % 100 == 0 and i != 0:
        print('score mean:', np.mean(scores[-100:]), 'step mean:', np.mean(steps[-100:]), 'iteration:', iteration, 'epsilon:', epsilon)

    prev_obs = env.reset()

    score = 0
    step = 0
    not_move_list = np.array([1,1,1,1])
    prev_max = np.max(prev_obs)

    while True:
        iteration += 1

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            x = preprocess(prev_obs)
            logits = model.predict(np.expand_dims(x, axis=0))[0]
            prob = softmax(logits)
            prob = prob * not_move_list
            action = np.argmax(prob)

        obs, reward, done, info = env.step(action)

        score += reward
        step += 1

        # not moved situation
        if reward == 0 and np.array_equal(obs, prev_obs):
            not_move_list[action] = 0
            continue
        else:
            not_move_list = np.array([1,1,1,1])

        # custom reward
        now_max = np.max(obs)
        if prev_max < now_max:
            prev_max = now_max
            reward = math.log(now_max, 2) * 0.1
        else:
            reward = 0

        reward += np.count_nonzero(prev_obs) - np.count_nonzero(obs) + 1

        append_sample(preprocess(prev_obs), action, reward, preprocess(obs), done)

        if len(memory) >= max_memory:
            train_model()
            memory = []

            train_count += 1
            if train_count % 4 == 0:
                target_model.set_weights(model.get_weights())

        prev_obs = obs

        if epsilon > 0.01 and iteration % 2500 == 0:
            epsilon = epsilon / 1.005

        if done:
            break

    scores.append(score)
    steps.append(step)

    # print(i, 'score:', score, 'step:', step, 'max tile:', np.max(obs), 'memory len:', len(memory))

```

<br>

### 점수 확인

왼쪽 그래프는 각 에피소드에 획득한 점수를 산점도로 나타냈다. 학습의 처음부터 끝까지 고르게 분포하고 있다. 오른쪽 그래프는 이동 평균을 나타낸 것으로, 100에피소드의 평균을 연속적으로 계산한 것이다. 점수가 계선되는 추세가 뚜렸하다.

```python
import matplotlib.pyplot as plt

N = 100
rolling_mean = [np.mean(scores[x:x+N]) for x in range(len(scores)-N+1)]

plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.scatter(range(len(scores)), scores, marker='.')
plt.subplot(1, 2, 2)
plt.plot(rolling_mean)
plt.show()

```

![](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-03-12-ch10_Reinforcement_Learning/fig20.JPG?raw=true)

<br>

### 평가(1000 게임 테스트)

```python
test_scores = []
max_tile = {}
iteration = 0
train_count = 0

for i in range(1000):
  if i % 100 == 0 and i != 0:
    print('score mean:', np.mean(scores[-100:]), 'step mean:', np.mean(steps[-100]), 
          'iteration:', iteration, 'epsilon:', epsilon)
    
  prev_obs = env.reset()

  score = 0
  step = 0
  not_move_list = np.array([1,1,1,1])

  while True:
    iteration += 1

    x = preprocess(prev_obs)
    logits = model.predict(np.expand_dims(x, axis=0))[0]
    prob = softmax(logits)
    prob = prob * not_move_list
    action = np.argmax(prob)

    obs, reward, done, info = env.step(action)

    score += reward
    step += 1

    # not moved situation
    if reward == 0 and np.array_equal(obs, prev_obs):
      not_move_list[action] = 0
      continure
    else:
      not_move_list = np.array([1,1,1,1])

    prev_obs = obs

    if done:
      now_max = np.max(obs)
      max_tile[now_max] = max_tile.get(now_max, 0) + 1
      break

  test_scores.append(score)
  print(i, 'score:', score, 'step:', step, 'max tile:', np.max(obs), 'memory len:',
        len(memory))
  
print(max_tile)
#---------------------출력---------------------#
{1024: 488, 512: 403, 2048: 50, 256: 58, 128: 1}

```

<br>

### 학습 데이터 보강

성능 개선을 위해 data augmentaion 기법을 사용할 수 있다. 하나의 게임 보드를 좌우 반전이나 상하 반전, 대각선 반전 등으로 뒤집어서 데이터의 수를 늘려준다. action_swap_array는 state가 바뀜에 따라 action을 어떻게 변화해야할지 나타낸다.

```python
max_memory = 512*64

action_swap_array = [[0, 0, 2, 2, 1, 3, 1, 3],
                     [1, 3, 1, 3, 0, 0, 2, 2],
                     [2, 2, 0, 0, 3, 1, 3, 1],
                     [3, 1, 3, 1, 2, 2, 0, 0]]

def append_sample(state, action, reward, next_state, done):
    g0 = state
    g1 = g0[::-1,:,:]
    g2 = g0[:,::-1,:]
    g3 = g2[::-1,:,:]
    r0 = state.swapaxes(0,1)
    r1 = r0[::-1,:,:]
    r2 = r0[:,::-1,:]
    r3 = r2[::-1,:,:]

    g00 = next_state
    g10 = g00[::-1,:,:]
    g20 = g00[:,::-1,:]
    g30 = g20[::-1,:,:]
    r00 = next_state.swapaxes(0,1)
    r10 = r00[::-1,:,:]
    r20 = r00[:,::-1,:]
    r30 = r20[::-1,:,:]

    states = [g0, g1, g2, g3, r0, r1, r2, r3]
    next_states = [g00, g10, g20, g30, r00, r10, r20, r30]

    for i in range(8):
        memory.append([
            states[i],
            action_swap_array[action][i],
            reward,
            next_states[i],
            done
        ])

```