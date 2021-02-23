---
layout: post
title: Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning review
featured-img: 2020-08-18-Motion_Planning_Among_Dynamic,_Decision_Making_Agents_with_Deep_Reinforcement_Learning/fig1
permalink: /paper_review/Motion_Planning_Among_Dynamic_Decision_Making_Agents_with_Deep_Reinforcement_Learning/
category: paper_review
---

#### 본 포스트는 Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning의 리뷰이다.

<style>.embed-container { position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; } .embed-container iframe, .embed-container object, .embed-container embed { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }</style><div class='embed-container'><iframe src='https://www.youtube.com/embed/XHoXkWLhwYQ' frameborder='1' allowfullscreen></iframe></div>

## Abstact
로봇에서 사람들 사이를 부딪히지 않으면서 안전하고, 효과적으로 주행하는 것은 중요한 문제이다. 본 연구에서는 LSTM을 사용해서 사람들의 수가 몇 명이든지 주행할 수 있었다. 또한 이 알고리즘은 시뮬레이션 그리고 현실에서도 사람들 사이를 주행을 부드럽게 할 수 있었다.

 사람들을 피하는 방법은 다음과 같은 단계로 발전하였다.
 1. 사람들이 고정되었다고 가정하고, 회피할 수 있을만큼 재빨리 re-plan
 2. 사람들을 움직인다고 생각하지만, 같은 속도로 운동한다고 가정
 3. 사람들이 known or unknown policies에 따라서 움직이는 decision-makers로 가정한다. 그러나 이 방법은 목적지 등과 같은 모르는 값이 있으면 경로를 결정할 수 없다.
 4. 다른 사람들의 행동을 예측하는 대신 강화학습을 사용하여 복잡한 interaction과 cooperation among agents를 알아서 학습하도록 함.

### A. related work
충돌을 회피하는 문제는 reaction-based 방법과 trajectory-based 방법으로 나눌 수 있다. reaction-based 방법은 경로를 geometry, physics적으로 확실하게 회피하지 않는 행동을 결정하는 것이다. short sighted in time이라는 단점이 있다.
trajectory-based방법은 longer timescale에서 경로를 계산하는 것이다. 그러나 이 방법은 computationally expensive하고 unobservable information(agents' destination)이 필요하다는 단점이 있다.

이 두가지의 장점만 가져오는 것이 강화학습을 이용한 방법이다. 강화학습을 통해서 state의 value를 학습하여 좋은 행동을 결정할 수 있다.

강화학습을 이용한 방법도 여러가지로 나뉜다. 먼저 raw sensor data(camera image, laser sensor data)를 input으로 넣어주어 action을 얻는 방법이다. 이 방법의 장점은  고정된 물체와 움직이는 물체의 회피를 하나의 framedwork로 구축할 수 있다는 것이다. 그러나 좀 더 agent level에서 정보를 추출하는 방법이 더 유용하다. 만약 laser scan데이터를 input으로 넣어준다면 쓰레기통과 멈춰있는 사람은 비슷하게 보인다. 이럴경우 쓰레기통으로 부터 멀리 주행하는 모습을 볼 수 있다.

Agent level에서 정보를 추출해서 넣어주는 방법의 문제 중 하나는 agent의 수이다. 이 문제의 해결법중 하나가 maximum number of agents를 정해주는 것이다. 그러나 이는 복잡한 환경에서는 주행이 불안정하다는 단점이 있다. 

### B. Collision Avoidance with Deep RL(CADRL)

로봇의 state :  <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{s}_{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{s}_{t}" title="\mathbf{s}_{t}" /></a>    
로봇의 action : <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{u}_{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{u}_{t}" title="\mathbf{u}_{t}" /></a>   
other agent의 state: <a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{\mathbf{S}_{t}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{\mathbf{S}_{t}}" title="\tilde{\mathbf{S}_{t}}" /></a>    
state벡터 : <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{s}_{t}=\left[\mathbf{s}_{t}^{o},&space;\mathbf{s}_{t}^{h}\right]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{s}_{t}=\left[\mathbf{s}_{t}^{o},&space;\mathbf{s}_{t}^{h}\right]" title="\mathbf{s}_{t}=\left[\mathbf{s}_{t}^{o}, \mathbf{s}_{t}^{h}\right]" /></a>    
관측가능한 state : <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{s}^{o}=\left[p_{x},&space;p_{y},&space;v_{x},&space;v_{y},&space;r\right]&space;\in&space;\mathbb{R}^{5}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{s}^{o}=\left[p_{x},&space;p_{y},&space;v_{x},&space;v_{y},&space;r\right]&space;\in&space;\mathbb{R}^{5}" title="\mathbf{s}^{o}=\left[p_{x}, p_{y}, v_{x}, v_{y}, r\right] \in \mathbb{R}^{5}" /></a>(x_position, y_position, x_velocity, y_velocity, radius)       
관측불가능한 state : <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{s}^{h}=\left[p_{g&space;x},&space;p_{g&space;y},&space;v_{p&space;r&space;e&space;f},&space;\psi\right]&space;\in&space;\mathbb{R}^{4}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{s}^{h}=\left[p_{g&space;x},&space;p_{g&space;y},&space;v_{p&space;r&space;e&space;f},&space;\psi\right]&space;\in&space;\mathbb{R}^{4}" title="\mathbf{s}^{h}=\left[p_{g x}, p_{g y}, v_{p r e f}, \psi\right] \in \mathbb{R}^{4}" /></a>(x_goal_position, y_goal_position, preferred speed, orientation)    
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{u}_{t}=\left[v_{t},&space;\psi_{t}\right]&space;\in&space;\mathbb{R}^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{u}_{t}=\left[v_{t},&space;\psi_{t}\right]&space;\in&space;\mathbb{R}^{2}" title="\mathbf{u}_{t}=\left[v_{t}, \psi_{t}\right] \in \mathbb{R}^{2}" /></a>(속도, heading angle)    
policy : <a href="https://www.codecogs.com/eqnedit.php?latex=\pi:\left(\mathbf{s}_{t},&space;\tilde{\mathbf{s}}_{t}^{o}\right)&space;\mapsto&space;\mathbf{u}_{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi:\left(\mathbf{s}_{t},&space;\tilde{\mathbf{s}}_{t}^{o}\right)&space;\mapsto&space;\mathbf{u}_{t}" title="\pi:\left(\mathbf{s}_{t}, \tilde{\mathbf{s}}_{t}^{o}\right) \mapsto \mathbf{u}_{t}" /></a>    
목표는 goal까지 충돌하지 않으면서 도달하는데 걸리는 시간의 기댓값 <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbb{E}\left[t_{g}\right]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbb{E}\left[t_{g}\right]" title="\mathbb{E}\left[t_{g}\right]" /></a>를  최소로 만드는 것이다. 수식으로 쓰면 다음과 같다. 
    
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\underset{\pi\left(\mathbf{s},&space;\tilde{\mathbf{s}}^{o}\right)}{\operatorname{argmin}}&space;&&space;\mathbb{E}\left[t_{g}&space;\mid&space;\mathbf{s}_{0},&space;\tilde{\mathbf{s}}_{0}^{o},&space;\pi\right]&space;\\&space;\text&space;{&space;s.t.&space;}&space;&\left\|\mathbf{p}_{t}-\tilde{\mathbf{p}}_{t}\right\|_{2}&space;\geq&space;r&plus;\tilde{r}&space;\quad&space;\forall&space;t&space;\\&space;&&space;\mathbf{p}_{t_{g}}=\mathbf{p}_{g}&space;\\&space;&&space;\mathbf{p}_{t}=\mathbf{p}_{t-1}&plus;\Delta&space;t&space;\cdot&space;\pi\left(\mathbf{s}_{t-1},&space;\tilde{\mathbf{s}}_{t-1}^{o}\right)&space;\\&space;&&space;\tilde{\mathbf{p}}_{t}=\tilde{\mathbf{p}}_{t-1}&plus;\Delta&space;t&space;\cdot&space;\pi\left(\tilde{\mathbf{s}}_{t-1},&space;\mathbf{s}_{t-1}^{o}\right)&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\underset{\pi\left(\mathbf{s},&space;\tilde{\mathbf{s}}^{o}\right)}{\operatorname{argmin}}&space;&&space;\mathbb{E}\left[t_{g}&space;\mid&space;\mathbf{s}_{0},&space;\tilde{\mathbf{s}}_{0}^{o},&space;\pi\right]&space;\\&space;\text&space;{&space;s.t.&space;}&space;&\left\|\mathbf{p}_{t}-\tilde{\mathbf{p}}_{t}\right\|_{2}&space;\geq&space;r&plus;\tilde{r}&space;\quad&space;\forall&space;t&space;\\&space;&&space;\mathbf{p}_{t_{g}}=\mathbf{p}_{g}&space;\\&space;&&space;\mathbf{p}_{t}=\mathbf{p}_{t-1}&plus;\Delta&space;t&space;\cdot&space;\pi\left(\mathbf{s}_{t-1},&space;\tilde{\mathbf{s}}_{t-1}^{o}\right)&space;\\&space;&&space;\tilde{\mathbf{p}}_{t}=\tilde{\mathbf{p}}_{t-1}&plus;\Delta&space;t&space;\cdot&space;\pi\left(\tilde{\mathbf{s}}_{t-1},&space;\mathbf{s}_{t-1}^{o}\right)&space;\end{aligned}" title="\begin{aligned} \underset{\pi\left(\mathbf{s}, \tilde{\mathbf{s}}^{o}\right)}{\operatorname{argmin}} & \mathbb{E}\left[t_{g} \mid \mathbf{s}_{0}, \tilde{\mathbf{s}}_{0}^{o}, \pi\right] \\ \text { s.t. } &\left\|\mathbf{p}_{t}-\tilde{\mathbf{p}}_{t}\right\|_{2} \geq r+\tilde{r} \quad \forall t \\ & \mathbf{p}_{t_{g}}=\mathbf{p}_{g} \\ & \mathbf{p}_{t}=\mathbf{p}_{t-1}+\Delta t \cdot \pi\left(\mathbf{s}_{t-1}, \tilde{\mathbf{s}}_{t-1}^{o}\right) \\ & \tilde{\mathbf{p}}_{t}=\tilde{\mathbf{p}}_{t-1}+\Delta t \cdot \pi\left(\tilde{\mathbf{s}}_{t-1}, \mathbf{s}_{t-1}^{o}\right) \end{aligned}" /></a>

부딪히면 negative reward를 받고, 골에 도달하면 positive reward를 받는다.    
reward : <img src="https://latex.codecogs.com/gif.latex?R_{c&space;o&space;l}\left(\mathbf{s}^{j&space;n},&space;\mathbf{u}\right)" title="R_{c o l}\left(\mathbf{s}^{j n}, \mathbf{u}\right)" />

다른 논문에서는 optimal value function을 학습하는 것을 목표로 가지고 있었다. 그러나 value function은 직관적으로 policy와 연결되지 않는다. optimal policy는 optimal value function으로 다음과 같이 표현된다.    
     
<img src="https://latex.codecogs.com/gif.latex?\pi^{*}\left(\mathbf{s}_{t}^{j&space;n}\right)=\underset{\mathbf{u}}{\operatorname{argmax}}&space;R_{c&space;o&space;l}\left(\mathbf{s}_{t},&space;\mathbf{u}\right)&plus;\gamma^{\Delta&space;t&space;\cdot&space;v_{p&space;r&space;e&space;f}}&space;\int_{\mathbf{s}_{t}^{j&space;n}}&space;P\left(\mathbf{s}_{t&plus;1}^{j&space;n}&space;\mid&space;\mathbf{s}_{t}^{j&space;n},&space;\mathbf{u}\right)&space;V^{*}\left(\mathbf{s}_{t&plus;1}^{j&space;n}\right)&space;d&space;\mathbf{s}_{t&plus;1}^{j&space;n}" title="\pi^{*}\left(\mathbf{s}_{t}^{j n}\right)=\underset{\mathbf{u}}{\operatorname{argmax}} R_{c o l}\left(\mathbf{s}_{t}, \mathbf{u}\right)+\gamma^{\Delta t \cdot v_{p r e f}} \int_{\mathbf{s}_{t}^{j n}} P\left(\mathbf{s}_{t+1}^{j n} \mid \mathbf{s}_{t}^{j n}, \mathbf{u}\right) V^{*}\left(\mathbf{s}_{t+1}^{j n}\right) d \mathbf{s}_{t+1}^{j n}" />    
    
이 방법은 current state <img src="https://latex.codecogs.com/gif.latex?\mathbf{s}_{t}^{j&space;n}" title="\mathbf{s}_{t}^{j n}" />로 부터 다음 step의 state <img src="https://latex.codecogs.com/gif.latex?\mathbf{s}_{t&plus;1}^{j&space;n}" title="\mathbf{s}_{t+1}^{j n}" />를 예측한다. 그리고 <img src="https://latex.codecogs.com/gif.latex?V^{*}\left(\mathbf{s}_{t&plus;1}^{j&space;n}\right)" title="V^{*}\left(\mathbf{s}_{t+1}^{j n}\right)" />를 최대로 하는 action u를 선택한다.
그러나 collision avoidance문제에서는 다른 agents의 policies와 의도를 모른다. 이것은 state transition <img src="https://latex.codecogs.com/gif.latex?P\left(\mathbf{s}_{t&plus;1}^{j&space;n}&space;\mid&space;\mathbf{s}_{t}^{j&space;n},&space;\mathbf{u}\right)" title="P\left(\mathbf{s}_{t+1}^{j n} \mid \mathbf{s}_{t}^{j n}, \mathbf{u}\right)" />이 유동적이라는 것이다.

이전의 연구에서 사용한 다른 agents가 현재 속도인 <img src="https://latex.codecogs.com/gif.latex?\hat{\mathbf{v}}_{t}" title="\hat{\mathbf{v}}_{t}" />로 <img src="https://latex.codecogs.com/gif.latex?\Delta&space;t" title="\Delta t" /> 시간동안 계속 움직인다고 가정하면 다음과 같이 정리할 수 있다.   
    
<img src="https://latex.codecogs.com/gif.latex?\hat{\mathbf{s}}_{t&plus;1,&space;\mathbf{u}}^{j&space;n}&space;\leftarrow\left[\right.&space;propagate&space;\left(\mathbf{s}_{t},&space;\Delta&space;t&space;\cdot&space;\mathbf{u}\right),&space;propagate&space;\left.\left(\tilde{\mathbf{s}}_{t}^{o},&space;\Delta&space;t&space;\cdot&space;\hat{\mathbf{v}}_{t}\right)\right]" title="\hat{\mathbf{s}}_{t+1, \mathbf{u}}^{j n} \leftarrow\left[\right. propagate \left(\mathbf{s}_{t}, \Delta t \cdot \mathbf{u}\right), propagate \left.\left(\tilde{\mathbf{s}}_{t}^{o}, \Delta t \cdot \hat{\mathbf{v}}_{t}\right)\right]" />    
     
<img src="https://latex.codecogs.com/gif.latex?\pi_{C&space;A&space;D&space;R&space;L}^{*}\left(\mathbf{s}_{t}^{j&space;n}\right)=\underset{\mathbf{u}}{\operatorname{argmax}}&space;R_{c&space;o&space;l}\left(\mathbf{s}_{t},&space;\mathbf{u}\right)&plus;\gamma^{\Delta&space;t&space;\cdot&space;v_{p&space;r&space;e&space;f}}&space;V^{*}\left(\hat{\mathbf{s}}_{t&plus;1,&space;\mathbf{u}}^{j&space;n}\right)" title="\pi_{C A D R L}^{*}\left(\mathbf{s}_{t}^{j n}\right)=\underset{\mathbf{u}}{\operatorname{argmax}} R_{c o l}\left(\mathbf{s}_{t}, \mathbf{u}\right)+\gamma^{\Delta t \cdot v_{p r e f}} V^{*}\left(\hat{\mathbf{s}}_{t+1, \mathbf{u}}^{j n}\right)" />    
여기서 <img src="https://latex.codecogs.com/gif.latex?\Delta&space;t" title="\Delta t" />는 조정하기 어려운 trade off를 가지고 있다. 너무 큰 <img src="https://latex.codecogs.com/gif.latex?\Delta&space;t" title="\Delta t" />는 <img src="https://latex.codecogs.com/gif.latex?V^{*}\left(\mathbf{s}_{t&plus;1,&space;\mathbf{u}}^{j&space;n}\right)" title="V^{*}\left(\mathbf{s}_{t+1, \mathbf{u}}^{j n}\right)" />의 정확도를 떨어뜨린다. 또한 agents가 많은 환경에서는 일정한 속도라는 가정이 의미가 없어진다. 너무 작은 <img src="https://latex.codecogs.com/gif.latex?\Delta&space;t" title="\Delta t" />는 value의 비교가 힘들어진다. <img src="https://latex.codecogs.com/gif.latex?\Delta&space;t" title="\Delta t" />가 너무 작거나 크면 수렴하지도 않는다. ∆t = 1sec가 경험적으로 가장 적당했다. <img src="https://latex.codecogs.com/gif.latex?\Delta&space;t" title="\Delta t" />의 수정이 필요하다는 점은 다른 RL 프레임워크의 motivation이 되었다.

### C. Policy-Based Learning
이 방법은 state transition의 추정이 없이 바로 policy를 생성하는 것이다. 최근 유행하는 A3C는 value와 policy를 모두 추정한다. 학습에는 다음과 같은 두 가지 loss function이 사용된다.   
<img src="https://latex.codecogs.com/gif.latex?f_{v}=\left(R_{t}-V\left(\mathbf{s}_{t}^{j&space;n}\right)\right)^{2}" title="f_{v}=\left(R_{t}-V\left(\mathbf{s}_{t}^{j n}\right)\right)^{2}" />   
<img src="https://latex.codecogs.com/gif.latex?f_{\pi}=\log&space;\pi\left(\mathbf{u}_{t}&space;\mid&space;\mathbf{s}_{t}^{j&space;n}\right)\left(R_{t}-V\left(\mathbf{s}_{t}^{j&space;n}\right)\right)&plus;\beta&space;\cdot&space;H\left(\pi\left(\mathbf{s}_{t}^{j&space;n}\right)\right)" title="f_{\pi}=\log \pi\left(\mathbf{u}_{t} \mid \mathbf{s}_{t}^{j n}\right)\left(R_{t}-V\left(\mathbf{s}_{t}^{j n}\right)\right)+\beta \cdot H\left(\pi\left(\mathbf{s}_{t}^{j n}\right)\right)" />   
Reward estimate <img src="https://latex.codecogs.com/gif.latex?R_{t}=\sum_{i=0}^{k-1}&space;\gamma^{i}&space;r_{t&plus;i}&plus;\gamma^{k}&space;V\left(\mathbf{s}_{t&plus;k}^{j&space;n}\right)" title="R_{t}=\sum_{i=0}^{k-1} \gamma^{i} r_{t+i}+\gamma^{k} V\left(\mathbf{s}_{t+k}^{j n}\right)" />   

A3C에서는 병렬적으로 각 threads에 배치된 환경에서 각각의 agents가 결정을 내린다. A3C에서는 CPU상에서 병렬적으로 처리했던 것을 training experiences를 더 빠르게 병렬로 처리하기 위해 GPU에서 실행한 것을 GA3C라고 부른다. 

## Approach
### A. GA3C-CADRL
이 approach의 목적은 optimal한 policy <img src="https://latex.codecogs.com/gif.latex?\pi:\left(\mathrm{s}_{t},&space;\tilde{\mathrm{s}_{t}}\right)&space;\mapsto&space;\mathbf{u}_{t}" title="\pi:\left(\mathrm{s}_{t}, \tilde{\mathrm{s}_{t}}\right) \mapsto \mathbf{u}_{t}" />를 찾는 것이다. 

state는 agent의 관측 가능한 특성과 관측불가능한 특성으로 나눈다.    
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\mathbf{s}&space;&=\left[d_{g},&space;v_{p&space;r&space;e&space;f},&space;\psi,&space;r\right]&space;\\&space;\tilde{\mathbf{s}}^{o}&space;&=\left[\tilde{p}_{x},&space;\tilde{p}_{y},&space;\tilde{v}_{x},&space;\tilde{v}_{y},&space;\tilde{r},&space;\tilde{d}_{a},&space;\tilde{r}&plus;r\right]&space;\end{aligned}" title="\begin{aligned} \mathbf{s} &=\left[d_{g}, v_{p r e f}, \psi, r\right] \\ \tilde{\mathbf{s}}^{o} &=\left[\tilde{p}_{x}, \tilde{p}_{y}, \tilde{v}_{x}, \tilde{v}_{y}, \tilde{r}, \tilde{d}_{a}, \tilde{r}+r\right] \end{aligned}" />   
<img src="https://latex.codecogs.com/gif.latex?d_{g}=\left\|\mathbf{p}_{g}-\mathbf{p}\right\|_{2}" title="d_{g}=\left\|\mathbf{p}_{g}-\mathbf{p}\right\|_{2}" />는 agent와 goal사이의 거리이다.    
<img src="https://latex.codecogs.com/gif.latex?\tilde{d}_{a}=\|\mathbf{p}-\tilde{\mathbf{p}}\|_{2}" title="\tilde{d}_{a}=\|\mathbf{p}-\tilde{\mathbf{p}}\|_{2}" />는 다른 agent까지의 거리이다.
   
agent의 action space는 속도와 heading angle로 이뤄져있다. 총 11개의 action으로 나눴다.     
<img src="https://latex.codecogs.com/gif.latex?v_{\text&space;{pref}}" title="v_{\text {pref}}" />: <img src="https://latex.codecogs.com/gif.latex?\pm\pi" title="\pm\pi" />/3
<img src="https://latex.codecogs.com/gif.latex?\pm\pi" title="\pm\pi" />/6, 0   
<img src="https://latex.codecogs.com/gif.latex?\frac{1}{2}&space;v_{p&space;r&space;e&space;f}" title="\frac{1}{2} v_{p r e f}" />, 0 : <img src="https://latex.codecogs.com/gif.latex?\pm\pi" title="\pm\pi" />/6, 0   
   
Reward function은 다음과 같다. 
<img src="https://latex.codecogs.com/gif.latex?R_{c&space;o&space;l}\left(\mathbf{s}^{j&space;n}\right)=\left\{\begin{array}{ll}1&space;&&space;\text&space;{&space;if&space;}&space;\mathbf{p}=\mathbf{p}_{g}&space;\\&space;-0.25&space;&&space;\text&space;{&space;if&space;}&space;d_{\min&space;}<0&space;\\&space;-0.1&plus;0.05&space;\cdot&space;d_{\min&space;}&space;&&space;\text&space;{&space;if&space;}&space;0<d_{\min&space;}<0.2&space;\\&space;0&space;&&space;\text&space;{&space;otherwise&space;}\end{array}\right." title="R_{c o l}\left(\mathbf{s}^{j n}\right)=\left\{\begin{array}{ll}1 & \text { if } \mathbf{p}=\mathbf{p}_{g} \\ -0.25 & \text { if } d_{\min }<0 \\ -0.1+0.05 \cdot d_{\min } & \text { if } 0<d_{\min }<0.2 \\ 0 & \text { otherwise }\end{array}\right." />   
   
<img src="https://latex.codecogs.com/gif.latex?d_{\min&space;}" title="d_{\min }" />은 가장 가까운 agent까지의 거리다.    
각 GPU의 threads에서는 실행된 experience(<img src="https://latex.codecogs.com/gif.latex?\left\{\mathbf{s}_{t}^{j&space;n},&space;\mathbf{u}_{t},&space;r_{t}\right\}" title="\left\{\mathbf{s}_{t}^{j n}, \mathbf{u}_{t}, r_{t}\right\}" />)를 모은다. 

### B. Handling a Variable Number of Agents
여러 learning-based collision avoidance methods의 문제점은 고정된 input size이다. 여기서는 Long short-term memory (LSTM)를 이용한다. LSTM을 이용하면 몇 개의 input이 들어오더라도 상관없이 fixed size output을 낼 수 있다. 

![LSTM](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-08-18-Motion_Planning_Among_Dynamic,_Decision_Making_Agents_with_Deep_Reinforcement_Learning/fig2.JPG?raw=true)   
다음 그림과 같이  other agents의 observable states가 순차적으로 들어간다. LSTM은 새로운 input이 들어감에 따라서 오래된 정보는 잊혀지는 경향이 있다. 이런 점을 보완하고자 hidden state의 크기를 충분히 크게 해준다. 또한 agent로부터 거리가 먼 순서부터 넣도록 한다. 

전체적인 모델은 다음과 같다.   
![Full Model](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-08-18-Motion_Planning_Among_Dynamic,_Decision_Making_Agents_with_Deep_Reinforcement_Learning/fig3.JPG?raw=true)   
### C. Training the Policy
본 연구에서는 모델의 파라미터를 초기화시키기 위해 첫번째로 supervised learning을 했다. 이전의 연구인 CADRL을 통해 얻은 state-action-value pairs를 데이터로 하여 value를 예측하고 policy를 예측하는 모델을 훈련시켰다.  
다음은 시뮬레이션을 통해 훈련할 수 있는 환경을 만들었다. 여기서 agent의 파라미터는 사람과 비슷하도록 <img src="https://latex.codecogs.com/gif.latex?r&space;\in[0.2,0.8]&space;\mathrm{m}" title="r \in[0.2,0.8] \mathrm{m}" />, <img src="https://latex.codecogs.com/gif.latex?v_{p&space;r&space;e&space;f}&space;\in[0.5,2.0]&space;\mathrm{m}&space;/&space;\mathrm{s}" title="v_{p r e f} \in[0.5,2.0] \mathrm{m} / \mathrm{s}" />로 구성했다. 
처음에는 2-4 agents의 환경에서 훈련을 시켰고, 이것으로 어느정도 수렴이 된 이후에 2-10 agents의 환경에서 훈련시켰다.

## Results
### A. Computational Details
이전 연구에서 미래의 state를 예측하고, policy를 계산하는 과정이 필요했던 반면에 이번 연구에서는 바로 policy를 예측한다. i7 cpu에서 돌렸을 때 한 query당 0.4~0.5ms가 걸렸고, 이는 이전보다 20배 빠른 결과이다.    
또한 이전에는 4 agent의 환경에서 value function을 학습하는데 8시간에 걸린데에 비해, 이번 연구에서는 같은 시간동안 policy와 value function을 모두 학습했다.    
supervised learning을 통한 initialization이후에는 에피소드당 0.15의 reward를 얻었다. RL phase1 이후에는 에피소드당 0.9 reward를 얻을 수 있었다. RL phase2에 들어가서는 더 어려운 task로 인해서 초기에는 0.85로 떨어졌으나 마지막에는 0.93의 reward를 얻을 수 있었다. 

### B. Simulation Results
이전의 CADRL은 only 2-agent의 환경에서만 구동되었다. 그리고 SA-CADRL은 4-agent의 환경에서 구동되었다. 그리고 SA-CADRL의 경우 실제 환경에서도 잘 돌아가는 것을 보여주었다. 우리가 이번에 만든 GA3C-CADRL을 시뮬레이션 환경에서 SA-CADRL과 비교해보았다.
1) <img src="https://latex.codecogs.com/gif.latex?n&space;\leq&space;4" title="n \leq 4" /> agents.
500개의 랜덤한 시나리오에 대해서 실험해보았다. 적은 수의 agents의 결과에서는 SA-CADRL이 가장 짧은 시간에 골까지 도달하였다. 
![simulation](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-08-18-Motion_Planning_Among_Dynamic,_Decision_Making_Agents_with_Deep_Reinforcement_Learning/fig4.JPG?raw=true)
2) <img src="https://latex.codecogs.com/gif.latex?n>4" title="n>4" /> agents
SA-CADRL은 input으로 3개를 넘는 agents는 넣을 수 없다. 따라서 이 실험에서는 가장 가까운 3개의 agents를 인풋으로 넣어주었다. 확실히 많은 수의 agents에 대해서는 GA3C-CADRL이 좋은 성능을 보여주었다.
![test table](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-08-18-Motion_Planning_Among_Dynamic,_Decision_Making_Agents_with_Deep_Reinforcement_Learning/fig5.JPG?raw=true)
### C. Hardware Experiment
로봇에 2D Lidar, 3 Intel Realsense R200만을 통해 실제로 주행도 해보았다. 보행자의 위치와 속도는 2D Lidar의 scan데이터를 clustering하여 알아내었다. 보행자는 RGB이미지를 통해서 감지하였다. 다음 영상에서 보다시피 안전하게 주행하는 것을 확인할 수 있었다. 
youtube 영상

## Conclusion
- LSTM을 통해 agents의 수가 많아도 문제가 없이 주행이 가능함
- 기존의 연구들에 비해 agents의 수가 많아질 수록 성능이 확연히 차이가 남.
- Real world에서의 테스트를 통해 현실에서도 주행가능함을 증명.
