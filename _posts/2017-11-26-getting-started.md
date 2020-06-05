---
layout: post
title: Learning to Fly by Clashing
---

## 목적
적은 비용으로 indoor 환경에서 드론 자율주행 학습

## 선행연구
localization 및 path_planning 에 대한 연구들
1. slam
..* 비싼 computational cost으로 real time에는 적합하지 않음
2. depth estimation method
..* 짧은 배터리와 적재 용량의 문제
3. stereo vision bsaed estimation
..* 하얀 벽과 같은 plain surface에서는 localization이 힘듬
4. monocular camera based methods use vanishing points as a guidance
..* 여전히 range sensor에 의존
5. 사람이 운행한 데이터를 통한 학습
..* 데이터를 수집하기 어렵고, 실패에 대한 데이터를 찾기가 힘듬.
6. 시뮬레이션을 통한 학습
..* 실제 환경과의 괴리가 있어 제대로 적용이 안됨.

RL을 통한 학습은 많은 데이터가 필요하기 때문에 supervised learning을 통해 학습을 사용함.

## 연구과정
1. 데이터 수집
