---
layout: post
title: (밑바닥부터 시작하는 딥러닝, 2장) 퍼셉트론
featured-img: 2021-01-09-ch2_perceptron/fig1
permalink: /book_review/ch2_perceptron/
category: book_review

---
## 퍼셉트론
퍼셉트론은 다수의 신호를 입력으로 받아 하나의 신호를 출력하는 것을 말한다. 입력신호인 x1과 x2를 넣으면 가중치인 w1과 w2에 각각 곱해진 후 더해진다. 만약 이 값이 특정 역치값(theta)보다 크면 1, 작으면 0을 return한다.

![perceptron](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-09-ch2_perceptron/fig1.jpg?raw=true){: width="300" height="300"}

![perceptron](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-09-ch2_perceptron/fig2.jpg?raw=true)



## 논리 회로
논리 회로는 AND, OR, NAND, XOR 게이트가 있다. 

### AND 게이트
입력이 모두 참일 때 출력도 참이다.
~~~
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
~~~
![perceptron](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-09-ch2_perceptron/fig3.jpg?raw=true){: width="300" height="300"}

### OR 게이트
입력이 하나라도 참이면 출력은 참이다.
~~~
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.3
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
~~~
![perceptron](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-09-ch2_perceptron/fig5.jpg?raw=true){: width="300" height="300"}

### NAND 게이트
입력이 하나라도 0이면 출력은 0이다.
~~~
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
~~~
![perceptron](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-09-ch2_perceptron/fig4.jpg?raw=true){: width="300" height="300"}

그리고 게이트를 나타내는 기호는 아래 그림과 같다.
![gate symbols](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-09-ch2_perceptron/fig6.jpg?raw=true)


### XOR게이트
입력이 같으면 0을 출력하고, 서로 다르면 1을 출력한다.

![perceptron](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-09-ch2_perceptron/fig7.jpg?raw=true){: width="300" height="300"}

XOR 게이트는 한 층의 퍼셉트론으로 만드는 것은 불가능하다. 따라서 OR, NAND, AND게이트를 섞어서 XOR게이트를 만들 수 있다. 아래 그림과 같이 조합하면 가능하다.

![perceptron](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2021-01-09-ch2_perceptron/fig8.jpg?raw=true)


#### 코드로 구현하기
~~~
def XOR(x1, x2):
    s1 = NAND(x1, x2)
		s2 = OR(x1, x2)
		y = ANd(s1, s2)
		return y
~~~
