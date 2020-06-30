---
layout: post
title: Road Surface Monitoring Using Smartphone Sensors:A_Review
featured-img: 2020-06-30-Road_Surface_Monitoring_Using_Smartphone_Sensors_A_Review/figure1
---

### 이글은 [Road Surface Monitoring Using Smartphone Sensors: A Review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6263868/pdf/sensors-18-03845.pdf)를 정리한 포스트이다.

도로 표면을 모니터링하는 것은 부드럽고, 안전한 도로 시설을 위해 중요하다. 
도로 표면을 모니터링하는 것의 핵심은 표면의 특이점(움푹패인 곳, 갈라짐, 울퉁불퉁함)을 판정하는 것이다. 최근에는 스마트폰을 통한 센싱으로 도로 상태를 모니터링 하는 것이 트렌드이다. 

## introduction
캐나다의 ARAN(Automated Road Analyzer), 뉴질랜드의 ROMDAS(Road Measurement Data Acquisition System) 등이 실제로 운영되고 있다. 이 둘은 모두 레이저, 초음파 센서, 카메라를 사용한다. 그러나 이들은 노동 집약적이고, 비싸고, 모든 도시를 커버하기에는 데이터의 양이 부족하다. 이런 문제로 저렴하고, 고효율의 방식이 필요하다. 그리고 real time으로 감시할 수 있는 방식이 필요하다. 도로 표면 감지는 보통 3가지 방법이 있다. 3D reconstruction, vibration, vision-based의 방법이다.

3D reconstruction방법은 3D laser scanning으로 surface modeling을 진행한다. 그리고 생성된 포인트 클라우드로 특이점을 검출한다. 그러나 이 방법은 비싼 laser scanner가 필요하고, 비용이 많이 드는 방법이다.

vision-based 방법은 texture extraction과 같은 이미지 처리를 사용한다. geotagged(위치 정보가 표시된) 이미지를 이용한다. 3D reconstruction 방법보다는 저렴하지만, environmental conditions(빛, 그림자)등에 영향을 많이 받는다. 

vibration-based 방법은 움직이는 vehicle에 장착된 motion 센서(accelerometers or gyroscopes)를 통해 얻은 데이터로 도로 표면에서 특이점을 검출한다. motion 데이터를 측정하고, 분석하는것은 센서의 특성, 질, 스마트폰의 위치, vehicle의 suspension 시스템, 속도 등 많은 요인에 영향을 받는다. 


## main
5개의 단계로 이루어진다.
1. Sensing (data collection)   
스마트폰의 센서로 부터 데이터를 가져온다. 스마트폰의 센서는 motion 센서, position 센서가 있다.
   * **motion 센서** : accelerometer, gyroscope, linear accelerometer, rotation
   * **position 센서** : GPS, manometer, rotation
2. Preprocessing   
전처리는 2가지의 목적을 가지고 있다.
   * 센서의 noise 처리
   * 센서의 좌표계를 geographic 좌표계로 변환
3. Processing for Feature Extraction   
데이터를 통해 원하는 정보를 추출
4. Post-Processing   
처리된 센서 데이터를 서버에 전송하고, 여러 sources를 통해 얻은 정보를 integration
5. Performanc Evaluation   
평가

![figure1.jpg](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-06-30-Road_Surface_Monitoring_Using_Smartphone_Sensors_A_Review/figure1.jpg?raw=true)

위의 과정에 대해 더 자세히 살펴보자.
1. **Sensor Data Collection**   
스마트폰의 센서는 크게 hardware-based(physical)센서, software-based(virtual)센서로 나눠진다. 
   * **physical 센서**는 raw 데이터 그 자체를 받아오는 센서이다. accelerometers, gyroscopes, magnetometers, light, tempurature 등의 정보를 수집한다.
   *  **software-based 센서**는 physical 센서로부터 받아오는 raw 데이터를 계산하여 새로운 정보를 내보내는 센서이다. linear acceleration, rotation, gravity 등의 정보를 얻는다.
![sensor](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-06-30-Road_Surface_Monitoring_Using_Smartphone_Sensors_A_Review/figure2.jpg?raw=true)

대부분의 연구들은 accelerometer 센서를 통해서만 이루어졌다. 하지만 Yagi[[21](http://www.bumprecorder.com/wp-content/uploads/2013/12/225c966eb8450f15af993862b032ba6e.pdf)], Douangphachanh and Oneyama[[22](https://ieeexplore.ieee.org/document/7049855)], Mohamed[[30](https://www.researchgate.net/publication/266387427_RoadMonitor_An_Intelligent_Road_Surface_Condition_Monitoring_System)]의 연구는 accelerometer와 gyroscope를 결합하여 더 좋은 성능을 이끌어 냈다. 

Data sampling rate는 특이점을검출하는 과정에서 중요한 역할을 한다. 적절한 sampling rate를 찾는 것은 여러 요인(available resources, 요구되는 정확도, type of data)에 의해 결정된다[[31](http://dx.doi.org/10.3390/s150102059)].    
Douangphachanh and Oneyama[[32](https://ieeexplore.ieee.org/document/6685585)]의 연구에 따르면 도로의 특이점은 40-50Hz에서 검출이 가장 잘 되었다. 높은 sampling rate는 검출할 chances를 늘려주지만 배터리의 사용량, 저장소의 용량, data의 처리에 악영향을 준다. 적절한 sampling rate를 찾는 것은 speed of movement에도 크게 달라진다.   
Singaray[[19](https://www.researchgate.net/publication/236255718_Low_Computational_Approach_for_Road_Condition_Monitoring_Using_Smartphones)]는 낮은 sampling rate로 도로 표면의 특이점을 검출하는 방법을 개발했다.   
![research](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-06-30-Road_Surface_Monitoring_Using_Smartphone_Sensors_A_Review/figure3.jpg?raw=true)

2. **Preprocessing**   
전처리의 첫 번째 목적은 sensor data를 smooth하게 만들고, noise를 제거하는 것이다. 이를 위해서 filtering을 거친다. filtering 방법에는 3가지 방법이 있다.
   * moving-average filtering : 가장 흔하고, 간단하게 sensor의 prior 정보가 없어도 사용할 수 있는 방법이다.  random noise를 줄일 때 사용한다.
   * low-/high-pass filtering : 미리 정해진 frequencies 정해 자르는 방식으로 noise를 제거한다.
   * band-pass filtering : 센서 데이터에서 정해진 range내의 frequencies만 통과시키고, 나머지는 제거한다.

   전처리의 두 번째  목적은 센서 데이터의 좌표계를 geographic 좌표계로 변환하는 것이다.

   Sebestyen[[33](https://ieeexplore.ieee.org/document/7145123)] 그리고 Seraj[[34](https://ris.utwente.nl/ws/portalfiles/portal/5493041/Seraj_et_al._-_RoADS.pdf)]는 2가지 다른 filter를 사용했다. 하나는 noise를 지우기 위한 filter, 하나는 도로의 특이점에 의해 발생하는 acceleration의 variation을 증폭시켜주는 filter이다.   
   Douangphachanh and Oneyama[[22](https://www.scitepress.org/Papers/2014/51174/51174.pdf)]는 vehicle의 속도, 주행(회전)에 따라 발생하는 low-frequency 정보를 detect하기 위해 high-pass filter를 사용했다.   
   Mohamed[[30](https://www.researchgate.net/publication/266387427_RoadMonitor_An_Intelligent_Road_Surface_Condition_Monitoring_System)]는 이차 high-pass butterworth filter[[35](https://www.changpuak.ch/electronics/downloads/On_the_Theory_of_Filter_Amplifiers.pdf)]를 사용했다.   
   Singh[[39](http://dx.doi.org/10.1016/j.pmcj.2017.06.002)]는 data를 smooth하게 하기 위해 simple moving-average와 band-passs filter를 사용했다.   
   Mohan et al. [[37](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Nericell-Sensys2008.pdf)], Bhoraskar et al. [[27](https://ieeexplore.ieee.org/document/6151382)], Vittorio et al. [[38](http://dx.doi.org/10.1016/j.sbspro.2014.01.057)], Sebestyen et al. [[33](https://ieeexplore.ieee.org/document/7145123)], Wang et al. [[9](http://dx.doi.org/10.1155/2015/869627)], and Singh et al. [[39](http://dx.doi.org/10.1016/j.pmcj.2017.06.002)] 의 논문에서는 센서의 좌표를 geographic 좌표로 바꾸기 위한 방법이 제시되었다.    
   Silva et al.[[40](http://dx.doi.org/10.1016/j.procs.2017.11.056)]에서는 null sensor data를 없애고, 일정하지 않은 timestamp data 값을 처리하는 방법이 나와있다.   
   
3. **Processing**   
전처리된 센서 값을 통해 도로 표면의 특이점을 검출하는 과정이다. 맨홀, road joints와 같은 사람이 만든 시설과 실제로 pothole, bump 등을 구분하는 과정도 포함된다. 
처리되는 방식에 따라 Online과 Offline으로 나뉜다.    Offline 방식은 sensor data를 수집한 후 일괄적으로 처리하고, Online 방식은 수집, 전처리, Processing의 과정이 동시에 일어난다.

   Processing에는 보통 3가지 방법이 사용된다.
   1. **Threshold-based**   
증폭된 accelerometer signal을 정해진 threshold 이상이 되면 특이점이라고 판단한다. Thershold-based 방법도 3가지 관점에 따라 나누었다. 
      1. length of interval for a window function([window function 참고](https://en.wikipedia.org/wiki/Window_function))   
         스펙트럼 분석에서 window function의 interval 길이를 결정하는 것은 여러 요인(vehicle의 speed, 앞 뒤 바퀴 사이의 거리 등)에 의해 변하므로, 상당히 challenging하다. window function에서는 미리 정해진 interval의 길이를 통해 분석한다. 아래의 표는 각 논문에서 사용된 window function의 interval의 길이이다. 
         ![compare](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-06-30-Road_Surface_Monitoring_Using_Smartphone_Sensors_A_Review/figure4.jpg?raw=true)
         
      2. fixed vs flexible threshold determination   
      threshold 값을 결정하는 것도 여러 요인(vehicle의 suspension 시스템, sensor의 특성, 센서의 위치)에 의해 결정된다.          
         Mohan et al. [[37](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Nericell-Sensys2008.pdf)], Mednis et al. [[41](https://www.researchgate.net/publication/224253154_Real_Time_Pothole_Detection_Using_Android_Smartphones_with_Accelerometers)], Sinharay et al. [[19](https://www.researchgate.net/publication/236255718_Low_Computational_Approach_for_Road_Condition_Monitoring_Using_Smartphones)], and Yi et al. [[43](http://dx.doi.org/10.1109/TITS.2014.2378511)]에서는 경험을 통해 fixed threshold를 결정한다.   
         Sebestyen et al. [[33](https://ieeexplore.ieee.org/document/7145123)], Wang et al. [[9](http://dx.doi.org/10.1155/2015/869627)], and Harikrishnan and Gopi [[36](http://dx.doi.org/10.1109/JSEN.2017.2719865)]에서는 일정하지 않은 signal pattern을 인지하기 위해 dynamic threshold를 사용한다.   
         dynamic threshold가 여러 환경에 적용될 수 있기 때문에 더 매력적이다.
         
      3. amplitude of signal vs other properties of signal amplitude   
      Mohan et al. [[37](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Nericell-Sensys2008.pdf)], Mednis et al. [[41](https://www.researchgate.net/publication/224253154_Real_Time_Pothole_Detection_Using_Android_Smartphones_with_Accelerometers)], Sebestyen et al. [[33](https://ieeexplore.ieee.org/document/7145123)], Wang et al. [[9](http://dx.doi.org/10.1155/2015/869627)], and Harikrishnan and Gopi [[36](http://dx.doi.org/10.1109/JSEN.2017.2719865)]에서는 증폭된 signal 값을 통해 threshold를 결정한다.   
      Yagi [[21](http://www.bumprecorder.com/wp-content/uploads/2013/12/225c966eb8450f15af993862b032ba6e.pdf)], Nomura and Shiraishi [[42](http://www.infsoc.org/journal/vol07/IJIS_07_1_029-036.pdf)], Vittorio et al. [[38](http://dx.doi.org/10.1016/j.sbspro.2014.01.057)], and Yi et al. [[43](http://dx.doi.org/10.1109/TITS.2014.2378511)]에서는 통계적인 값(표준편차 등)으로 threshold를 결정한다.   
      Mednis et al.[[41](https://www.researchgate.net/publication/224253154_Real_Time_Pothole_Detection_Using_Android_Smartphones_with_Accelerometers)]은 accelerometer 데이터로 부터 도로 표면의 특이점을 검출할 때 표준편차가 가장 중요한 파라미터라고 규정했다.

   2. **Machine Learning**   
   도로 표면 특이점 검출을 위해 unsupervised 또는 supervised를 사용한다.
   Bhoraskar et al.[[27](https://ieeexplore.ieee.org/document/6151382)]은 unsupervised learning 중 k-means clustering을 통해 도로의 smooth, bump를 구분하였고, SVM 알고리즘을 학습시켰다. 여기서 데이터는 하나하나 label되었다.   
   Mohamed et al.[[30](https://www.researchgate.net/publication/266387427_RoadMonitor_An_Intelligent_Road_Surface_Condition_Monitoring_System)]에서는 threshold-based 방법을 통해 labeling하고, 이것을 통해 smooth, bump를 구분하는 SVM을 학습시켰다.    
   Perttunen et al. [[28](https://link.springer.com/chapter/10.1007/978-3-642-23641-9_8)], Jain et al. [[29](https://www.usenix.org/conference/nsdr12/speed-breaker-early-warning-system)], Seraj et al. [[34](https://ris.utwente.nl/ws/portalfiles/portal/5493041/Seraj_et_al._-_RoADS.pdf)], and Mohamed et al. [[30](https://www.researchgate.net/publication/266387427_RoadMonitor_An_Intelligent_Road_Surface_Condition_Monitoring_System)]은 sensor data를 구분하기 위해 SVM을 사용하였다. 캡처된 비디오, 이미지, 소리, 현장 검증 등을 통해서는 ground truth를 얻고, 특이점을 labeling하는데 사용했다.   
   Silva et al.[[40](http://dx.doi.org/10.1016/j.procs.2017.11.056)]에서는 여러 supervised learning의 방법을 서로 비교하였다. Gradient boosting (GB), decision tree (DT), multilayer perceptron classifier (MPL), Gaussian Naive Bayes (GNB), and linear SVC 각각 을 short bump, long bump, unleveled manholes, and others를 구분하도록 학습시키고, 비교하였다.    
   여러 방법을 통해 sensor data가 성공적으로 분류되었으나 supervised learning을 위해서 labeled samples가 필요하고, real-time으로 적절하지 않다는 단점이 있다.   

   3. **Dynamic Time Warping(DTW)**    
   speech recognition의 방법에서 영감을 얻은 방법이다. 시계열 분석인데, 들어오는 signal data를 미리 정해진 templates과 비교하여, 유사도를 통해 분석한다.   
   Singh et al.[[39](http://dx.doi.org/10.1016/j.pmcj.2017.06.002)]에서는 DTW 방법으로 accelerometer data를 활용해 도로 표면의 특이점을 검출하였다. pothole(움푹패임)과 bump(울퉁불퉁함)에서 accelerometer data를 시계열로 수집하고, 이것을 server에 template으로 저장한다. 이후 새로 들어온 데이터와 비교하여 유사도를 검사하고, 이를 통해 특이점을 검출한다. 이 방법의 성능은 reference template의 quality에 연관되었다. 이 방법은 computationall intensive and unreliable하다. 또한 다른 환경(vehicle의 종류, 도로의 상태, vehicle의 speed)마다 templates이 필요하다는 단점이 있다.    

   **도로 표면 특이점의 분류**   
   도로의 특이점은 2가지로 나눌 수 있다.   
      1. 실제로 표면이 움푹패인 곳이나 bump(울퉁불퉁함)   
      2. Man-made 도로의 특이점, 맨홀, road joints, catchment basin(도로의 물이 모이는곳), 과속 방지턱 등   

   두 경우를 구분해야하지만 비슷한 패턴을 보이기 때문에 challenging 하다.    
   Sebestyen et al.[[33](https://ieeexplore.ieee.org/document/7145123)]에 따르면 과속 방지턱은 움푹패인 곳과 구분할 수 있다. 차가 움푹패인곳을 지나면 처음에는 내려가고 이후 올라간다. 사람이 만든 bump에서는 처음에 올라가고, 이후 내려간다. 이런 패턴을 기준으로 구분할 수 있다.       
   Harikrishnan and Gopi[[36](http://dx.doi.org/10.1109/JSEN.2017.2719865)]는 X-Z filter를 통해 potholes 와 과속방지턱을 구분한다.   
   Eriksson et al.[[46](https://www.cs.uic.edu/~jakob/papers/p2-mobisys08.pdf)]에서는 pothole은 accelerometer sensor data의 x-direction의 variation이 커지도록 영향을 주지만 과속 방지턱은 x, y 두 방향 모두에 영향을 주고, x-direction의 variation을 작게 만든다고 제안했다.   
   
   **Speed Dependency**   
   도로의 특이점 검출은 vehicle의 속도에도 큰 영향을 받는다.   
   Douangphachanh and Oneyama [[32](https://ieeexplore.ieee.org/document/6685585)]는 평균 속도가 road roughness estimation에 중요하다고 말한다. pothole과 같은 도로의 특이점을 지날 때 다른 속도로 가면 sensor data도 다른 모양을 보인다. 속도에 따른 센서 데이터의 모델링은 [[47](http://dx.doi.org/10.3390/s17020305)]에서 확인할 수 있다.   
   Fox et al.[[20](https://ieeexplore.ieee.org/document/7338353)]의 연구에서는 빠른 속도로는 도로 특이점 검출이 힘들다는 사실을 밝혔다.   
   Sebestyen et al.[[33](https://ieeexplore.ieee.org/document/7145123)]에서는 서로 다른 속도에서(15,30,60km/h) 데이터를 수집했다.다른 속도에서 수집된 데이터들은 30km/h에 맞게 normalize 되었다.   
   Perttunen et al.[[19](https://www.researchgate.net/publication/236255718_Low_Computational_Approach_for_Road_Condition_Monitoring_Using_Smartphones)]에서는 데이터에서 속도의 영향을 없애도록 설계되었다.    
   Mednis et al.[[28](https://link.springer.com/chapter/10.1007/978-3-642-23641-9_8)]에서는 속도에 따라 다른 알고리즘을 적용하였다. 예로 25km/h이하에서는 z-sus 알고리즘을 적용하고, 25km/h 이상에서는 z-peak 알고리즘을 적용했다.   
   Harikrishnan and Gopi[[36](http://dx.doi.org/10.1109/JSEN.2017.2719865)]는 false positive를 최소화하기 위해 velocity-dependent 변수를 제시했다. 이 변수는 속도에 따라 threshold value에 반영된다.   
   여러 연구들이 속도에 대한 방법을 제시했으나 아직 충분히 robust하게 속도를 보완할 수 있는 방법이 없다.   
 
4. **Post-Processing**   
후처리는 crowdsourcing 과 여러 sources로 부터 들어온 데이터를 integrating 하는 과정이다. 여러 정보를 수집하고, 합치는 과정에서 정확성과 신뢰성을 높일 수 있다.   
Chen et al.[[49](https://link.springer.com/article/10.1007/s11276-015-0996-y#:~:text=We%20propose%20a%20crowdsourcing%2Dbased,monitoring%20system%2C%20simply%20called%20CRSM.&text=The%20results%20show%20that%20CRSM,from%20small%20bumps%20or%20potholes.)] and Fox et al.[[20](https://ieeexplore.ieee.org/document/7338353)]에서는 pathole에 대한 정보를 클라우드로 전송한 후 voting algorithm을 통해 진짜 pathole을 검출한다. 여러 source로 부터 온 pathole의 갯수가 threshold를 넘어가면 그 부분에는 pathole이 있다고 판단한다.    
Yi et al.[[43](http://dx.doi.org/10.1109/TITS.2014.2378511)]은 grid-based clustering algorithm called DENCLUE(DENsity CLUstering)은 5m내의 grid zone에서 reporting frequency를 확인해서 false detection을 거른다.    

5. **Performance Evaluations**   
평가를 위해 accuracy ratio, precision, false positives ratio, and false negatives ration 등이 쓰인다. 각 application의 용도에 따라서 평가해야하는 지표는 달라진다. 다음 표는 여러 연구들의 성능이다.    
![performance](https://github.com/SUNGBEOMCHOI/SungBeomChoi.github.io/blob/master/assets/img/posts/2020-06-30-Road_Surface_Monitoring_Using_Smartphone_Sensors_A_Review/figure5.jpg?raw=true)

## Discussion   
processing의 방법을 비교해보자.
   1. Threshold-based approach   
      - 사람의 습관, vehicle의 suspension system, sensor의 특성에 따라서 바뀌는 부분을 최소화 하기 위해 쓰였고, 가장 많이 연구되고 있다.
      - 그러나 robust하거나 inclusive한 환경을 커버하지 못했다.
   2. Machine-learning approach   
      - 연구가 많이 안된 방법이다.
      - Threshold-based 방법에서 나온 단점을 보완할 수 있다.
      - 그러나 이것도 inclusive하거나 robust한 솔루션은 되지 못한다.
      - SVM같은 알고리즘을 위해서는 많은 데이터가 필요하다.

   위의 두 방법을 섞은 것은 서로의 단점을 보완해줄 수 있다.
preprocessing이 상당히 복잡해서 이것보다는 여러 소스를 통한 post processing이 성능을 많이 높일 수 있다.

또한 data transferring에서 RESTful(representational state transfer)같은 기술이나 JSON같은 데이터 포맷을 사용하는 것이 packet rate을 주이기 위해 필요하다.

## Conclusion   
**Challenging tasks**   
- 성능 지표에 대한 비교가 어렵다.
- 많은 threshold-based approach에도 일관된 manner가 없다.
- ML 모델을 위해서는 많은 데이터가 필요하다.
