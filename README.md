# ExploringDongguk_deeplearning


✔ 동국탐방에 적용될 딥러닝 모델 (.tflite) 생성 및 성능 향상 진행<br>



### ⏳ 제작기간

------

2021.03.02 ~ 2021.06.16<br>




### 💫 사용모델

------

- InceptionV3


### 🥇 성능향샹

------

- 양질의 많은 **데이터 셋** 수집
- **Training** - 136536 images <br>
  **Testing** - 11157 images
  
    <p style="text-align: center;">
    <img src="https://user-images.githubusercontent.com/57933061/125451044-04712e0d-f62e-41db-80ae-ded27b01e7e4.png" width="400" float = "center">
  </p>

- Shearing, Horizontal flip, Zoom, Random Erasing **Augmentation** 수행

- **Confusion Matrix** 기반 정확도 개선
  <p style="text-align: center;">
    <img src="https://user-images.githubusercontent.com/57933061/125451792-e3b4aeea-1603-42f3-9b71-57c39384a1b1.png" width="550" float = "center">
  </p>


- **Fine tuning** 수행
  
   
🔴 위 과정 수행 후 정확도(Validation Accuracy) 향상 <br>
<p style="text-align: center;">
    <img src="https://user-images.githubusercontent.com/57933061/125456917-154ac86e-a883-4ec7-a6d7-bbfa2db2ab72.png" width="200" float = "center">
  </p><br>

  
### 🖍 추가구현

------

- Mask R-CNN 사용
- 건물 Segmentation 기술 개발 결과 (상록원, 정각원, 명진관 등)

![image](https://user-images.githubusercontent.com/57933061/125457678-b6aa9a38-af62-47b6-a011-4720af70093d.png)

![image](https://user-images.githubusercontent.com/57933061/125457802-74fd9d11-0c9d-483e-b9df-a6e5bc70b4ac.png)

![image](https://user-images.githubusercontent.com/57933061/125457879-535ed914-7b82-4f21-889e-abb9d2212916.png)

### 😺 GitHub Repository

------

https://github.com/minjuu/tflite_building.git<br>
https://github.com/minjuu/deeplearning_building.git





