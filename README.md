# 동국탐방

동국대학교 건물 인식 모델 구현 및 정확도 개선


### 데이터셋
------

1. 18개의 건물을 다양한 각도, 환경에서 동영상 촬영 
2. 수집한 동영상을 프레임 단위로 잘라 Train 데이터셋 생성 (224*224)
3. 같은 방식으로 Test 데이터셋 생성 (224*224)

Train: 136536 images<br>
Test: 11157 images<br>

### 딥러닝 모델
------

- Keras + Tensorflow
- InceptionV3


### 정확도 향상

------

- **Data Augmentation**
  <p style="text-align: center;">
    <img src="https://user-images.githubusercontent.com/72545216/141383561-a9abe40c-489a-4d75-a9ff-4a98b937c8ad.PNG" width="550" float = "center">
  </p>

- **Fine tuning** 수행

  
### Segmentation 기술
------

- Pytorch + Detectron2
- Mask R-CNN
- 데이터 셋<br>
  Train: 2000 images<br>
  Test: 200 images<br>
- 결과<br>
![명진관](https://user-images.githubusercontent.com/72545216/141384092-27b40b65-cc0c-4e81-896a-ecd8e1b3ee03.png)
![사회과학관](https://user-images.githubusercontent.com/72545216/141384135-6f465b86-fee4-4315-b33e-247674de4d1f.png)
![정각원](https://user-images.githubusercontent.com/72545216/141384140-468ee1bc-89bf-499e-a094-08ba7aedd3cb.png)
![정보문화관](https://user-images.githubusercontent.com/72545216/141384149-2254e675-6f75-45c8-9056-6ee3b5d6d765.png)
![중앙도서관](https://user-images.githubusercontent.com/72545216/141384151-1d7e13c3-d22e-4189-86cd-a0b8cf1e75bd.png)


### 결과
------

- 초기 78% -> 최종 98%<br>
![결과](https://user-images.githubusercontent.com/72545216/141384308-2ffa562f-90dd-44df-bab4-c3b49bd717cd.png)

- Confusion Matrix<br>
![confusion](https://user-images.githubusercontent.com/72545216/141384337-eac3d0e2-7fd4-42fe-9d1a-64e92b9d2c2e.png)<br>

- 앱 실행 화면(낮 밤 임의 3개씩)<br>

![그림1](https://user-images.githubusercontent.com/72545216/141385085-cca0fdc8-d9e1-4aba-9d2e-5899f86ad9e5.gif)
![그림2](https://user-images.githubusercontent.com/72545216/141384889-e34bdfea-dec3-4c81-bee1-8a7c54ce08b6.gif)
![그림3](https://user-images.githubusercontent.com/72545216/141384895-80f81271-2f0d-4c99-af63-048104df5d60.gif)<br>
![그림4](https://user-images.githubusercontent.com/72545216/141384903-d6a1b56e-1604-43d7-8609-1807f7af1906.gif)
![그림5](https://user-images.githubusercontent.com/72545216/141384911-6a7fc717-d268-41b9-91f9-042e2af9ab46.gif)
![그림6](https://user-images.githubusercontent.com/72545216/141384924-7637bdbf-4377-43e9-be6e-908c63c8a569.gif)

