# Nvidia AI 인공지능 응용(MD)

---

## 주제: YOLOv5를 활용한 RC카 고속 충돌 사고 예방

## Title: Preventing High-Speed Collision Accidents of RC Cars Using YOLOv5

---

## 프로젝트 개요(OverView of the Progect):

- **배경 정보 소개(Opening Background Information):**

자율주행 기술이 발전함에 따라 소형 이동 장치에서도 안전성과 사고 방지 기술의 중요성이 커지고 있다. 특히 고속 RC카는 어린이나 차량과 충돌 사고 예방을 위해 객체 탐지와 안전 제어 기술이 필요하며, 이는 더 큰 자율주행 차량의 안전성을 높이는 기반 기술로도 활용될 수 있다.

As autonomous driving technology advances, the importance of safety and accident prevention technology is increasing even for small mobile devices. In particular, high-speed RC cars require object detection and safety control technologies to prevent collision accidents with children or vehicles, and these can also be utilized as foundational technologies to enhance the safety of larger autonomous vehicles.

- **프로젝트의 전반적인 설명(General Description of the Current Project):**

이 프로젝트는 YOLOv5 객체 탐지 모델을 활용하여 RC카의 충돌 사고를 예방하는 시스템을 개발하는 것을 목표로 하고있다. RC카가 주행 중 장애물이나 사람을 인식하면 3초간 정지했다가 자동으로 재출발한다. RC카에 장착된 카메라가 실시간으로 전방을 모니터링하며, 장애물이 감지되면 즉각 정지 신호를 전달하여 사고를 방지한다.

This project aims to develop a system that prevents collision accidents of RC cars using the YOLOv5 object detection model. When the RC car recognizes obstacles or people while driving, it stops for 3 seconds and then automatically restarts. A camera mounted on the RC car monitors the front in real-time, and when an obstacle is detected, it immediately transmits a stop signal to prevent accidents.

- **제안하고 싶은 프로젝트의 강점(Proposed Idea for Enhancements to the Project):**
1. **정확한 사고 방지와 신속한 대응**: YOLOv5의 뛰어난 정확도와 실시간 처리 능력을 활용해 RC카가 고속 주행 중에도 장애물과 사람을 빠르게 감지하고 즉시 정지함으로써 충돌 사고를 예방한다.
2. **확장성과 활용성**: RC카의 자율주행 안전 기술은 소형 이동 장치뿐 아니라 대형 자율주행 차량에도 적용할 수 있다. 따라서 이 기술은 다양한 주행 환경에서 활용 가능한 기반 기술로 발전할 수 있다.

1. **Accurate accident prevention and rapid response**: By utilizing YOLOv5's excellent accuracy and real-time processing capabilities, the RC car can quickly detect obstacles and people even during high-speed driving and immediately stop, thereby preventing collision accidents.
2. **Scalability and applicability**: The autonomous driving safety technology for RC cars can be applied not only to small mobile devices but also to large autonomous vehicles. Therefore, this technology can develop into a foundational technology that can be utilized in various driving environments.

- **프로젝트의 중요성(Value and Signifiance of the Project):**

YOLOv5를 활용한 RC카의 실시간 환경 인식 시스템은 장애물과 차량을 감지하여 충돌을 방지한다. 이는 시야 제한으로 인한 사고를 예방하고, RC카의 안정성을 높이며, 더 안전한 주행 환경을 제공한다. 또한 이 기술은 향후 자율주행차에 적용 가능한 중요한 기반 기술로 발전할 잠재력이 크다.

The real-time environment recognition system for RC cars using YOLOv5 detects obstacles and vehicles to prevent collisions. This prevents accidents caused by limited visibility, increases the stability of RC cars, and provides a safer driving environment. Furthermore, this technology has great potential to develop into an important foundational technology that can be applied to future autonomous vehicles.

- **직면하고 있는 한계(Current Limitations):**

낮은 조도나 복잡한 환경에서는 정확도와 처리 속도가 저하될 수 있다. 또한, 3초 정지 후 재출발하는 메커니즘은 일반적인 장애물 인식에는 효과적이지만, 지속적으로 존재하는 장애물의 경우 원활한 주행이 어려울 수 있다.

Accuracy and processing speed may decrease in low-light conditions or complex environments. Additionally, while the mechanism of stopping for 3 seconds and then restarting is effective for recognizing general obstacles, it may be difficult to achieve smooth driving in the case of persistently present obstacles.

- **문헌 고찰(Literature Review):**

이 프로젝트에서 YOLOv5를 활용하기 위해 최신 연구와 기술 동향을 반영한 관련 논문들을 조사하고 있다. YOLOv5의 성능과 적용 가능성에 대한 문헌 검토를 통해 프로젝트의 최적화 방안과 실제 적용 가능성을 심도 있게 탐구하고 있다.

For this project, we are investigating relevant papers that reflect the latest research and technological trends in order to utilize YOLOv5. Through a literature review on the performance and applicability of YOLOv5, we are deeply exploring optimization methods and practical applicability of the project.

## 영상 취득 방법(Image Acquisition Method):

1. YouTube에서 차량과  RC카 충돌 영상을 다운로드하여 프로젝트에 필요한 영상을 확보했다.

We obtained the necessary footage for the project by downloading vehicle and RC car collision videos from YouTube.

1. YouTube에서 어린이들과 RC카, 그리고 킥보드의 충돌 영상을 다운로드하여 프로젝트에 필요한 영상을 확보했다.

We obtained the necessary footage for the project by downloading videos of collisions involving children, RC cars, and kick scooters from YouTube.

1. https://drive.google.com/file/d/1jkzntwb7617X_Rr9v1s3TySF3K6OnEnh/view?usp=drive_web
2. https://drive.google.com/file/d/1X90ox__9O0X9_HYIklbQ3UzbD3KTNQXB/view?usp=drive_web

## 학습 데이터 추출과 학습 어노테이션(Learning Data Extraction and Learning Annotation):

YOLOv5에서 640해상도 이미지로 학습하기 위해서 먼저 영상을 640 x 640 해상도 영상으로 만들었다.

To train using 640 resolution images in YOLOv5, we first converted the video into 640 x 640 resolution footage.

### 비디오 해상 조정(Video resolution adjustment)

[비디오 리사이저 - 온라인에서 무료로 비디오 해상도 변경](https://online-video-cutter.com/ko/resize-video#google_vignette)

![image.png](image.png)

640 x 640 해상도로 만들어진 영상을 프레임 단위로 이미지로 만들거나 어노테이션을 하기 위해서 Video/Image Labeling and Annotation Tool로 잘 알려진 DarkLabel을 사용했다.

To create frame-by-frame images or annotations from the video with 640 x 640 resolution, we used DarkLabel, a well-known Video/Image Labeling and Annotation Tool.

### DarkLabel.zip

[https://drive.google.com/file/d/1Aow0IWcd7MrlhPos4uQFZVOqvRQUA6n5/view?usp=drive_web](https://drive.google.com/file/d/1Aow0IWcd7MrlhPos4uQFZVOqvRQUA6n5/view?usp=drive_web)

![image.png](image%201.png)

DarkLabel 프로그램에서 영상을 프레임 단위로 이미지로 변환할 수 있다. 먼저 "Open Video"를 통해 640 × 640 해상도 영상을 선택한다. 이후 "Labeled frames only"가 체크 표시가 활성화되어 있다면 체크 표시를 비활성화한다. 이후 "as images"를 통해 "images", ”imag”라는 폴더 안에 이미지로 변환한다.

In the DarkLabel program, you can convert video into frame-by-frame images. First, select the 640 × 640 resolution video through "Open Video". Then, if "Labeled frames only" is checked, uncheck it. Afterwards, convert the video into images in a folder called "images" and “imag” using "as images".

![image.png](image%202.png)

images와 imag 폴더 안에 이미지가 들어온 걸 확인할 수 있다.

You can check that the image came into the images and imag folder.

![image.png](f95c8a65-c6b3-45fd-899d-379f6284fee1.png)

![image.png](image%203.png)

이제 변환된 이미지를 DarkLabel을 통해 Annotation을 한다.

Now the converted image is annotated through DarkLabel.

먼저 Annotation을 하기 전에 darklabel.yml을 통해 classes를 추가한다.

First, add classes through darklabel.yml before annotation.

yaml파일 안에 my_classes2,3을 만들고 class명은 car(자동차), RCcar(RC카), human(사람)/ bike(킥보드), RCcar(RC카), human(사람), house(집)을 추가한다.

Create my_classes2 and my_classes3 in the yaml file and add  car, RCcar, human/ bike, RCcar, human, house.

<aside>

my_classes2: ["car", "RCcar", "human"]
my_classes3: ["bike", "RCcar", "human", "house"]

</aside>

이제 Annotation할 때 DarkLabel GUI에서 설정한 classes를 볼 수 있게 classes_set은 미리 설정해 높은 my_classes2, my_classes3을 넣고 GUI에서 볼 name을 darknet yolo로 설정한다.

Now, in order to see the classes set in the Dark Label GUI during annotation, classes_set puts the pre-set my_classes2, my_classes3 and sets the ball name in the GUI to darknet yolo.

<aside>

format9:    # darknet yolo (predefined format]
fixed_filetype: 1                 # if specified as true, save setting isn't changeable in GUI
data_fmt: [classid, ncx, ncy, nw, nh]
gt_file_ext: "txt"                 # if not specified, default setting is used
gt_merged: 0                    # if not specified, default setting is used
delimiter: " "                     # if not spedified, default delimiter(',') is used
classes_set: "my_classes2"     # if not specified, default setting is used
name: "darknet yolo"           # if not specified, "[fmt%d] $data_fmt" is used as default format name

format10:    # darknet yolo (predefined format]
fixed_filetype: 1                 # if specified as true, save setting isn't changeable in GUI
data_fmt: [classid, ncx, ncy, nw, nh]
gt_file_ext: "txt"                 # if not specified, default setting is used
gt_merged: 0                    # if not specified, default setting is used
delimiter: " "                     # if not spedified, default delimiter(',') is used
classes_set: "my_classes3"     # if not specified, default setting is used
name: "darknet yolo"           # if not specified, "[fmt%d] $data_fmt" is used as default format name

</aside>

DarkLabel 프로그램에 darknet yolo라는 classes가 추가 되었고 밑에 3~4개의 class가 추가된 것을 확인할 수 있다.

It can be seen that classes called darknet yolo were added to the DarkLabel program and three and four classes were added at the bottom.

![KakaoTalk_20241115_083045691_01.jpg](06196564-d645-4bb7-b117-c7551fa71645.png)

![KakaoTalk_20241115_083045691.jpg](f78ab710-6c7e-4b74-b0d8-30c054256b02.png)

DarkLabel에서 Open Image Folder를 통해 images와 imag 폴더를 선택하여 변환된 이미지를 불러왔다. Box + Label로 선택 후 아래 사진과 같이 해당 class에 부합하는 car, RCcar, human에 Annotation을 했다. Annotation이 끝난 후 GT Save As를 통해 각 폴더 안에 저장을 했다.

The converted image was loaded by selecting the images and imag folders through Open Image Folder in DarkLabel. After selecting with Box + Label, annotations were made to the car, RCcar, and human that fit the class as shown in the photo below. After the annotation was completed, it was saved in each folder using GT Save As.

![image.png](image%204.png)

labels 안에 Annotation한 txt파일이 있음을 확인할 수 있다.

You can see that there is an annotated txt file in labels. 

![image.png](image%205.png)

## Nvidia Jetson Nano 학습 과정(**Nvidia Jetson Nano Training Process**):

Google drive에 Google Colaboratory을 설치한다.

Install Google Colaboratory on Google Drive.

![KakaoTalk_20241120_085113586.jpg](b9807b03-aed7-4ddb-abbc-d92d830cd4c5.png)

명령창에 prompt: 구글 드라이브랑 연결하는 코드를 입력한다.

Enter the code to connect to Google Drive in the command prompt:

<aside>

```
from google.colab import drive
drive.mount('/content/drive')
```

</aside>

### YLOLv5:

yolov5 google drive에 접속하여 다운로드하고 Requirements.txt파일로 필요한 라이브러리들을 일괄 설치한다.

Access yolov5 on Google Drive, download it, and install all the necessary libraries in bulk using the Requirements.txt file.

<aside>

```
# If you have already completed the installation, you only need to move to the corresponding path.
%cd /content/drive/MyDrive/yolov5

#clone YOLOv5 and
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
%pip install -qr requirements.txt # install dependencies

!pip install Pillow==10.3
```

</aside>

yolov5라는 폴더가 만들어지고 그 안에 Val파일이 있는데 DarkLabel에서 만든 images와 labels 안에 각각 사진과 txt파일을 넣는다. 그러고나서 data.yaml 파일을 classes에 맞게 파일을 수정한다.

A folder named yolov5 is created, and inside it there is a Val file. Place the photos and txt files into the images and labels folders created in DarkLabel, respectively. Then, modify the data.yaml file according to the classes.

![화면 캡처 2024-11-20 091258.png](%25ED%2599%2594%25EB%25A9%25B4_%25EC%25BA%25A1%25EC%25B2%2598_2024-11-20_091258.png)

### 이미지 파일 관리(**Image File Management**)

이미지 파일을 관리하기 위한 폴더를 생성한다.

We create folders to manage image files.

<aside>

```
!mkdir -p Train/labels
!mkdir -p Train/images
!mkdir -p Val/labels
!mkdir -p Val/images
```

</aside>

라벨 폴더와 이미지 폴더를 만든 후 yolov5 폴더를 선택하여 각각의 폴더에 이미지를 넣는다. 그 후 yolov5 폴더에 가이드와 함께 제공된 **yolov5n.pt** 파일과 라벨링한 class의 이름에 맞게 수정한 **data.yaml** 파일을 넣습니다.

After creating the label folder and image folder, select the yolov5 folder and place the images in each folder. Then, put the **yolov5n.pt** file provided with the guide and the **data.yaml** file modified to match the names of the labeled classes into the yolov5 folder.

<aside>

```
##데이터를 학습용 : 검증용 7:3 검증 데이터 만들기
import os
import shutil
from sklearn.model_selection import train_test_split

def create_validation_set(train_path, val_path, split_ratio=0.3):
  """
  Train 데이터의 일부를 Val로 이동
  """
  #필요한 디렉토리 생성
  os.makedirs(os.path.join(val_path, 'images'), exist_ok=True)
  os.makedirs(os.path.join(val_path, 'labels'), exist_ok=True)

  #Train 이미지 리스트 가져오기
  train_images = os.listdir(os.path.join(train_path, 'images'))
  train_images = [f for f in train_images if f.endswith(('.jpg','.jpeg','.png'))]

  #Train 이미지를 Train, Val로 분할
  _, val_images = train_test_split(train_images, test_size=split_ratio, random_state=42)

  #Val로 파일 복사
  for img in val_images:
    #이미지 복사
    src_image = os.path.join(train_path, 'images', img)
    dst_image = os.path.join(val_path, 'images', img)
    shutil.copy(src_image, dst_image)
    # 라벨 파일 복사
    label_file = os.path.splitext(img)[0] + '.txt'
    src_label = os.path.join(train_path, 'labels', label_file)
    dst_label = os.path.join(val_path, 'labels', label_file)
    if os.path.exists(src_label):
      shutil.copy(src_label, dst_label)

  print(f"Validation set created with {len(val_images)} images.")

#실행
train_path = '/content/drive/MyDrive/yolov5/Train'
val_path = '/content/drive/MyDrive/yolov5/Val'

create_validation_set(train_path, val_path)

def check_dataset():
  train_path = '/content/drive/MyDrive/yolov5/Train'
  val_path = '/content/drive/MyDrive/yolov5/Val'

  #Train 데이터셋 확인
  train_images = len(os.listdir(os.path.join(train_path, 'images')))
  train_labels = len(os.listdir(os.path.join(train_path, 'labels')))

  #Val 데이터셋 확인
  val_images = len(os.listdir(os.path.join(val_path, 'images')))
  val_labels = len(os.listdir(os.path.join(val_path, 'labels')))

  print("Dataset status:")
  print(f"Train - Images: {train_images}, {train_labels}")
  print(f"Val - Images: {val_images}, Labels: {val_labels}")

check_dataset()
```

</aside>

### **yolov5 모델 학습 시작(Start of YOLOv5 Model Training)**

필요 라이브러리 임포트한다.

Import the necessary libraries.

<aside>

```
import torch
import os
from IPython.display import Image, clear_output  # to display images
%pwd

import numpy as np
import tensorflow as tf
import os
from PIL import Image
from tensorflow.python.eager.context import eager_mode

def _preproc(image, output_height=512, output_width=512, resize_side=512):
    ''' imagenet-standard: aspect-preserving resize to 256px smaller-side, then central-crop to 224px'''
    with eager_mode():
        h, w = image.shape[0], image.shape[1]
        scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
        resized_image = tf.compat.v1.image.resize_bilinear(tf.expand_dims(image, 0), [int(h*scale), int(w*scale)])
        cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(resized_image, output_height, output_width)
        return tf.squeeze(cropped_image)

def Create_npy(imagespath, imgsize, ext) :
    images_list = [img_name for img_name in os.listdir(imagespath) if
                os.path.splitext(img_name)[1].lower() == '.'+ext.lower()]
    calib_dataset = np.zeros((len(images_list), imgsize, imgsize, 3), dtype=np.float32)

    for idx, img_name in enumerate(sorted(images_list)):
        img_path = os.path.join(imagespath, img_name)
        try:
            # 파일 크기가 정상적인지 확인
            if os.path.getsize(img_path) == 0:
                print(f"Error: {img_path} is empty.")
                continue

            img = Image.open(img_path)
            img = img.convert("RGB")  # RGBA 이미지 등 다른 형식이 있을 경우 강제로 RGB로 변환
            img_np = np.array(img)

            img_preproc = _preproc(img_np, imgsize, imgsize, imgsize)
            calib_dataset[idx,:,:,:] = img_preproc.numpy().astype(np.uint8)
            print(f"Processed image {img_path}")

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    np.save('calib_set.npy', calib_dataset)

# "cannot identify image file" 에러가 발생하는 경우, PILLOW Version을 "!pip install Pillow==10.1" 로 변경하여 설치합니다.
Create_npy('./Train/images', 512, 'jpg')

#모델 학습하기
!python train.py  --img 512 --batch 16 --epochs 300 --data /content/drive/MyDrive/yolov5/data.yaml --weights yolov5n.pt --cache
```

</aside>

- `--img 512`: 입력 이미지의 크기를 512x640으로 설정한다.
- `--batch 16`: 배치 크기를 설정한다. 한 번에 처리되는 이미지의 수를 나타낸다.
- `--epochs 300`: 학습할 총 에폭(epoch) 수를 설정합니다.
- `--data /content/drive/MyDrive/yolov5/data.yaml`: 데이터셋 및 모델 구성에 대한 설정이 담긴 YAML파일의 경로를 지정한다.
- `--weights yolov5n.pt`: 미리 훈련된 가중치 파일의 경로를 지정한다. 여기서는 yolov5n.pt파일을 사용하고 있다.

- `--img 512`: Sets the input image size to 512x640.
- `--batch 16`: Sets the batch size. This represents the number of images processed at once.
- `--epochs 300`: Sets the total number of epochs for training.
- `--data /content/drive/MyDrive/yolov5/data.yaml`: Specifies the path to the YAML file containing dataset and model configuration settings.
- `--weights yolov5n.pt`: Specifies the path to the pre-trained weights file. Here, the [yolov5n.pt](http://yolov5n.pt) file is being used.

아래 사진과 같이 여러 데이터들을 확인할 수 있다. YOLOv5의 훈련 중에 사용되는 손실(loss) 함수들이 있다. YOLOv5은 주로 세 가지의 손실 함수를 사용한다.

1. **Localization Loss(box):**
    - Localization 손실은 예측된 바운딩 박스의 좌표와 실제 객체의 바운딩 박스의 좌표 간의 차이를 측정한다. 이 손실은 예측된 바운딩 박스가 실제 객체에 더 가까워지도록 유도한다.
2. **Objectness Loss(obj):**
    - Objectness 손실은 각 그리드 셀이 객체를 포함해야 하는지 여부를 결정한다. 이 손실은 객체의 존재 여부와 무관하게 예측된 바운딩 박스의 신뢰도를 측정한다.
    - Objectness 손실은 실제 객체가 있는 그리드 셀에 대해서만 계산된다.
        
        즉, 해당 그리드 셀에 객체가 없는 경우에는 손실이 0이 된다.
        
3. **Classification Loss(cls):**
    - Classification 손실은 객체를 탐지한 경우에 대한 분류 손실을 계산한다. 이 손실은 객체가 포함된 그리드 셀에 대해서만 계산되며, 해당 객체의 클래스를 정확하게 분류하는 데 사용된다.

이러한 손실들은 YOLOv5의 훈련 과정에서 함께 최적화되며, 최종 손실은 이러한 세가지 손실들의 합으로 정의된다. 훈련 과정에서 이 손실들은 역전파(backpropagation)를 통해 모델의 가중치를 업데이트하는 데 사용된다. 이러한 결과들을 바탕으로 YOLOv5 모델은 객체 탐지 작업을 수행하는 데 더 나은 성능을 발휘할 수 있도록 학습된다.

As shown in the image below, various data can be observed. There are loss functions used during the training of YOLOv5. YOLOv5 primarily uses three loss functions.

1. **Localization Loss(box):**
    - Localization loss measures the difference between the coordinates of the predicted bounding box and the actual object's bounding box. This loss guides the predicted bounding box to be closer to the actual object.
2. **Objectness Loss(obj):**
    - Objectness loss determines whether each grid cell should contain an object. This loss measures the confidence of the predicted bounding box regardless of the presence of an object.
    - Objectness loss is calculated only for grid cells that actually contain objects.
        
        In other words, the loss becomes 0 if there is no object in that grid cell.
        
3. **Classification Loss(cls):**
    - Classification loss calculates the classification loss for detected objects. This loss is calculated only for grid cells containing objects and is used to accurately classify the object in question.

These losses are optimized together during the training process of YOLOv5, and the final loss is defined as the sum of these three losses. During training, these losses are used to update the model's weights through backpropagation. Based on these results, the YOLOv5 model is trained to perform better in object detection tasks.

![화면 캡처 2024-11-20 100104.png](%25ED%2599%2594%25EB%25A9%25B4_%25EC%25BA%25A1%25EC%25B2%2598_2024-11-20_100104.png)

학습이 끝나면 yolov5 폴더 안에 train에 exp이 학습된 걸 확인할 수 있다.

After the training is complete, you can see that exp has been trained in the train folder inside the yolov5 folder.

![화면 캡처 2024-11-20 100417.png](%25ED%2599%2594%25EB%25A9%25B4_%25EC%25BA%25A1%25EC%25B2%2598_2024-11-20_100417.png)

학습 결과는 다음과 같다.

The training results are as follows.

[exp](https://www.notion.so/exp-144100fb06dc808cadb1c4b227e15e21?pvs=21) 

**confusion_matrix**

![confusion_matrix.png](confusion_matrix.png)

**labels**

![labels.jpg](labels.jpg)

**F1_curve**

![F1_curve.png](F1_curve.png)

**P_curve**

![P_curve (1).png](P_curve_(1).png)

**PR_curve**

![PR_curve.png](PR_curve.png)

**R_curve**

![R_curve.png](R_curve.png)

**results**

![results (1).png](results_(1).png)

**train_batch**

![train_batch0.jpg](train_batch0.jpg)

**val_batch**

![val_batch0_labels.jpg](val_batch0_labels.jpg)

### **yolov5 모델 학습 결과 검증(Validation of YOLOv5 Model Training Results)**

학습 후 학습 결과를 확인하기 위한 작업이 필요하다. exp 파일 안에 weights 파일이 있다. 그 중 `best.pt`를 활용하여 `detect.py`를 진행해 학습 결과를 확인한다.

After training, a process is needed to verify the training results. There is a weights file inside the exp file. Among them, we use `best.pt` to run `detect.py` to check the training results.

```python
# Start tensorboard
# Launch after you have started training
# logs save in the folder "runs"
%load_ext tensorboard
%tensorboard --logdir runs

!python detect.py --weights runs/train/exp/weights/best.pt --img 512 --conf 0.1 --source /content/drive/MyDrive/yolov5/Train/images

#display inference on ALL test images

import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/drive/MyDrive/yolov5/runs/detect/exp/*.jpg')[:10]: #이미지 파일 형식에 맞춰 .png 또는 .jpg 등으로 수정
    display(Image(filename=imageName))
    print("\n")
```

- `python detect.py`: Python 스크립트를 실행한다.
- `--weights runs/train/exp/weights/best.pt`: 가중치 파일의 경로를 지정한다. 이 가중치는 모델 학습을 통해 얻은 것으로, 가장 잘된 것만 모은 차선 감지 파일이다.
- `--img 512`: 입력 이미지의 크기를 512x640으로 설정한다.
- `--conf 0.1`: 객체 탐지의 신뢰도 임계값을 0.1로 설정한다. 이는 모델이 객체를 탐지했다고 판단하는 최소 확률을 의미한다. 0.1 이상의 확률로 탐지된 객체만 결과에 포함된다.
- `--source /content/drive/MyDrive/yolov5/Train/images`: 입력 이미지 또는 비디오 파일의 경로를 지정한다.
- `for imageName in glob.glob('/content/drive/MyDrive/yolov5/runs/detect/exp/*.jpg')[:10]:` :이미지 파일 형식에 맞춰 .png 또는 .jpg 등으로 수정

- `python detect.py`: Executes the Python script.
- `--weights runs/train/exp2/weights/best.pt`: Specifies the path to the weights file. These weights are obtained through model training and represent the best lane detection file compiled.
- `--img 512`: Sets the input image size to 512x640.
- `--conf 0.1`: Sets the confidence threshold for object detection to 0.1. This means the minimum probability at which the model considers an object to be detected. Only objects detected with a probability of 0.1 or higher are included in the results.
- `--source /content/drive/MyDrive/yolov5/Train/images`: Specifies the path to the input image or video file.
- `for imageName in glob.glob('/content/drive/MyDrive/yolov5/runs/detect/exp2/*.jpg')[:10]:` : Modify to .png or .jpg etc. according to the image file format

**detect.py 실행 이미지 (detect.py execution image):**

![image.png](image%206.png)

**detect.py를 통한 학습 결과 (Learning outcomes with detect.py):**

![화면 캡처 2024-11-21 023726.png](%25ED%2599%2594%25EB%25A9%25B4_%25EC%25BA%25A1%25EC%25B2%2598_2024-11-21_023726.png)

![화면 캡처 2024-11-21 023755.png](%25ED%2599%2594%25EB%25A9%25B4_%25EC%25BA%25A1%25EC%25B2%2598_2024-11-21_023755.png)

![화면 캡처 2024-11-21 023901.png](%25ED%2599%2594%25EB%25A9%25B4_%25EC%25BA%25A1%25EC%25B2%2598_2024-11-21_023901.png)

![화면 캡처 2024-11-21 023928.png](%25ED%2599%2594%25EB%25A9%25B4_%25EC%25BA%25A1%25EC%25B2%2598_2024-11-21_023928.png)

**detect.py 실행 영상**

```python
!python detect.py --weights /content/drive/MyDrive/yolov5/runs/train/exp/weights/best.pt --img 512 --conf 0.1 --source /content/drive/MyDrive/video.mp4
```

[yolov51]

[https://drive.google.com/file/d/1hL262ZSgrq0dQfRf7SpUJw4I5UnzONe_/view?usp=drive_link](https://drive.google.com/file/d/1hL262ZSgrq0dQfRf7SpUJw4I5UnzONe_/view?usp=drive_link)

[yolov5]

[https://drive.google.com/file/d/1hNufb75hTbkJbWdUpjuQcTs4eE4T-Sze/view?usp=drive_link](https://drive.google.com/file/d/1hNufb75hTbkJbWdUpjuQcTs4eE4T-Sze/view?usp=drive_link)