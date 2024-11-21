# Nvidia AI YOLOv5(MD)

---

## Title: Preventing High-Speed Collision Accidents of RC Cars Using YOLOv5

---

[ssu57/YOLOv5_repot Issues](ssu57%20YOLOv5_repot%20Issues%20145100fb06dc800a998ec02225a1fb9e.csv)

## OverView of the Progect:

- **Opening Background Information:**

As autonomous driving technology advances, the importance of safety and accident prevention technology is increasing even for small mobile devices. In particular, high-speed RC cars require object detection and safety control technologies to prevent collision accidents with children or vehicles, and these can also be utilized as foundational technologies to enhance the safety of larger autonomous vehicles.

- **General Description of the Current Project:**

This project aims to develop a system that prevents collision accidents of RC cars using the YOLOv5 object detection model. When the RC car recognizes obstacles or people while driving, it stops for 3 seconds and then automatically restarts. A camera mounted on the RC car monitors the front in real-time, and when an obstacle is detected, it immediately transmits a stop signal to prevent accidents.

- **Proposed Idea for Enhancements to the Project:**
1. **Accurate accident prevention and rapid response**: By utilizing YOLOv5's excellent accuracy and real-time processing capabilities, the RC car can quickly detect obstacles and people even during high-speed driving and immediately stop, thereby preventing collision accidents.
2. **Scalability and applicability**: The autonomous driving safety technology for RC cars can be applied not only to small mobile devices but also to large autonomous vehicles. Therefore, this technology can develop into a foundational technology that can be utilized in various driving environments.

- **Value and Signifiance of the Project:**

The real-time environment recognition system for RC cars using YOLOv5 detects obstacles and vehicles to prevent collisions. This prevents accidents caused by limited visibility, increases the stability of RC cars, and provides a safer driving environment. Furthermore, this technology has great potential to develop into an important foundational technology that can be applied to future autonomous vehicles.

- **Current Limitations:**

Accuracy and processing speed may decrease in low-light conditions or complex environments. Additionally, while the mechanism of stopping for 3 seconds and then restarting is effective for recognizing general obstacles, it may be difficult to achieve smooth driving in the case of persistently present obstacles.

- **Literature Review:**

For this project, we are investigating relevant papers that reflect the latest research and technological trends in order to utilize YOLOv5. Through a literature review on the performance and applicability of YOLOv5, we are deeply exploring optimization methods and practical applicability of the project.

## Image Acquisition Method:

1. We obtained the necessary footage for the project by downloading vehicle and RC car collision videos from YouTube.
2. We obtained the necessary footage for the project by downloading videos of collisions involving children, RC cars, and kick scooters from YouTube.
3. https://drive.google.com/file/d/1jkzntwb7617X_Rr9v1s3TySF3K6OnEnh/view?usp=drive_web
4. https://drive.google.com/file/d/1X90ox__9O0X9_HYIklbQ3UzbD3KTNQXB/view?usp=drive_web

## Learning Data Extraction and Learning Annotation:

To train using 640 resolution images in YOLOv5, we first converted the video into 640 x 640 resolution footage.

### Video resolution adjustment

[비디오 리사이저 - 온라인에서 무료로 비디오 해상도 변경](https://online-video-cutter.com/ko/resize-video#google_vignette)

![image.png](image.png)

To create frame-by-frame images or annotations from the video with 640 x 640 resolution, we used DarkLabel, a well-known Video/Image Labeling and Annotation Tool.

### DarkLabel.zip

[https://drive.google.com/file/d/1Aow0IWcd7MrlhPos4uQFZVOqvRQUA6n5/view?usp=drive_web](https://drive.google.com/file/d/1Aow0IWcd7MrlhPos4uQFZVOqvRQUA6n5/view?usp=drive_web)

![image.png](image%201.png)

In the DarkLabel program, you can convert video into frame-by-frame images. First, select the 640 × 640 resolution video through "Open Video". Then, if "Labeled frames only" is checked, uncheck it. Afterwards, convert the video into images in a folder called "images" and “imag” using "as images".

![image.png](image%202.png)

You can check that the image came into the images and imag folder.

![image.png](f95c8a65-c6b3-45fd-899d-379f6284fee1.png)

![image.png](image%203.png)

Now the converted image is annotated through DarkLabel.

First, add classes through darklabel.yml before annotation.

Create my_classes2 and my_classes3 in the yaml file and add  car, RCcar, human/ bike, RCcar, human, house.

<aside>

my_classes2: ["car", "RCcar", "human"]
my_classes3: ["bike", "RCcar", "human", "house"]

</aside>

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

It can be seen that classes called darknet yolo were added to the DarkLabel program and three and four classes were added at the bottom.

![KakaoTalk_20241115_083045691_01.jpg](06196564-d645-4bb7-b117-c7551fa71645.png)

![KakaoTalk_20241115_083045691.jpg](f78ab710-6c7e-4b74-b0d8-30c054256b02.png)

The converted image was loaded by selecting the images and imag folders through Open Image Folder in DarkLabel. After selecting with Box + Label, annotations were made to the car, RCcar, and human that fit the class as shown in the photo below. After the annotation was completed, it was saved in each folder using GT Save As.

![image.png](image%204.png)

You can see that there is an annotated txt file in labels. 

![image.png](image%205.png)

## **Nvidia Jetson Nano Training Process**:

Install Google Colaboratory on Google Drive.

![KakaoTalk_20241120_085113586.jpg](b9807b03-aed7-4ddb-abbc-d92d830cd4c5.png)

Enter the code to connect to Google Drive in the command prompt:

<aside>

```
from google.colab import drive
drive.mount('/content/drive')
```

</aside>

### YLOLv5:

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

A folder named yolov5 is created, and inside it there is a Val file. Place the photos and txt files into the images and labels folders created in DarkLabel, respectively. Then, modify the data.yaml file according to the classes.

![화면 캡처 2024-11-20 091258.png](%25ED%2599%2594%25EB%25A9%25B4_%25EC%25BA%25A1%25EC%25B2%2598_2024-11-20_091258.png)

### **Image File Management**

We create folders to manage image files.

<aside>

```
!mkdir -p Train/labels
!mkdir -p Train/images
!mkdir -p Val/labels
!mkdir -p Val/images
```

</aside>

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

### **Start of YOLOv5 Model Training**

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

- `--img 512`: Sets the input image size to 512x640.
- `--batch 16`: Sets the batch size. This represents the number of images processed at once.
- `--epochs 300`: Sets the total number of epochs for training.
- `--data /content/drive/MyDrive/yolov5/data.yaml`: Specifies the path to the YAML file containing dataset and model configuration settings.
- `--weights yolov5n.pt`: Specifies the path to the pre-trained weights file. Here, the [yolov5n.pt](http://yolov5n.pt) file is being used.

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

After the training is complete, you can see that exp has been trained in the train folder inside the yolov5 folder.

![화면 캡처 2024-11-20 100417.png](%25ED%2599%2594%25EB%25A9%25B4_%25EC%25BA%25A1%25EC%25B2%2598_2024-11-20_100417.png)

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

### **Validation of YOLOv5 Model Training Results**

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

- `python detect.py`: Executes the Python script.
- `--weights runs/train/exp2/weights/best.pt`: Specifies the path to the weights file. These weights are obtained through model training and represent the best lane detection file compiled.
- `--img 512`: Sets the input image size to 512x640.
- `--conf 0.1`: Sets the confidence threshold for object detection to 0.1. This means the minimum probability at which the model considers an object to be detected. Only objects detected with a probability of 0.1 or higher are included in the results.
- `--source /content/drive/MyDrive/yolov5/Train/images`: Specifies the path to the input image or video file.
- `for imageName in glob.glob('/content/drive/MyDrive/yolov5/runs/detect/exp2/*.jpg')[:10]:` : Modify to .png or .jpg etc. according to the image file format

**detect.py execution image:**

![image.png](image%206.png)

**Learning outcomes with detect.py:**

![화면 캡처 2024-11-21 023726.png](%25ED%2599%2594%25EB%25A9%25B4_%25EC%25BA%25A1%25EC%25B2%2598_2024-11-21_023726.png)

![화면 캡처 2024-11-21 023755.png](%25ED%2599%2594%25EB%25A9%25B4_%25EC%25BA%25A1%25EC%25B2%2598_2024-11-21_023755.png)

![화면 캡처 2024-11-21 023901.png](%25ED%2599%2594%25EB%25A9%25B4_%25EC%25BA%25A1%25EC%25B2%2598_2024-11-21_023901.png)

![화면 캡처 2024-11-21 023928.png](%25ED%2599%2594%25EB%25A9%25B4_%25EC%25BA%25A1%25EC%25B2%2598_2024-11-21_023928.png)

**detect.py Execution Video**

```python
!python detect.py --weights /content/drive/MyDrive/yolov5/runs/train/exp/weights/best.pt --img 512 --conf 0.1 --source /content/drive/MyDrive/video.mp4
```

[yolov51]

[https://drive.google.com/file/d/1hL262ZSgrq0dQfRf7SpUJw4I5UnzONe_/view?usp=drive_link](https://drive.google.com/file/d/1hL262ZSgrq0dQfRf7SpUJw4I5UnzONe_/view?usp=drive_link)

[yolov5]

[https://drive.google.com/file/d/1hNufb75hTbkJbWdUpjuQcTs4eE4T-Sze/view?usp=drive_link](https://drive.google.com/file/d/1hNufb75hTbkJbWdUpjuQcTs4eE4T-Sze/view?usp=drive_link)