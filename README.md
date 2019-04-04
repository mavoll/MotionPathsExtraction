# MotionPathsExtraction

Multi object and classes detection and tracking pipeline to extract motion paths of objects like vehicles and pedestrians from videos.

#### Please note:
This approach does not implement it´s own detection and tracking algorithms but makes use of the following algorithms:  
* Facebook´s [Detectron](https://github.com/facebookresearch/Detectron) Mask/Faster R-CNN implementations:
  * e2e_mask_rcnn_R-101-FPN_2x (coco_2014_train and coco_2014_valminusminival)
  * e2e_faster_rcnn_R-101-FPN_2x (coco_2014_train and coco_2014_valminusminival)
  * others from the [Detectron Model Zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md) can be used per config file and detectrons auto-download functionality
* [SORT](https://github.com/abewley/sort) - A simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences
* [Deep SORT](https://github.com/nwojke/deep_sort) - Simple Online Realtime Tracking with a Deep Association Metric

I recommend to have a look at:

-  [Simple Online and Realtime Tracking](http://arxiv.org/abs/1602.00763)
-  [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)
-  [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497)

We also make use of [ChainerCV](https://github.com/chainer/chainercv) and its detector implementations of FasterR-CNN, SSD and YOLO:
* faster_rcnn_vgg16_voc0712_trained
* faster_rcnn_fpn_resnet50_coco_trained
* faster_rcnn_fpn_resnet101_coco_trained
* ssd300_voc0712_converted
* ssd512_voc0712_converted
* yolo_v2_tiny_voc0712_converted
* yolo_v2_voc0712_converted
* yolo_v3_voc0712_converted

![Poster](/poster/poster_A0.jpg?raw=true "MotionPathsExtraction")

## Prerequisites: ###

- Ubuntu 16.04 or 18.04
- CUDA-ready NVIDIA GPU ([check](https://www.geforce.com/hardware/technology/cuda/supported-gpus))
- CUDA >= 9.0 (< 10.1)
- cuDNN >= 7.1.3
- Python 2.7 or 3.6 (not 3.7)
- OpenCV 3.4 (not 4)
- Caffe2 >= 0.7.0 or PyTorch >= 1.0 (to do inference and training with detector Detectron)
- COCO API (see [here](https://github.com/mavoll/MotionPathsExtraction/blob/master/Detectron/INSTALL.md#coco) and [here](https://github.com/cocodataset/cocoapi))
- TensorFlow >= 1.4.0 (person re-identification feature generation for tracker deep_sort)
- Chainer >= 4
- cupy >= 4
- chainercv >= 0.10 (to do inference and training with detector chainercv and its implementations of FasterR-CNN, SSD and YOLO)
- (Anaconda 2018.12)

## Get your environment ready: ###

Tested with:
- NVIDIA GeForce GTX 1080ti 11 GB (Ubuntu 16.04, python 2.7, CUDA 9.0, cuDNN 7.1.3, Driver 384.111, TensorFlow 1.8.1, Caffe2 0.7.0 , OpenCV 3.4, Chainer 5.3.0, Cupy 5.1.0, , ChainerCV 0.10)
- Dual-GPU: 2 x NVIDIA GeForce GTX 1080ti 11 GB (Ubuntu 16.04, python 2.7, CUDA 9.0, cuDNN 7.1.3, Driver 384.111, TensorFlow 1.8.1, Caffe2 0.7.0 , OpenCV 3.4, Chainer 5.3.0, Cupy 5.1.0, , ChainerCV 0.10)
- NVIDIA GeForce RTX 2070 8 GB (Ubuntu 18.04, python 3.6, CUDA 10.0, cuDNN 7.3.1, Driver 418.43, TensorFlow 1.11.0, PyTorch (Caffe2) 1.0.1, OpenCV 3.4, Chainer 5.3.0, Cupy 5.1.0 , ChainerCV 0.10)

I installed Anaconda ([from here](https://www.anaconda.com/distribution/#linux)) to create an environment and to install components.
For example:


### Install Anaconda: ###

```
wget https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh
```
```
bash Anaconda3-2018.12-Linux-x86_64.sh
```
```
source ~/.bashrc
```
```
conda info
```
```
conda update conda
```
```
conda update anaconda
```
```
conda update anaconda
```

### Create environment: ###

```
conda create --name envName python=3.6
```
```
conda activate envName
```
```
conda install ipykernel
```
```
python -m ipykernel install --user --name envName
```

### Install packages: ###

```
conda install pip  
```
```
pip install imutils
```
```
conda install numpy pyyaml matplotlib setuptools scipy protobuf future mkl mkl-include libtool
```
```
conda install -c mingfeima mkldnn
```
```
conda install -c nnpack
```
```
conda install -c conda-forge ffmpeg
```
```
conda install -c conda-forge opencv=3.4
```
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```
```
conda install tensorflow-gpu
```
```
conda install scikit-learn
```
```
conda install -c hcc pycocotools
```
```
conda install -c anaconda chainer
```
```
conda install cupy
```
```
pip install chainercv
```
```
conda install -c conda-forge filterpy
```

### Test Installations within Python: ###

```
import chainer
import cv2
import tensorflow as tf
import torch
import caffe2
chainer.__version__
cv2.__version__
tf.__version__
torch.__version__
caffe2.__version__
```

## Install: ##

```
cd ~
```
```
git clone https://github.com/mavoll/MotionPathsExtraction.git
```
```
cd MotionPathsExtraction
```
```
git clone https://github.com/facebookresearch/Detectron.git
```
```
cd Detectron
```
```
make
```
```
cd ..
```
```
git clone https://github.com/nwojke/deep_sort.git
```
```
git clone https://github.com/abewley/sort.git
```

### Download Resources: ###

You need to download some resources in order to make it work. 

-  Pre-generated CNN checkpoint file (mars-small128.pb) from [here](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp) provided from [Deep SORT](https://github.com/nwojke/deep_sort). 
-  End-to-End Faster & Mask R-CNN Baselines from Detectrons Model Zoo ([Faster-RCNN: R-101-FPN](https://dl.fbaipublicfiles.com/detectron/35857952/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_2x.yaml.01_39_49.JPwJDh92/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl) and [Mask-RCNN: R-101-FPN](https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl) used here.

You can use other models from the model zoo by downloading them and set cfg and wts within config.ini accordingly. 

The easiest way is to use the auto download functionalities provided by Detectron and Chainer and additionally download manually CNN checkpoint file which is used for feature generation (Tensorflow) by the Tracker Deep SORT.

Just make sure that you set following parameter in config.ini:

```
[Detectron]
download_cache = SOMEPATH/detectron-download-cache

[deep_sort_features]
model = SOMEPATH/networks/mars-small128.pb
```
`ChainerCV will automatically download to ~/.chainer`.

```
python detect_and_track.py
```

### Parameter: ###
Use the GUI or the config.ini to change and test detector or tracker related parameter.

Two parameter to mention:

`per_process_gpu_mem_fraction = 0.1` is set here depending on used GPU. It is necessary to make sure that the GPU can load both, the detector and the tracker model, at the same time to be able to initialize detector and tracker at the beginning of the pipeline and not for each frame or detector and tracker sequentially.  

`imutils_queue_size = 128` sets the input buffer queue size. 0 is infinite. Imagine, that this queue will got filled fast if input frame rate is higher than processing framerate. Thanks to [imutils](https://github.com/jrosebr1/imutils) for providing this out-of-the-box.


[This](https://github.com/facebookresearch/Detectron) page provides detailed information about Facebooks tracker Detectron and its Model Zoo.
Here is a end-to-end trained Mask R-CNN model with a ResNet-101-FPN backbone from [here](https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl) used.
All models based on the [COCO dataset](http://cocodataset.org/#home).

**Detectron Troubleshooting**

[INSTALL.md](https://github.com/mavoll/MotionPathsExtraction/edit/master/Detectron/INSTALL.md)

**deep_sort Troubleshooting**

[README.md](https://github.com/mavoll/MotionPathsExtraction/blob/master/deep_sort/README.md)

## Usage: ##

-  Change options, save and load config files
-  Choose detector (and options) to use
-  Choose tracker (and options) to use 
-  Choose input source and run

<p align="center">
  <img src="/images/gui.png" width="300" align="middle">
  <img src="/images/config_gui.png" width="500" align="middle">
</p>

-  See FPS (also displayed on screen) and other timer infos on logging window

<p align="center">
  <img src="/images/logging_window.png" width="800" align="middle">
</p>

Mask R-CNN in combination with deep_sort (here 1 FPS on GeForce RTX2070 8GB):

<p align="center">
  <img src="/images/maskrcnn.png" width="800" align="middle">
</p>

Faster R-CNN in combination with deep_sort (here 5 FPS on GeForce RTX2070 8GB):

<p align="center">
  <img src="/images/fasterrcnn_deep_sort.png" width="800" align="middle">
</p>

SSD and SORT (here 15 FPS on GeForce RTX2070 8GB):

<p align="center">
  <img src="/images/ssd.png" width="800" align="middle">
</p>

You have to consider, that if you want to have a Faster R-CNN -powerful detection engine running almost in real-time, than it might be possible or good enough with limited FPS. But if you also need to get good tracking results with less identity switches, than you need to feed your tracker with >20 FPS. So a real-time detection/tracking pipeline (with Detections of Pedestrians on Faster R-CNN level) is hard to archive (especially with mobile Hardware like NVIDIA Jetson TX2 but also not possible with cards like GTX 1080 Ti 11GB or RTX 2070 8GB). SSD with SORT runs on NVIDIA TX2 almost fluently. See later results.

## Multi GPU batch usage (dataset-level): ###

Only on dataset level, because tracker and detector models fit into one GPU at the same time so using multiple processes (Pythons multiprocessing) and instances, one per GPU, to batch process bigger input data should be ok.  

## Further Usage based on the tracking results: ##

### Count intersections

Use the [CountingTool](https://github.com/mavoll/TrafficCountingTool) to draw lines and count intersections.

### Mapping

Use the [Mapping](https://github.com/mavoll/SimpleTPSMapping) to map pixel coordinates to geo-coordinates.

### Animate and analyze tracks

#### Import tracking results to PostGIS

Install PostGIS:
```
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt bionic-pgdg main" >> /etc/apt/sources.list'
wget --quiet -O - http://apt.postgresql.org/pub/repos/apt/ACCC4CF8.asc | sudo apt-key add -
sudo apt update
sudo apt-get install postgresql-11-postgis-2.5
sudo apt install postgis
```
For configuration see [here](http://trac.osgeo.org/postgis/wiki/UsersWikiPostGIS24UbuntuPGSQL10Apt).

Use one of the python scripts  ??? to import tracks to PostGIS as LineStringM or PointM data.

#### Using QGIS and its TimeManager

Install QGIS like [here](https://freegistutorial.com/how-to-install-qgis-on-ubuntu-18-04-bionic-beaver/) and connect to the former installed and configured PostGIS.

<video>
<source src="/images/map_animation_heatmap_persons.mp4" type='video/mp4'>
</video>

### Tracks to the SparkPipeline

#### Import
[SparkPipeline](https://github.com/mavoll/SparkPipeline)

#### Using Apache Zeppelin and Spark to analyze and visualize tracks

## Further development and research opportunities

* Use-case specific circumstances:
  * Due to tracking-related identity switches this approach produces (depending on the crowded scene) shorter and fractional trajectories compared for example with approaches using mobile GPS devices to produce long and unique tracks. 
  * We are using multiple cams and perspectives to observe the whole plaza. Those perspectives only overlap at the edges. 
* Corresponding research questions:
  * Besides improvement within the field of multiple object detection and tracking producing less and less identity switches, is it possible to develop a post-processing process to re-connect corresponding trajectories or to generalize those trajectories in order to use well known analysis techniques requiring longer trajectories.
  * How to connect the trajectories from diffenet perspectives if objects moving from one perspective/cam to another.
  * Find a metric (besides multiple object detection as well as tracking challenges like for example the MOT challange) to evaluate the whole process of trajectory extraction focusing on the needs of social scientists and there requirements on the data to be able to produce significant experiments. 
  * State-of-the-art path manipulation and analysis keeping our specific circumstances in mind.
  * Using extracted real motion paths of vehicles and pedestrians to train, test and evaluate Agent-based simulation models or to configurate models with help from real countings/statistics (for example the measured usual and real frequency and dwell time of Pedestrians and cars at specific time and place)?


## Authors

* **Marc-André Vollstedt** - marc.vollstedt@gmail.com

## Acknowledgments
