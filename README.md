# MotionPathsExtraction

UNDER CONSTRUCTION

per_process_gpu_mem_fraction = 0.1

imutils_queue_size = 128 

download detectron ressources (weights and configfiles) from here and extract to MotionPathsExtraction root directory
or
change folder path in config.ini and let detectron download corresponding resources
detectron_download_cache

download deep_sort ressources (tf model to generate person re-identification features) from here and extract to MotionPathsExtraction root directory

copy into or let auto download by chainer into:
$HOME/.chainer/dataset/pfnet/chainercv/models

Multi object and classes detection and tracking pipeline to extract motion paths of objects like vehicles and pedestrians from videos.

#### Please note:
This approach does not implement it´s own detection and tracking algorithms but makes use of the following algorithms:  
* Facebook´s [Detectron](https://github.com/facebookresearch/Detectron) Mask/Faster R-CNN implementations:
  * e2e_mask_rcnn_R-101-FPN_2x (coco_2014_train and coco_2014_valminusminival)
  * e2e_faster_rcnn_R-101-FPN_2x (coco_2014_train and coco_2014_valminusminival)
  * others from the [Detectron Model Zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md) can be used per config file and detectrons auto-download functionality
* [SORT](https://github.com/abewley/sort) - A simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences
* [Deep SORT](https://github.com/nwojke/deep_sort) - Simple Online Realtime Tracking with a Deep Association Metric

We also make use of [ChainerCV](https://github.com/chainer/chainercv) and its detector implementations of FasterR-CNN, SSD and YOLO:
* faster_rcnn_vgg16_voc0712_trained
* faster_rcnn_fpn_resnet50_coco_trained
* faster_rcnn_fpn_resnet101_coco_trained
* ssd300_voc0712_converted
* ssd512_voc0712_converted
* yolo_v2_tiny_voc0712_converted
* yolo_v2_voc0712_converted
* yolo_v3_voc0712_converted

![Poster](/poster/poster_A0_tracks.jpg?raw=true "MotionPathsExtraction")

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

I have installed Anaconda ([from here](https://www.anaconda.com/distribution/#linux)) to create an environment and to install components. 
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
chainer.__version__
cv2.__version__
tf.__version__
torch.__version__
```
## Install: ##

```
cd ~
```
```
git clone https://github.com/mavoll/MotionPathsExtraction.git
```
```
cd MotionPathsExtraction/Detectron
```
```
make
```
### Test Detectron: ###
```
python detectron/tests/test_spatial_narrow_as_op.py
```

[This](https://github.com/facebookresearch/Detectron) page provides detailed informations about Facebooks tracker Detectron and its Model Zoo.
Here is a end-to-end trained Mask R-CNN model with a ResNet-101-FPN backbone from [here](https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl) used.
All models based on the [COCO dataset](http://cocodataset.org/#home).

Pre-generated detections and the CNN checkpoint file for the tracker deep_sort has been downloaded from [here](https://drive.google.com/drive/folders/18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp) and can be found here:
```
MotionPathsExtraction/deep_sort/resources/networks/
```
More information about the tracker deep_sort, its training and its options can be found [here](https://github.com/nwojke/deep_sort).


**Detectron Troubleshooting**

[INSTALL.md](https://github.com/mavoll/MotionPathsExtraction/edit/master/Detectron/INSTALL.md)

**deep_sort Troubleshooting** 

[README.md](https://github.com/mavoll/MotionPathsExtraction/blob/master/deep_sort/README.md)

## Usage: ##

## Further Usage based on the tracking results: ##

### Count intersections

Use the [CountingTool](https://github.com/mavoll/TrafficCountingTool) to draw lines and count intersections. 

### Mapping

Use the [Mapping](?) to map pixel coordinates to geo-coordinates. 

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

### Tracks to the SparkPipeline

#### Import
[SparkPipeline](https://github.com/mavoll/SparkPipeline)

#### Using Apache Zeppelin and Spark to analyze and visualize tracks

## Authors

* **Marc-André Vollstedt** - marc.vollstedt@gmail.com

## Acknowledgments
