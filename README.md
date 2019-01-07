# MotionPathsExtraction
Approach to extract motion paths of vehicles and pedestrians from videos (Multiple object detection &amp; tracking)

To run the tool:

* Install prerequisites and run the python script (counting_tool.py), or
* just run the executable file (.exe file for windows; .app for mac will follow) 

![Poster](/poster/poster_A0_tracks.jpg)

## Installing Detectron

See Detectrons [INSTALL.md](https://github.com/mavoll/MotionPathsExtraction/blob/master/Detectron/INSTALL.md)

This document covers how to install Detectron, its dependencies (including Caffe2), and the COCO dataset.

- For general information about Detectron, please see [`README.md`](README.md).

**Requirements:**

- NVIDIA GPU, Linux, Python2
- Caffe2, various standard Python packages, and the COCO API; Instructions for installing these dependencies are found below

**Notes:**

- Detectron operators currently do not have CPU implementation; a GPU system is required.
- Detectron has been tested extensively with CUDA 8.0 and cuDNN 6.0.21.

### Caffe2

To install Caffe2 with CUDA support, follow the [installation instructions](https://caffe2.ai/docs/getting-started.html) from the [Caffe2 website](https://caffe2.ai/). **If you already have Caffe2 installed, make sure to update your Caffe2 to a version that includes the [Detectron module](https://github.com/pytorch/pytorch/tree/master/modules/detectron).**

Please ensure that your Caffe2 installation was successful before proceeding by running the following commands and checking their output as directed in the comments.

#### Please note:
This approach does not implement it´s own detection and tracking algorithms but makes use of the following algorithms:  
* Facebook´s [Detectron](https://github.com/facebookresearch/Detectron) Mask R-CNN implementation 
* [Deep SORT](https://github.com/nwojke/deep_sort) - Simple Online Realtime Tracking with a Deep Association Metric

## Installing deep_sort

See deep_sorts [README.md](https://github.com/mavoll/MotionPathsExtraction/blob/master/deep_sort/README.md)

### Dependencies

The code is compatible with Python 2.7 and 3. The following dependencies are
needed to run the tracker:

* NumPy
* sklearn
* OpenCV
* TensorFlow (>= 1.0; for feature generation) CUDA???

### Installation

First, clone the repository:
```
git clone https://github.com/nwojke/deep_sort.git
```
Then, download pre-generated detections and the CNN checkpoint file from
[here](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp).

We have replaced the appearance descriptor with a custom deep convolutional
neural network (see below).


## Authors

* **Marc-André Vollstedt** - marc.vollstedt@gmail.com

## Acknowledgments
