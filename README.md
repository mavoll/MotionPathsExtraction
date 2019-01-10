# MotionPathsExtraction
Approach to extract motion paths of vehicles and pedestrians from videos (Multiple object detection &amp; tracking)

#### Please note:
This approach does not implement it´s own detection and tracking algorithms but makes use of the following algorithms (cloned to this repository):  
* Facebook´s [Detectron](https://github.com/facebookresearch/Detectron) Mask R-CNN implementation 
* [Deep SORT](https://github.com/nwojke/deep_sort) - Simple Online Realtime Tracking with a Deep Association Metric

To run the tool:

* Install prerequisites 
* and run the python script (counting_tool.py)

![Poster](/poster/poster_A0_tracks.jpg)

## Installing

This document covers how to install Detectron, its dependencies (including Caffe2), and the COCO dataset.

- For general information about Detectron, please see [README.md](https://github.com/mavoll/MotionPathsExtraction/blob/master/Detectron/README.md).

### Requirements: ###

- NVIDIA GPU (CUDA 8.0 and cuDNN 6.0.21.)
- Linux
- Python2
- Caffe2 (install Caffe2 with CUDA support, follow the [installation instructions](https://caffe2.ai/docs/getting-started.html))
- various standard Python packages (see [requirements.txt](https://github.com/mavoll/MotionPathsExtraction/blob/master/requirements.txt))
- COCO API; Instructions for installing these dependencies are found below

* NumPy
* sklearn
* OpenCV
* TensorFlow (>= 1.0; for feature generation) CUDA???TensorFlow 1.5

Clone the repository:

```
# MotionPathsExtraction=/path/to/clone/MotionPathsExtraction
git clone https://github.com/mavoll/MotionPathsExtraction.git $MotionPathsExtraction
```

Install Python dependencies:

```
pip install -r $MotionPathsExtraction/requirements.txt
```

Set up Python modules for Detectron:

```
cd $MotionPathsExtraction/Detectron && make
```

Check that Detectron tests pass (e.g. for [`SpatialNarrowAsOp test`](detectron/tests/test_spatial_narrow_as_op.py)):

```
python $MotionPathsExtraction/Detectron/detectron/tests/test_spatial_narrow_as_op.py
`

**Detectron Troubleshooting**

[INSTALL.md](https://github.com/mavoll/MotionPathsExtraction/edit/master/Detectron/INSTALL.md)

**deep_sort Troubleshooting** 

[README.md](https://github.com/mavoll/MotionPathsExtraction/blob/master/deep_sort/README.md)

The pre-generated CNN checkpoint file from [here](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp) is already included to this repository ([`resources`](https://github.com/mavoll/MotionPathsExtraction/edit/master/deep_sort/resources/networks/)).



## Authors

* **Marc-André Vollstedt** - marc.vollstedt@gmail.com

## Acknowledgments
