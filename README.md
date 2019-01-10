# MotionPathsExtraction
Approach to extract motion paths of vehicles and pedestrians from videos (Multiple object detection &amp; tracking)

#### Please note:
This approach does not implement it´s own detection and tracking algorithms but makes use of the following algorithms (cloned to this repository):  
* Facebook´s [Detectron](https://github.com/facebookresearch/Detectron) Mask R-CNN implementation 
* [Deep SORT](https://github.com/nwojke/deep_sort) - Simple Online Realtime Tracking with a Deep Association Metric

![Poster](/poster/poster_A0_tracks.jpg)

## Installing

- For general information about Detectron, please see [README.md](https://github.com/mavoll/MotionPathsExtraction/blob/master/Detectron/README.md).
- For general information about deep_sort, please see [README.md](https://github.com/mavoll/MotionPathsExtraction/blob/master/deep
_sort/README.md).

### Prerequisites: ###

- Linux
- NVIDIA GPU
- CUDA 8
- cuDNN 6
- Python 2.7
- Caffe2 (install Caffe2 with CUDA support, follow the [installation instructions](https://caffe2.ai/docs/getting-started.html))
- COCO API (see [here](https://github.com/mavoll/MotionPathsExtraction/blob/master/Detectron/INSTALL.md#coco) and [here](https://github.com/cocodataset/cocoapi))
- TensorFlow (== 1.4.0.; for feature generation)

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
```

**Detectron Troubleshooting**

[INSTALL.md](https://github.com/mavoll/MotionPathsExtraction/edit/master/Detectron/INSTALL.md)

**deep_sort Troubleshooting** 

[README.md](https://github.com/mavoll/MotionPathsExtraction/blob/master/deep_sort/README.md)

The pre-generated deep_sort CNN checkpoint files from [here](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp) are already included to this repository ([`resources`](https://github.com/mavoll/MotionPathsExtraction/edit/master/deep_sort/resources/networks/)).



## Authors

* **Marc-André Vollstedt** - marc.vollstedt@gmail.com

## Acknowledgments
