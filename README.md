# MotionPathsExtraction
Approach to extract motion paths of vehicles and pedestrians from videos (Multiple object detection &amp; tracking)

To run the tool:

* Install prerequisites and run the python script (counting_tool.py), or
* just run the executable file (.exe file for windows; .app for mac will follow) 

![Poster](/poster/poster_A0_tracks.pdf)

### Please note:
This approach does not implement it´s own detection and tracking algorithms but makes use of the following algorithms:  
* Facebook´s [Detectron](https://github.com/facebookresearch/Detectron) Mask R-CNN implementation 
* [Deep SORT](https://github.com/nwojke/deep_sort) - Simple Online Realtime Tracking with a Deep Association Metric

### Prerequisites and used versions

* Python 3.6
* OpenCV 3.2
* Pandas 0.19.2
* Tkinter 8.6
* openpyxl 2.4.1

## Authors

* **Marc-André Vollstedt** - marc.vollstedt@gmail.com

## Acknowledgments
