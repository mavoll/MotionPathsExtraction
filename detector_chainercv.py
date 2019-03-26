import matplotlib.pyplot as plt

import chainer

from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import coco_bbox_label_names
from chainercv.experimental.links import YOLOv2Tiny
from chainercv.links import YOLOv2
from chainercv.links import YOLOv3
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv.links import FasterRCNNVGG16
from chainercv.links import FasterRCNNFPNResNet50
from chainercv.links import FasterRCNNFPNResNet101
from chainercv.visualizations import vis_bbox

import numpy as np
import cv2

class Detector:
    
    def __init__(self, gpu, model, nms_thresh= 0.45, score_thresh=0.6):
        
        self.gpu = gpu
        
        if model == 'yolo_v2_tiny':
            self.model = YOLOv2Tiny(
                n_fg_class=len(voc_bbox_label_names),
                pretrained_model='voc0712')
            
        elif model == 'yolo_v3':
            self.model = YOLOv3(
                n_fg_class=len(voc_bbox_label_names),
                pretrained_model='voc0712')
            
        elif model == 'ssd300':
            self.model = SSD300(
                n_fg_class=len(voc_bbox_label_names),
                pretrained_model='voc0712')        
            
        elif model == 'ssd512':
            self.model = SSD512(
                n_fg_class=len(voc_bbox_label_names),
                pretrained_model='voc0712')
            
        elif model == 'fasterrcnnvgg16':
            self.model = FasterRCNNVGG16(
                n_fg_class=len(coco_bbox_label_names),
                pretrained_model='voc0712')
            
        elif model == 'fasterrcnnfpnresnet50':
            self.model = FasterRCNNFPNResNet50(
                n_fg_class=len(coco_bbox_label_names),
                pretrained_model='coco')
            
        elif model == 'fasterrcnnfpnresnet101':
            self.model = FasterRCNNFPNResNet101(
                n_fg_class=len(coco_bbox_label_names),
                pretrained_model='coco')
        
        else:
            self.model = YOLOv2(
                n_fg_class=len(voc_bbox_label_names),
                pretrained_model='voc0712')
            
        #self.model.nms_thresh = nms_thresh
        #self.model.score_thresh = score_thresh
            
        if self.gpu >= 0:
            chainer.cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()
        
    def predict(self, image):        
        
        img = Detector.trans_img_chainer(image)        
        return self.model.predict([img])
           

    def save_result_images(self, image, save_as, bboxes, labels, scores):
    
        if (self.model == 'fasterrcnnfpnresnet50') or (self.model == 'fasterrcnnfpnresnet101'):
            bbox, label, score = bboxes[0], labels[0], scores[0]
            vis_bbox(image, bbox, label, score, label_names=coco_bbox_label_names)
        else:
            vis_bbox(image, bboxes[0], labels[0], scores[0], label_names=voc_bbox_label_names)
        
        plt.savefig(save_as)        
        
    def trans_img_cv2(img):
        buf = np.asanyarray(img, dtype=np.uint8).transpose(1, 2, 0)
        dst = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        return dst

    # OpenCV -> Chainer
    def trans_img_chainer(img):
        buf = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dst = np.asanyarray(buf, dtype=np.float32).transpose(2, 0, 1)
        return dst