import configparser
import cv2
from imutils.video import FPS
from imutils.video import FileVideoStream
import numpy as np
import sys
import csv
import os
import queue
from threading import Thread
import time
import glob

from tkinter import Toplevel
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
from tkinter import messagebox
import tkinter.scrolledtext as tkst
from tkinter import Tk
from tkinter import N
from tkinter import S
from tkinter import W
from tkinter import E
from tkinter import END
from tkinter import IntVar
from tkinter import StringVar

from caffe2.python import workspace
import main_window
import config_window
import detector_chainercv
import tracker_deep_sort
import tracker_sort

from Detectron.detectron.core.config import assert_and_infer_cfg
from Detectron.detectron.core.config import cfg
from Detectron.detectron.core.config import merge_cfg_from_file
from Detectron.detectron.core.config import _merge_a_into_b
from Detectron.detectron.utils.io import cache_url
from Detectron.detectron.utils.timer import Timer
import Detectron.detectron.utils.vis as detectron_visualizator
from Detectron.detectron.utils.collections import AttrDict
import Detectron.detectron.core.test_engine as infer_engine
import Detectron.detectron.utils.c2 as c2_utils
import logging

from chainercv.datasets import voc_bbox_label_names
from collections import defaultdict

c2_utils.import_detectron_ops()
cv2.ocl.setUseOpenCL(False)
        
class LoggingQueueHandler(logging.Handler):
    
    def __init__(self, log_queue):        
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):        
        self.log_queue.put(record)

class App(object):
    
    def __init__(self, bulk_processing=False, show_logging=True, gpu_id=None, instance_id=None, config_file=None, input_folder=None, file_types=None):
        
        self.root = Tk()
        self.bulk_processing = bulk_processing        
        self.show_logging = show_logging
        self.cam = None
        self.cam_id = 0
        self.input_source = None
        self.source_changed = False
        self.opencv_thread = None
        self.input_folder = input_folder
        self.file_types = file_types
        
        if self.bulk_processing:  
            
            workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
            self.load_config_file(config_file)            
            self.app_gpu = gpu_id
            self.app_save_det_result_path = input_folder
            self.app_save_tracking_result_path = input_folder
            self.setup_logging(__name__)                       
            self.logger = logging.getLogger(__name__)
            
        
        else:   
            self.v_1: IntVar = IntVar()        
            self.v_2: IntVar = IntVar()        
            self.v_3: IntVar = IntVar()
            self.v_4: IntVar = IntVar()
            self.v_5: IntVar = IntVar()
            self.v_detector_to_use: StringVar = StringVar()
            self.v_tracker_to_use: StringVar = StringVar()
            self.v_detectron_model: StringVar = StringVar()
            self.v_chainer_model: StringVar = StringVar()           
            
    
            self.chainercv_models = ['yolo_v2', 'yolo_v3', 'yolo_v2_tiny', 
                                   'ssd300', 'ssd512', 
                                   'fasterrcnnvgg16', 'fasterrcnnfpnresnet50', 'fasterrcnnfpnresnet101']
            
            self.detectron_models = ['e2emaskrcnnR101FPN2x', 'e2efasterrcnnR101FPN2x', 'use config file settings']
            
            self.trackers = ['deep_sort', 'sort']
            
            self.detectors = ['detectron', 'chainer']
            
            self.load_config_file('config.ini')
            
            self.logger = logging.getLogger(__name__)
        
    def run(self):
        self.main_window = main_window.MainWindow(self)        
        self.root.mainloop()
        cv2.destroyAllWindows()
        sys.exit()
        
    def ask_for_options(self):
        self.config_window = config_window.ConfigWindow(self)

    def open_webcam(self):
        if self.cam_id is not None:
            self.input_source = self.cam_id
            self.source_changed = True
            self.start_video()

    def open_video(self):
        options = {}
        options['title'] = "Choose video"

        filename = askopenfilename(**options)

        if filename:
            self.input_source = filename
            self.source_changed = True
            self.start_video()
            
    def start_bulk(self):
            self.start_bulk_process()                
            
    def start_bulk_process(self):
            #ffmpeg -i GP067902.MP4 -vcodec copy -an GP067902_nosound.MP4
            for filename in glob.iglob(self.input_folder + '/**/*.' + self.file_types, recursive=True):
                self.input_source = filename
                self.source_changed = True
                self.start_video()
                self.root.mainloop()
            
    def start_video(self):
        if self.opencv_thread is None:            
            self.source_changed = False
            self.opencv_thread = Thread(target=self.run_opencv_thread)
            self.opencv_thread.daemon = True
            self.opencv_thread.start()
        
        if self.show_logging is True:
            self.show_logging_window()

    def run_opencv_thread(self):
        
        if self.app_display is True:
            cv2.namedWindow('source')
        
        self.start_processing()

        cv2.destroyAllWindows()
        self.opencv_thread = None

    def start_processing(self):
        if self.input_source is not None:

            file_stream = FileVideoStream(self.input_source, queue_size=self.app_imutils_queue_size).start()
            
            time.sleep(0.001)            
            detector = self.initializeDetector()
            self.tracker = self.initializeTracker()
            self.setDataset()
                
            fps = FPS().start()
            
            frame_id = 0
            all_boxes = {}
            tracking_boxes = []

            while (not self.source_changed) and file_stream.running(): 
                
                time.sleep(0.001)
                
                try:
                        
                    self.image = file_stream.read()
                
                    if frame_id % self.app_process_every_nth_frame == 0:                            
                                           
                        if(self.image is not None):

                            vis = self.image.copy()                            
                            cls_boxes = None 
                            cls_segms = None 
                            cls_keyps = None 
                            timers = defaultdict(Timer)
                            t = time.time()
                            fps.update()
                            fps.stop()
                            
                            self.logger.info('Processing frame {}'.format(frame_id))
                            
                            fps_text = "FPS " + "{:.2f}".format(fps.fps())
                            self.logger.info('FPS: ' + fps_text)        
                            
                            if self.app_do_detection and not self.source_changed:
                                
                                cls_boxes, cls_segms, cls_keyps = self.infer(vis, timers, detector)     
                                                                                                
                                all_boxes[frame_id] = cls_boxes
                                
                                self.logger.info('Inference time: {:.3f}s'.format(time.time() - t))
                                
                                for k, v in timers.items():
                                    self.logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
                                if frame_id == 0:
                                    self.logger.info(
                                        ' \ Note: inference on the first image will be slower than the '
                                        'rest (caches and auto-tuning need to warm up)'
                                    )            
                                fps_text = "FPS " + "{:.2f}".format(fps.fps())
                                self.logger.info('FPS: ' + fps_text)
            
                                                            
                                if self.app_display_det_result_img:
                                    if frame_id % self.app_display_det_every_nth_frame == 0 :                                        
                                        vis = self.visualize_det(vis, cls_boxes, fps_text, 
                                                                 segms=cls_segms, keypoints=cls_keyps)
      
            
                                if self.app_save_det_result_img:
                                    if not self.app_display_det_result_img:
                                        ret = self.visualize_det(vis, cls_boxes, fps_text, 
                                                                 segms=cls_segms, keypoints=cls_keyps)
                                        self.save_det_result_img(ret, frame_id)
                                    else:
                                        self.save_det_result_img(vis, frame_id)
                                    
                            if self.app_do_tracking and not App.is_list_empty(cls_boxes) and not self.source_changed:
                                
                                t = time.time()
                                tmp_tracking_boxes = self.track(self.image.copy(), cls_boxes, frame_id, timers)
                                                                
                                self.logger.info('Tracking time (incl. feature generation): {:.3f}s'.format(time.time() - t))

                                if self.app_display_tracking_result_img:
                                    if frame_id % self.app_display_tracking_every_nth_frame == 0 :                              
                                        vis = self.visualize_tracking(vis, tmp_tracking_boxes, fps_text)                                       
                                        
                                                                
                                if self.app_save_tracking_result_img:
                                    if not self.app_display_tracking_result_img:
                                        ret = self.visualize_tracking(vis, tmp_tracking_boxes, fps_text)
                                        self.save_tracking_result_img(ret, frame_id)
                                    else:
                                        self.save_tracking_result_img(vis, frame_id)
                                
                                tracking_boxes = self.extend_result_boxes(frame_id, tracking_boxes, tmp_tracking_boxes)
                                
                            if self.app_display:
                                cv2.imshow('source', vis)  
                                ch = 0xFF & cv2.waitKey(1)
                                if ch == 27:
                                    break
         
                            self.logger.info('Total time frame {}: {:.3f}s'.format(frame_id, time.time() - t))                            
                                
                            frame_id += 1                              

                except Exception:
                    print(sys.exc_info()[0] + sys.exc_info()[1])     
                    #continue
            
            if self.app_save_det_result_boxes:                                
                self.save_det_result_boxes(all_boxes)    
                self.logger.info('Wrote detections to: {}'.format(os.path.abspath(self.app_save_det_result_path)))
                
            if self.app_save_tracking_result_boxes:                                
                self.save_tracking_result_boxes(list(tracking_boxes))
                self.logger.info('Wrote tracks to: {}'.format(os.path.abspath(self.app_save_tracking_result_path)))
                
            file_stream.stop()
            self.source_changed = False
            
            if not self.bulk_processing:
                self.start_processing()

    def initializeDetector(self):
                
        if self.app_detector_to_use == 'chainer':
                        
            return detector_chainercv.Detector(self.app_gpu, 
                                               self.chainer_model, 
                                               self.chainer_ms_thresh, 
                                               self.chainer_score_thresh)
            
        elif self.app_detector_to_use == 'detectron':
            
            cfg.immutable(False)
            
            if self.detectron_model == 'e2emaskrcnnR101FPN2x':
                detectron_cfg_tmp = 'Detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml'
                detectron_wts_tmp = (str(self.detectron_download_cache) + '/35861858/12_2017_baselines/'
                                    'e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/'
                                    'coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl')
                                    
            elif self.detectron_model == 'e2efasterrcnnR101FPN2x':
                detectron_cfg_tmp = 'Detectron/configs/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_2x.yaml'
                detectron_wts_tmp = (str(self.detectron_download_cache) + '/35857952/12_2017_baselines/'
                                    'e2e_faster_rcnn_R-101-FPN_2x.yaml.01_39_49.JPwJDh92/output/train/'
                                    'coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl')
            else:
                detectron_cfg_tmp = self.detectron_cfg
                detectron_wts_tmp = self.detectron_wts
                
            merge_cfg_from_file(detectron_cfg_tmp)
            cfg.NUM_GPUS = 1
            cfg.DOWNLOAD_CACHE = self.detectron_download_cache                    
            cfg.TEST.NMS = self.detectron_nms_thresh
            cfg.TEST.DETECTIONS_PER_IM = self.detectron_detections_per_im
            cfg.TEST.PROPOSAL_LIMIT = self.detectron_proposal_limit
            cfg.TEST.RPN_NMS_THRESH = self.detectron_rpn_nms_thresh
            cfg.TEST.SCORE_THRESH = self.detectron_score_thresh
            weights = cache_url(detectron_wts_tmp, cfg.DOWNLOAD_CACHE)
            assert_and_infer_cfg()
            _merge_a_into_b(cfg, infer_engine.cfg)
            
            return infer_engine.initialize_model_from_cfg(weights, self.app_gpu)
                    
    def infer(self, image, timers, detector):
        
        cls_boxes = None 
        cls_segms = None 
        cls_keyps = None
        
        if self.app_detector_to_use == 'chainer':
            
            timers['chainer_detect'].tic()
            bboxes, labels, scores = detector.predict(image)
            timers['chainer_detect'].toc()
            cls_boxes = App.transform_to_detectron(bboxes, labels, scores)
            
        elif self.app_detector_to_use == 'detectron':
                                                        
            with c2_utils.NamedCudaScope(self.app_gpu):
                cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                    detector, image, None, timers=timers
                )
        
        return cls_boxes, cls_segms, cls_keyps
    
    def visualize_det(self, vis, cls_boxes, fps_text, segms=None, keypoints=None):
                                
        offsetX = 20
        offsetY = 20
        text_width = 100
        text_height = 10
        
        vis = detectron_visualizator.vis_one_image_opencv(vis, cls_boxes, segms=segms, keypoints=keypoints, 
                                                          thresh=self.app_vis_thresh,
                                                          kp_thresh=self.detectron_kp_thresh, 
                                                          show_box=True, 
                                                          dataset=self.dataset, 
                                                          show_class=True)
        
        #(text_width, text_height) = cv2.getTextSize(fps_text, fontScale=cv2.FONT_HERSHEY_SIMPLEX, thickness=1)
        cv2.rectangle(vis, 
                      (offsetX - 2, offsetY - text_height - 2), 
                      (offsetX + 2 + text_width, offsetY + 2), 
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(vis, fps_text, (offsetX, offsetY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        
        return vis     

    def save_det_result_img(self, vis, frame_id):
        
        try:   
            head, tail = os.path.split(self.input_source)   
            filename = os.path.splitext(tail)[0]                  
            cv2.imwrite(head + '/' + filename + '_' + str(frame_id) + '.png',vis)
            
        except Exception:

            e = sys.exc_info()[0]
            messagebox.showinfo("Error saving detection images", e)
            raise

    def save_det_result_boxes(self, all_boxes):
            
        try:
            target_lines = []
                   
            for frame_id, cls_boxes in all_boxes.items():
                for cls_idx in range(1, len(cls_boxes)):
                    
                    dataset = self.dataset.classes
                    
                    for dets in cls_boxes[cls_idx]:
                        
                        line = [frame_id + 1, '-1', round(dets[0], 1), round(dets[1], 1), 
                                round(dets[2] - dets[0], 1), round(dets[3] - dets[1], 1), 
                                round(dets[4] * 100, 1), dataset[cls_idx], '-1', '-1', '-1']
                        target_lines.append(line)
                                
            target_lines.sort(key=lambda x: x[0])
            
            head, tail = os.path.split(self.input_source)   
            filename = os.path.splitext(tail)[0]                  
            
            with open(head + '/' + filename + '_detections.csv', 'w+') as txtfile:
                wr = csv.writer(txtfile, lineterminator='\n')
                for val in target_lines:
                    wr.writerow(val)
                    
        except Exception:

            e = sys.exc_info()[0]
            messagebox.showinfo("Error saving detection images", e)
            raise
           
    def initializeTracker(self):
        
        if self.app_tracker_to_use == 'deep_sort':
            
            return tracker_deep_sort.DeepSortTracker(self.deep_sort_feature_model, 
                                                     self.deep_sort_max_cosine_distance, 
                                                     self.deep_sort_nn_budget, 
                                                     self.deep_sort_per_process_gpu_mem_fraction,
                                                     self.app_gpu)
            
        elif self.app_tracker_to_use == 'sort':
            
            return tracker_sort.Sort()
    
    def track(self, vis, cls_boxes, frame_id, timers):
        
        result = None
        cls_boxes = [np.append(item, i) for i, sublist in enumerate(cls_boxes) for item in sublist]
        cls_boxes = np.asarray(cls_boxes)
        
        if self.app_tracker_to_use == 'deep_sort':
            timers['deep_sort'].tic()
            result = self.tracker.track(vis, cls_boxes, frame_id, 
                                        self.deep_sort_min_detection_height, 
                                        self.deep_sort_min_confidence, 
                                        self.deep_sort_nms_max_overlap)
            timers['deep_sort'].toc()
            
        elif self.app_tracker_to_use == 'sort':
            timers['sort'].tic()
            cls_boxes[:,2:4] += cls_boxes[:,0:2]
            result = self.tracker.update(cls_boxes)
            timers['sort'].toc()             
         
        return result
        
    def visualize_tracking(self, vis, tracking_boxes, fps_text):
                
        offsetX = 20
        offsetY = 20
        text_width = 100
        text_height = 10        
            
        if self.app_tracker_to_use == 'deep_sort':        
            
            vis = self.tracker.draw_trackers(vis, tracking_boxes, 
                                             self.dataset.classes, 
                                             self.deep_sort_min_confidence)
            
        if self.app_tracker_to_use == 'sort':           
            
            vis = self.tracker.draw_trackers(vis, tracking_boxes)
        
        cv2.rectangle(vis, 
                      (offsetX - 2, offsetY - text_height - 2), 
                      (offsetX + 2 + text_width, offsetY + 2), 
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(vis, fps_text, (offsetX, offsetY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        
        return vis   
        
    def save_tracking_result_img(self, vis, frame_id):
        
        try:
                                
            cv2.imwrite(self.app_save_tracking_result_path + 
                        '/tracking_res_img_' + str(frame_id) + 
                        '.png',vis)
            
            head, tail = os.path.split(self.input_source)   
            filename = os.path.splitext(tail)[0]                  
            cv2.imwrite(head + '/' + filename + '_tracking_' + str(frame_id) + '.png',vis)
            
        except Exception:

            e = sys.exc_info()[0]
            messagebox.showinfo("Error saving tracking images", e)
            raise
        
    def save_tracking_result_boxes(self, tracking_boxes):                        
        
        try:
            dataset = self.dataset.classes
            tracking_boxes.sort(key = lambda x: (x[0], x[5]))
            
            head, tail = os.path.split(self.input_source)   
            filename = os.path.splitext(tail)[0]                  
            
            with open(head + '/' + filename + '_tracks.csv', 'w+') as txtfile:
                
                wr = csv.writer(txtfile, lineterminator='\n')
                
                for val in tracking_boxes:
                    
                    if self.app_tracker_to_use == 'deep_sort':            
                        val = [val[0] + 1, val[1], round(val[2], 1), round(val[3], 1), 
                               round(val[4], 1), round(val[5], 1), dataset[round(val[6], 1)]]
                        
                    if self.app_tracker_to_use == 'sort':
                        val = [val[0] + 1, int(val[5]), round(val[1], 1), 
                               round(val[2], 1), round(val[3], 1), round(val[4], 1)]
                        
                    wr.writerow(val)
                    
        except Exception:

            e = sys.exc_info()[0]
            messagebox.showinfo("Error saving tracking boxes", e)
            raise
    
    def extend_result_boxes(self, frame_id, tracking_boxes, tmp_tracking_boxes):
        
        if self.app_tracker_to_use == 'deep_sort':
            tracking_boxes.extend(self.tracker.get_confirmed_tracks(frame_id))
            
        elif self.app_tracker_to_use == 'sort':                                    
            for index, obj in enumerate(tmp_tracking_boxes):
                tracking_boxes.extend([[frame_id, obj[0], obj[1], obj[2], obj[3], obj[4]]])
                
        return tracking_boxes
    
    def setDataset(self):
        
        if self.app_detector_to_use == 'detectron':
            self.dataset = App.get_coco_dataset()
            
        elif self.app_detector_to_use == 'chainer' and (
                self.chainer_model == 'fasterrcnnfpnresnet50' or 
                self.chainer_model == 'fasterrcnnfpnresnet101'):
            self.dataset = App.get_coco_dataset(False)
            
        else:
            self.dataset = App.get_voc_dataset()
            
    def load_config_file(self, configfile=None):
        
        try:
        
            self.config = configparser.ConfigParser()
            self.config.sections()
            
            if configfile is None:
                
                options = {}
                options['title'] = "Choose config file"
                options['filetypes'] = [('Config file', '.ini')]
                options['defaultextension'] = "ini"
        
                filename = askopenfilename(**options)
        
                if filename:
                    configfile = filename
                    
            if configfile is not None:
            
                self.config.read(configfile)
                
                self.app_display = eval(self.config['App']['display'])
                self.app_gpu = int(self.config['App']['gpu'])    
                self.app_save_det_result_img = eval(self.config['App']['save_det_result_img'])
                self.app_save_det_result_boxes = eval(self.config['App']['save_det_result_boxes'])    
                self.app_save_det_result_path = self.config['App']['save_det_result_path']   
                self.app_save_tracking_result_img = eval(self.config['App']['save_tracking_result_img'])
                self.app_save_tracking_result_boxes = eval(self.config['App']['save_tracking_result_boxes'])
                self.app_save_tracking_result_path = self.config['App']['save_tracking_result_path']        
                self.app_display_det_result_img = eval(self.config['App']['display_det_result_img'])
                self.app_display_det_every_nth_frame = eval(self.config['App']['display_det_every_nth_frame'])
                self.app_process_every_nth_frame = eval(self.config['App']['process_every_nth_frame'])
                self.app_display_tracking_result_img = eval(self.config['App']['display_tracking_result_img'])
                self.app_display_tracking_every_nth_frame = eval(self.config['App']['display_tracking_every_nth_frame'])
                self.app_detector_to_use = self.config['App']['detector_to_use']
                self.app_tracker_to_use = self.config['App']['tracker_to_use']
                self.app_vis_thresh = float(self.config['App']['vis_thresh'])
                self.app_config_file = self.config['App']['config_file']
                self.app_do_detection = eval(self.config['App']['do_detection'])
                self.app_do_tracking = eval(self.config['App']['do_tracking'])
                self.app_imutils_queue_size = int(self.config['App']['imutils_queue_size'])
                self.cam_id = int(self.config['App']['web_cam'])
                
                self.detectron_model = self.config['Detectron']['model']
                self.detectron_cfg = self.config['Detectron']['cfg']
                self.detectron_wts = self.config['Detectron']['wts']
                self.detectron_kp_thresh = float(self.config['Detectron']['kp_thresh'])
                self.detectron_nms_thresh = float(self.config['Detectron']['nms_thresh'])
                self.detectron_download_cache = self.config['Detectron']['download_cache']
                self.detectron_detections_per_im = int(self.config['Detectron']['detections_per_im'])
                self.detectron_proposal_limit = int(self.config['Detectron']['proposal_limit'])
                self.detectron_rpn_nms_thresh = float(self.config['Detectron']['rpn_nms_thresh'])
                self.detectron_score_thresh = float(self.config['Detectron']['score_thresh'])
                
                self.deep_sort_min_confidence = float(self.config['deep_sort']['min_confidence'])
                self.deep_sort_nn_budget = int(self.config['deep_sort']['nn_budget'])
                self.deep_sort_max_cosine_distance = float(self.config['deep_sort']['max_cosine_distance'])
                self.deep_sort_nms_max_overlap = float(self.config['deep_sort']['nms_max_overlap'])
                self.deep_sort_min_detection_height = int(self.config['deep_sort']['min_detection_height'])
                self.deep_sort_per_process_gpu_mem_fraction = float(self.config['deep_sort']['per_process_gpu_mem_fraction'])
                
                self.deep_sort_feature_model = self.config['deep_sort_features']['model']
                
                self.chainer_model = self.config['chainer']['model']
                self.chainer_ms_thresh = float(self.config['chainer']['ms_thresh'])
                self.chainer_score_thresh = float(self.config['chainer']['score_thresh'])
                
                if self.opencv_thread is not None:
                        self.source_changed = True
                
                if not self.bulk_processing:
                    self.update_main_gui()
                
        except Exception:

            e = sys.exc_info()[0]
            messagebox.showinfo("Error loading file", e)
            raise
    
    def save_config_file(self, configfile=None):
        
        try:
        
            self.config['App'] = {'display': self.app_display,
                             'gpu': self.app_gpu,
                             'save_det_result_img': self.app_save_det_result_img,
                             'save_det_result_boxes': self.app_save_det_result_boxes,
                             'save_det_result_path': self.app_save_det_result_path,                             
                             'save_tracking_result_img': self.app_save_tracking_result_img,
                             'save_tracking_result_boxes': self.app_save_tracking_result_boxes,
                             'save_tracking_result_path': self.app_save_tracking_result_path,
                             'display_det_result_img': self.app_display_det_result_img,
                             'display_det_every_nth_frame': self.app_display_det_every_nth_frame,
                             'display_tracking_result_img': self.app_display_tracking_result_img,
                             'display_tracking_every_nth_frame': self.app_display_tracking_every_nth_frame,
                             'detector_to_use': self.app_detector_to_use,
                             'tracker_to_use': self.app_tracker_to_use,
                             'vis_thresh': self.app_vis_thresh,
                             'config_file': self.app_config_file,
                             'process_every_nth_frame': self.app_process_every_nth_frame,
                             'do_detection': self.app_do_detection,
                             'do_tracking': self.app_do_tracking,
                             'imutils_queue_size': self.app_imutils_queue_size,
                             'web_cam': self.cam_id}
            
            self.config['Detectron'] = {
                             'model': self.detectron_model,
                             'cfg': self.detectron_cfg,
                             'wts': self.detectron_wts,
                             'kp_thresh': self.detectron_kp_thresh,
                             'nms_thresh': self.detectron_nms_thresh,
                             'download_cache': self.detectron_download_cache,
                             'detections_per_im': self.detectron_detections_per_im,
                             'proposal_limit': self.detectron_proposal_limit,
                             'rpn_nms_thresh': self.detectron_rpn_nms_thresh,
                             'score_thresh': self.detectron_score_thresh}
            self.config['deep_sort'] = {'min_confidence': self.deep_sort_min_confidence,
                             'nn_budget': self.deep_sort_nn_budget,
                             'max_cosine_distance': self.deep_sort_max_cosine_distance,
                             'nms_max_overlap': self.deep_sort_nms_max_overlap,
                             'min_detection_height': self.deep_sort_min_detection_height,
                             'per_process_gpu_mem_fraction': self.deep_sort_per_process_gpu_mem_fraction
                             }
            self.config['deep_sort_features'] = {'model': self.deep_sort_feature_model}
            self.config['chainercv'] = {'model': self.chainer_model,
                             'ms_thresh': self.chainer_ms_thresh,
                             'score_thresh': self.chainer_score_thresh}
                            
            options = {}
            options['filetypes'] = [('Config file', '.ini')]
            options['defaultextension'] = "ini"
            options['initialfile'] = "config.ini"
            options['title'] = "Where to save?"
    
            configfile = asksaveasfilename(**options)
    
            if configfile:

                with open(configfile, 'w') as configfile:
                        self.config.write(configfile)
    
        except Exception:

            e = sys.exc_info()[0]
            messagebox.showinfo("Error saving file", e)
            raise
                   
    def trans_img_chainer(img):
        buf = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dst = np.asanyarray(buf, dtype=np.uint8).transpose(2, 0, 1)
        return dst
    
    def transform_to_detectron(bbox, label, score):
        #Detectron: all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
        label_names = voc_bbox_label_names        
        cls_boxes = [[] for _ in range(len(label_names))]
        
        if label is not None and not len(bbox) == len(label):
            raise ValueError('The length of label must be same as that of bbox')
        if score is not None and not len(bbox) == len(score):
            raise ValueError('The length of score must be same as that of bbox')
    
        if label is not None:
            order = np.argsort(label)
            
            bbox = np.array(bbox[0])
            score = np.array(score[0])
            label = np.array(label[0])
            
            bbox = bbox[order]
            score = score[order]
            label = label[order]
            
            if len(bbox) == 0:
                return
            
            tmp_label = None
            for i, bb in enumerate(bbox[0]):
                if tmp_label is None or tmp_label != label[0][i]:
                    tmp_label = label[0][i]
                cls_boxes[tmp_label].append([bb[1], bb[0], bb[3], bb[2], score[0][i]])
                
                 
        return cls_boxes
    
    def get_coco_dataset(incl_background=True):
        ds = AttrDict()
        classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        if incl_background:
            ds.classes = {i: name for i, name in enumerate(classes)}
        else:
            ds.classes = {i: name for i, name in enumerate(classes[1:])}        
        return ds
    
    def get_voc_dataset():
        ds = AttrDict()
    
        classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
            'train', 'tvmonitor'
        ]
        ds.classes = {i: name for i, name in enumerate(classes)}
        return ds
    
    def is_list_empty(inList):
        if isinstance(inList, list):
            return all( map(App.is_list_empty, inList) )
        return False    

        
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            if self.cam is not None:
                self.cam.release()
            
            sys.exit()
    
    def set_detector_to_use(self, event=None):
        if event is not None:
            self.app_detector_to_use = str(event.widget.get())
            if self.opencv_thread is not None:
                self.source_changed = True
    
    def set_tracker_to_use(self, event=None):
        if event is not None:
            self.app_tracker_to_use = str(event.widget.get())
            if self.opencv_thread is not None:
                self.source_changed = True
    
    def set_detectron_model(self, event=None):
        if event is not None:
            self.detectron_model = str(event.widget.get())
            if self.opencv_thread is not None:
                self.source_changed = True
    
    def set_chainercv_model(self, event=None):
        if event is not None:
            self.chainer_model = str(event.widget.get())
            if self.opencv_thread is not None:
                self.source_changed = True
            
    def set_display_det_result_img(self):
        self.app_display_det_result_img = bool(self.v_1.get())
            
    def set_display_tracking_result_img(self):
        self.app_display_tracking_result_img = bool(self.v_4.get())
    
    def set_do_tracking(self):
        self.app_do_tracking = bool(self.v_5.get())        
            
    def set_save_det_result_img(self):
        self.app_save_det_result_img = bool(self.v_2.get())
            
    def set_save_tracking_result_img(self):
        self.app_save_tracking_result_img = bool(self.v_3.get())

    def update_main_gui(self):
                
        self.v_1.set(self.app_display_det_result_img)
        self.v_2.set(self.app_save_det_result_img)
        self.v_4.set(self.app_display_tracking_result_img)
        self.v_3.set(self.app_save_tracking_result_img)
        self.v_5.set(self.app_do_tracking)
        
        self.v_detector_to_use.set(self.app_detector_to_use)
        self.v_tracker_to_use.set(self.app_tracker_to_use)
        self.v_detectron_model.set(self.detectron_model)
        self.v_chainer_model.set(self.chainer_model)        
        
        self.root.update()
    
    def show_logging_window(self):
        
        if self.bulk_processing:
            self.window = self.root
        else:
            self.window = Toplevel(self.root)
        
        self.window.wm_title("Process/GPU: " + str(self.app_gpu))
        self.window.resizable(width=True, height=True)
        self.window.attributes('-topmost', True)
                
        self.scrolled_text = tkst.ScrolledText(self.window, state='disabled', height=24)
        self.scrolled_text.grid(row=0, column=0, sticky=(N, S, W, E))
        self.scrolled_text.configure(font='TkFixedFont')
        self.scrolled_text.tag_config('INFO', foreground='black')
        self.scrolled_text.tag_config('DEBUG', foreground='gray')
        self.scrolled_text.tag_config('WARNING', foreground='orange')
        self.scrolled_text.tag_config('ERROR', foreground='red')
        self.scrolled_text.tag_config('CRITICAL', foreground='red', underline=1)
        self.log_queue = queue.Queue()
        self.queue_handler = LoggingQueueHandler(self.log_queue)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        self.queue_handler.setFormatter(formatter)
        self.logger.addHandler(self.queue_handler)
        self.window.after(100, self.poll_log_queue)
    
    def display_logging(self, record):        
        msg = self.queue_handler.format(record)
        self.scrolled_text.configure(state='normal')
        self.scrolled_text.insert(END, msg + '\n', record.levelname)
        self.scrolled_text.configure(state='disabled')
        self.scrolled_text.yview(END)

    def poll_log_queue(self):        
        while True:
            try:
                record = self.log_queue.get(block=False)
            except queue.Empty:
                break
            else:
                self.display_logging(record)
        self.window.after(1000, self.poll_log_queue)
        
    def setup_logging(self, name):        
        FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
        # Manually clear root loggers to prevent any module that may have called
        # logging.basicConfig() from blocking our logging setup
        logging.root.handlers = []
        logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
        logger = logging.getLogger(name)
        return logger
        
if __name__ == '__main__':
    
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    app = App()
    app.setup_logging(__name__)
    app.run()
