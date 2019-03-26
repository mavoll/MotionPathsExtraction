from tkinter import Button
from tkinter import Entry
from tkinter import Toplevel
from tkinter import Label
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfilename

class ConfigWindow(object):
    
    def __init__(self, parent):                
        self.window = Toplevel(parent.root)
        self.window.wm_title("Options")                
        self.window.resizable(width=False, height=False)
        self.window.geometry('{}x{}'.format(460, 650))
        self.window.attributes('-topmost', True)        
        
        self.e_app_web_cam = Entry(self.window)
        self.e_app_web_cam.insert(5, parent.cam_id)
        self.e_app_web_cam.grid(row=0, column=1)
        Label(self.window, text="cam_id:").grid(row=0, column=0)  
        
        self.e_app_gpu = Entry(self.window)
        self.e_app_gpu.insert(5, parent.app_gpu)
        self.e_app_gpu.grid(row=1, column=1)
        Label(self.window, text="app_gpu:").grid(row=1, column=0)  
        
        self.e_app_display_det_every_nth_frame = Entry(self.window)
        self.e_app_display_det_every_nth_frame.insert(5, parent.app_display_det_every_nth_frame)
        self.e_app_display_det_every_nth_frame.grid(row=2, column=1)
        Label(self.window, text="app_display_det_every_nth_frame:").grid(row=2, column=0)   
        
        self.e_app_display_tracking_every_nth_frame = Entry(self.window)
        self.e_app_display_tracking_every_nth_frame.insert(5, parent.app_display_tracking_every_nth_frame)
        self.e_app_display_tracking_every_nth_frame.grid(row=3, column=1)
        Label(self.window, text="app_display_tracking_every_nth_frame:").grid(row=3, column=0)    
          
        self.e_app_process_every_nth_frame = Entry(self.window)
        self.e_app_process_every_nth_frame.insert(5, parent.app_process_every_nth_frame)
        self.e_app_process_every_nth_frame.grid(row=4, column=1)
        Label(self.window, text="app_process_every_nth_frame:").grid(row=4, column=0)
        
        self.e_app_imutils_queue_size = Entry(self.window)
        self.e_app_imutils_queue_size.insert(5, parent.app_imutils_queue_size)
        self.e_app_imutils_queue_size.grid(row=5, column=1)
        Label(self.window, text="app_imutils_queue_size:").grid(row=5, column=0)
        
        self.e_app_save_det_result_path = Entry(self.window)
        self.e_app_save_det_result_path.insert(5, parent.app_save_det_result_path)
        self.e_app_save_det_result_path.grid(row=6, column=1)
        Label(self.window, text="app_save_det_result_path:").grid(row=6, column=0)
        
        btn6 = Button(self.window, text="change",
                      command=lambda *args: self.set_save_det_result_path(parent))
        btn6.grid(row=7, column=1)
        
        self.e_detectron_wts = Entry(self.window)
        self.e_detectron_wts.insert(5, parent.detectron_wts)
        self.e_detectron_wts.grid(row=8, column=1)
        Label(self.window, text="detectron_wts:").grid(row=8, column=0)
        btn4 = Button(self.window, text="change",
                      command=lambda *args: self.set_detectron_wts(parent))
        btn4.grid(row=9, column=1)
        
        self.e_detectron_cfg = Entry(self.window)
        self.e_detectron_cfg.insert(5, parent.detectron_cfg)
        self.e_detectron_cfg.grid(row=10, column=1)
        Label(self.window, text="detectron_cfg:").grid(row=10, column=0)
        
        btn5 = Button(self.window, text="change",
                      command=lambda *args: self.set_detectron_cfg(parent))
        btn5.grid(row=11, column=1)
        
        self.e_detectron_kp_thresh = Entry(self.window)
        self.e_detectron_kp_thresh.insert(5, parent.detectron_kp_thresh)
        self.e_detectron_kp_thresh.grid(row=12, column=1)
        Label(self.window, text="detectron_kp_thresh:").grid(row=12, column=0)
        
        self.e_detectron_nms_thresh = Entry(self.window)
        self.e_detectron_nms_thresh.insert(5, parent.detectron_nms_thresh)
        self.e_detectron_nms_thresh.grid(row=13, column=1)
        Label(self.window, text="detectron_nms_thresh:").grid(row=13, column=0)
        
        self.e_detectron_download_cache = Entry(self.window)
        self.e_detectron_download_cache.insert(5, parent.detectron_download_cache)
        self.e_detectron_download_cache.grid(row=14, column=1)
        Label(self.window, text="detectron_download_cache:").grid(row=14, column=0)
        
        btn3 = Button(self.window, text="change",
                      command=lambda *args: self.set_download_cache(parent))
        btn3.grid(row=15, column=1)
        
        self.e_detectron_detections_per_im = Entry(self.window)
        self.e_detectron_detections_per_im.insert(5, parent.detectron_detections_per_im)
        self.e_detectron_detections_per_im.grid(row=16, column=1)
        Label(self.window, text="detectron_detections_per_im:").grid(row=16, column=0)
        
        self.e_detectron_proposal_limit = Entry(self.window)
        self.e_detectron_proposal_limit.insert(5, parent.detectron_proposal_limit)
        self.e_detectron_proposal_limit.grid(row=17, column=1)
        Label(self.window, text="detectron_proposal_limit:").grid(row=17, column=0)
        
        self.e_detectron_rpn_nms_thresh = Entry(self.window)
        self.e_detectron_rpn_nms_thresh.insert(5, parent.detectron_rpn_nms_thresh)
        self.e_detectron_rpn_nms_thresh.grid(row=18, column=1)
        Label(self.window, text="detectron_rpn_nms_thresh:").grid(row=18, column=0)
        
        self.e_detectron_score_thresh = Entry(self.window)
        self.e_detectron_score_thresh.insert(5, parent.detectron_score_thresh)
        self.e_detectron_score_thresh.grid(row=19, column=1)
        Label(self.window, text="detectron_score_thresh:").grid(row=19, column=0)
        
        self.e_chainer_ms_thresh = Entry(self.window)
        self.e_chainer_ms_thresh.insert(5, parent.chainer_ms_thresh)
        self.e_chainer_ms_thresh.grid(row=20, column=1)
        Label(self.window, text="chainer_ms_thresh:").grid(row=20, column=0)
        
        self.e_chainer_score_thresh = Entry(self.window)
        self.e_chainer_score_thresh.insert(5, parent.chainer_score_thresh)
        self.e_chainer_score_thresh.grid(row=21, column=1)
        Label(self.window, text="chainer_score_thresh:").grid(row=21, column=0)
        
        self.e_app_save_tracking_result_path = Entry(self.window)
        self.e_app_save_tracking_result_path.insert(5, parent.app_save_tracking_result_path)
        self.e_app_save_tracking_result_path.grid(row=22, column=1)
        Label(self.window, text="app_save_tracking_result_path:").grid(row=22, column=0)
        
        btn2 = Button(self.window, text="change",
                      command=lambda *args: self.set_save_tracking_result_path(parent))
        btn2.grid(row=23, column=1)
        
        
        self.e_deep_sort_min_confidence = Entry(self.window)
        self.e_deep_sort_min_confidence.insert(5, parent.deep_sort_min_confidence)
        self.e_deep_sort_min_confidence.grid(row=24, column=1)
        Label(self.window, text="deep_sort_min_confidence:").grid(row=24, column=0)
        
        self.e_deep_sort_nn_budget = Entry(self.window)
        self.e_deep_sort_nn_budget.insert(5, parent.deep_sort_nn_budget)
        self.e_deep_sort_nn_budget.grid(row=25, column=1)
        Label(self.window, text="deep_sort_nn_budget:").grid(row=25, column=0)
        
        self.e_deep_sort_max_cosine_distance = Entry(self.window)
        self.e_deep_sort_max_cosine_distance.insert(5, parent.deep_sort_max_cosine_distance)
        self.e_deep_sort_max_cosine_distance.grid(row=26, column=1)
        Label(self.window, text="deep_sort_max_cosine_distance:").grid(row=26, column=0)
        
        self.e_deep_sort_nms_max_overlap = Entry(self.window)
        self.e_deep_sort_nms_max_overlap.insert(5, parent.deep_sort_nms_max_overlap)
        self.e_deep_sort_nms_max_overlap.grid(row=27, column=1)
        Label(self.window, text="deep_sort_nms_max_overlap:").grid(row=27, column=0)
        
        self.e_deep_sort_min_detection_height = Entry(self.window)
        self.e_deep_sort_min_detection_height.insert(5, parent.deep_sort_min_detection_height)
        self.e_deep_sort_min_detection_height.grid(row=28, column=1)
        Label(self.window, text="deep_sort_min_detection_height:").grid(row=28, column=0)
        
        self.e_deep_sort_per_process_gpu_mem_fraction = Entry(self.window)
        self.e_deep_sort_per_process_gpu_mem_fraction.insert(5, parent.deep_sort_per_process_gpu_mem_fraction)
        self.e_deep_sort_per_process_gpu_mem_fraction.grid(row=29, column=1)
        Label(self.window, text="deep_sort_per_process_gpu_mem_fraction:").grid(row=29, column=0)
        
        self.e_deep_sort_feature_model = Entry(self.window)
        self.e_deep_sort_feature_model.insert(5, parent.deep_sort_feature_model)
        self.e_deep_sort_feature_model.grid(row=30, column=1)
        Label(self.window, text="deep_sort_feature_model:").grid(row=30, column=0)
        
        btn4 = Button(self.window, text="change",
                      command=lambda *args: self.set_deep_sort_feature_model(parent))
        btn4.grid(row=31, column=1)
        
        btn1 = Button(self.window, text="Set Configs",
                      command=lambda *args: self.update_configs(parent))
        btn1.grid(row=33, column=0)
        
        parent.root.wait_window(self.window)
        
    def update_configs(self, parent):     
           
        parent.cam_id                                   = int(self.e_app_web_cam.get())
        parent.app_gpu                                  = int(self.e_app_gpu.get())
        parent.app_display_det_every_nth_frame          = int(self.e_app_display_det_every_nth_frame.get())
        parent.app_display_tracking_every_nth_frame     = int(self.e_app_display_tracking_every_nth_frame.get())
        parent.app_process_every_nth_frame              = int(self.e_app_process_every_nth_frame.get())
        parent.app_imutils_queue_size                   = int(self.e_app_imutils_queue_size.get())
        parent.app_save_det_result_path                 = str(self.e_app_save_det_result_path.get())
        parent.detectron_wts                            = str(self.e_detectron_wts.get())
        parent.detectron_cfg                            = str(self.e_detectron_cfg.get())
        parent.detectron_kp_thresh                      = float(self.e_detectron_kp_thresh.get())
        parent.detectron_nms_thresh                     = float(self.e_detectron_nms_thresh.get())
        parent.detectron_download_cache                 = str(self.e_detectron_download_cache.get())
        parent.detectron_detections_per_im              = int(self.e_detectron_detections_per_im.get())
        parent.detectron_proposal_limit                 = int(self.e_detectron_proposal_limit.get())
        parent.detectron_rpn_nms_thresh                 = float(self.e_detectron_rpn_nms_thresh.get())
        parent.detectron_score_thresh                   = float(self.e_detectron_score_thresh.get())
        parent.chainer_ms_thresh                        = float(self.e_chainer_ms_thresh.get())
        parent.app_save_tracking_result_path            = str(self.e_app_save_tracking_result_path.get())
        parent.deep_sort_min_confidence                 = float(self.e_deep_sort_min_confidence.get())
        parent.deep_sort_nn_budget                      = int(self.e_deep_sort_nn_budget.get())
        parent.deep_sort_max_cosine_distance            = float(self.e_deep_sort_max_cosine_distance.get())
        parent.deep_sort_nms_max_overlap                = float(self.e_deep_sort_nms_max_overlap.get())
        parent.deep_sort_min_detection_height           = int(self.e_deep_sort_min_detection_height.get())
        parent.deep_sort_per_process_gpu_mem_fraction   = float(self.e_deep_sort_per_process_gpu_mem_fraction.get())
        parent.deep_sort_feature_model                  = str(self.e_deep_sort_feature_model.get())
        
        if parent.opencv_thread is not None:
            parent.source_changed = True
                        
        parent.root.update()
        self.window.destroy()

        
    def set_save_det_result_path(self, parent):        
        options = {}
        options['title'] = "Choose detection output directory"

        directory = askdirectory(**options)

        if directory:
            self.e_app_save_det_result_path.insert(5, directory)
            
    def set_save_tracking_result_path(self, parent):        
        options = {}
        options['title'] = "Choose tracking output directory"

        directory = askdirectory(**options)

        if directory:
            self.e_app_save_tracking_result_path.insert(5, directory)
            
            
    def set_download_cache(self, parent):        
        options = {}
        options['title'] = "Choose tracking output directory"

        directory = askdirectory(**options)

        if directory:
            self.e_detectron_download_cache.insert(5, directory)
            
    def set_deep_sort_feature_model(self, parent):        
        options = {}
        options['title'] = "Choose deep_sort feature model file"
        options['filetypes'] = [('feature model', '.bp')]
        options['defaultextension'] = "bp"

        filename = askopenfilename(**options)

        if filename:
            self.e_deep_sort_feature_model.insert(5, filename)
            
    def set_detectron_wts(self, parent):        
        options = {}
        options['title'] = "Choose detectron wts file"
        options['filetypes'] = [('wts', '.wts')]
        options['defaultextension'] = "wts"

        filename = askopenfilename(**options)

        if filename:
            self.e_detectron_wts.insert(5, filename)
            
    def set_detectron_cfg(self, parent):        
        options = {}
        options['title'] = "Choose detectron cfg file"
        options['filetypes'] = [('cfg', '.cfg')]
        options['defaultextension'] = "cfg"

        filename = askopenfilename(**options)

        if filename:
            self.e_detectron_wts.insert(5, filename)