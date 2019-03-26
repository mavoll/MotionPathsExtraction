from tkinter import Button
from tkinter import Checkbutton
from tkinter import Label
from tkinter import font
from tkinter import ttk
from tkinter import W

class MainWindow(object):
    
    def __init__(self, parent):   
        
        self.root = parent.root
        self.root.protocol("WM_DELETE_WINDOW", parent.on_closing)
        self.root.wm_title("Select actions:")
        self.root.resizable(width=False, height=False)
        self.root.geometry('{}x{}'.format(250, 600))
        self.root.attributes("-topmost", True)
        default_font = font.Font(family=font.nametofont("TkDefaultFont").cget("family"), size=10)
        self.root.option_add("*Font", default_font)
        
        btn_config_load = Button(self.root, text="Load config file", command=parent.load_config_file)
        btn_config_load.pack(side="top", fill="both", expand="no", padx="10", pady="5")
        
        btn_config_save = Button(self.root, text="Save config file", command=parent.save_config_file)
        btn_config_save.pack(side="top", fill="both", expand="no", padx="10", pady="5")
        
        options = Button(self.root, text="Options", command=parent.ask_for_options)
        options.pack(side="top", fill="both", expand="no", padx="10", pady="5")
                
        labelDet = Label(self.root,
                         text="1. Choose detector:")
        labelDet.pack(side="top", anchor=W, padx="10", pady="5")        
        
        labelComboDetector = Label(self.root,
                         text="detector_to_use:")
        labelComboDetector.pack(side="top", padx="10", pady="5")
        
        comboDetector = ttk.Combobox(self.root,
                                    values=parent.detectors,
                                    textvariable=parent.v_detector_to_use)
        comboDetector.state(['readonly'])
        comboDetector.bind("<<ComboboxSelected>>", parent.set_detector_to_use)
        comboDetector.pack(side="top", padx="10", pady="5")
        
        labelDetectronModel = Label(self.root,
                         text="used detectron model:")
        labelDetectronModel.pack(side="top", padx="10", pady="5")
        comboDetectronModel = ttk.Combobox(self.root,
                                    values=parent.detectron_models,
                                    textvariable=parent.v_detectron_model)
        comboDetectronModel.state(['readonly'])
        comboDetectronModel.bind("<<ComboboxSelected>>", parent.set_detectron_model)
        comboDetectronModel.pack(side="top", padx="10", pady="5")
        
        labelChainercvModel = Label(self.root,
                         text="used chainercv model:")
        labelChainercvModel.pack(side="top", padx="10", pady="5")
        comboChainercvModel = ttk.Combobox(self.root,
                                    values=parent.chainercv_models,
                                    textvariable=parent.v_chainer_model)
        comboChainercvModel.state(['readonly'])
        comboChainercvModel.bind("<<ComboboxSelected>>", parent.set_chainercv_model)
        comboChainercvModel.pack(side="top", padx="10", pady="5")
        
        parent.v_1.set(parent.app_display_det_result_img)
        Checkbutton(self.root,
                    text="show detection results",
                    padx=20,
                    variable=parent.v_1,
                    command=parent.set_display_det_result_img).pack(side="top", anchor=W, padx="5", pady="5")
        
        parent.v_2.set(parent.app_save_det_result_img)
        Checkbutton(self.root,
                    text="save detection results",
                    padx=20,
                    variable=parent.v_2,
                    command=parent.set_save_det_result_img).pack(side="top", anchor=W, padx="5", pady="5")


        labelTracker = Label(self.root,
                         text="2. Choose tracker:")
        labelTracker.pack(side="top", anchor=W, padx="10", pady="5")
        
        parent.v_5.set(parent.app_do_tracking)
        Checkbutton(self.root,
                    text="do tracking?",
                    padx=20,
                    variable=parent.v_5,
                    command=parent.set_do_tracking).pack(side="top", anchor=W, padx="5", pady="5")
        
        parent.v_detector_to_use.set(parent.app_detector_to_use)
        labelComboTracker = Label(self.root,
                         text="tracker_to_use:")
        labelComboTracker.pack(side="top", padx="10", pady="5")
        comboTracker = ttk.Combobox(self.root,
                                    values=parent.trackers,
                                    textvariable=parent.v_tracker_to_use)
        comboTracker.state(['readonly'])
        comboTracker.bind("<<ComboboxSelected>>", parent.set_tracker_to_use)
        comboTracker.pack(side="top", padx="10", pady="5")
                
        parent.v_4.set(parent.app_display_tracking_result_img)
        Checkbutton(self.root,
                    text="show tracking results",
                    padx=20,
                    variable=parent.v_4,
                    command=parent.set_display_tracking_result_img).pack(side="top", anchor=W, padx="5", pady="5")        
        
        parent.v_3.set(parent.app_save_tracking_result_img)
        Checkbutton(self.root,
                    text="save tracking results",
                    padx=20,
                    variable=parent.v_3,
                    command=parent.set_save_tracking_result_img).pack(side="top", anchor=W, padx="5", pady="5")


        labelSource = Label(self.root,
                         text="3. Start processing:")
        labelSource.pack(side="top", anchor=W, padx="10", pady="5")
        
        btn1 = Button(self.root, text="using video file", command=parent.open_video)
        btn1.pack(side="top", fill="both", expand="no", padx="10", pady="5")

        btn2 = Button(self.root, text="using webcam", command=parent.open_webcam)
        btn2.pack(side="top", fill="both", expand="no", padx="10", pady="5")