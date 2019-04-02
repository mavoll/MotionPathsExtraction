from collections import deque
import cv2
import imutils
from threading import Thread

import numpy as np
import sys
from tkinter import Button
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
from tkinter import Label
from tkinter import messagebox
from tkinter import Tk
from tkinter import ttk

class App(object):
  
    def __init__(self):
                    
        self.alpha = 0.5
        self.rotate_grade = 0
        self.scale_width = 100
        self.move_x = 0
        self.move_y = 0    
      
        self.image = None
        self.ref_image = None
        self.clone = None
      
        self.scale_factor = 0.7
        self.root = Tk()
        self.opencv_thread = None
        
        self.refPt = []
        self.inter_line_counter = 0
        
    def run(self):

        self.root.wm_title("Select actions:")
        self.root.resizable(width=False, height=False)
        self.root.geometry('{}x{}'.format(250, 525))
        self.root.attributes("-topmost", True)
        
        labelTop = Label(self.root,
                         text="Choose image scale factor\n and press '1. Select an image' ")
        labelTop.pack(side="top", padx="10", pady="5")
        
        comboExample = ttk.Combobox(self.root,
                                    values=[
                                        0.5,
                                        0.6,
                                        0.7,
                                        0.8,
                                        0.9,
                                        1.0])

        comboExample.current(2)
        comboExample.state(['readonly'])
        comboExample.bind("<<ComboboxSelected>>", self.set_scale_factor)
        comboExample.pack(side="top", padx="10", pady="5")
        
          
        btn1 = Button(self.root, text="1. Select images \n(perspective image and \nreference map image)", command=self.open_image)
        btn1.pack(side="top", fill="both", expand="yes", padx="10", pady="5")
       
        labelTop = Label(self.root,
                         text="2. Set Points \n (click on the image)")
        labelTop.pack(side="top", padx="10", pady="5")
        
        btn8 = Button(self.root, text="3. Do TPS transformation \n and warping", command=self.do_tps_trans_and_warp)
        btn8.pack(side="top", fill="both", expand="yes", padx="10", pady="5")
        
        btn8 = Button(self.root, text="4. Save transformation matrix", command=self.save_image)
        btn8.pack(side="top", fill="both", expand="yes", padx="10", pady="5")
        
        btn8 = Button(self.root, text="5. Open tracking result \n files to transform", command=self.save_image)
        btn8.pack(side="top", fill="both", expand="yes", padx="10", pady="5")
        
        btn8 = Button(self.root, text="6. Transform tracking result \n ( (x,y) to (lat, long) ) \n and save csv", command=self.save_image)
        btn8.pack(side="top", fill="both", expand="yes", padx="10", pady="5")
                
        self.root.protocol("WM_DELETE_WINDOW", App.on_closing)
        self.root.mainloop()

        cv2.destroyAllWindows()
        sys.exit()
        
    def set_scale_factor(self, event=None):

        if event is not None:
            self.scale_factor = float(event.widget.get())

    def open_image(self):

        options = {}
        options['filetypes'] = [('Image file', '.jpg'), ('Image file', '.jpeg')]
        options['defaultextension'] = "jpg"
        options['title'] = "Choose image"

        filename = askopenfilename(**options)

        if filename:              
            self.image = cv2.imread(filename)
            h = int(self.image.shape[0] * self.scale_factor)
            w = int(self.image.shape[1] * self.scale_factor)
            self.image = cv2.resize(self.image, (w, h))
            self.clone = self.image.copy()
            
            if self.open_ref_image():
            
                self.pixel_points = []
                self.fix_points = []
                self.num_selected_points = 0
                self.refPt = []
                self.inter_line_counter = 0
                
                self.alpha = 0.5
                self.rotate_grade = 0
                self.scale_width = 100
                self.move_x = 0
                self.move_y = 0
                
                if self.opencv_thread is None:
                    self.opencv_thread = Thread(target=self.show_image)
                    self.opencv_thread.daemon = True
                    self.opencv_thread.start()

    def open_ref_image(self):

        options = {}
        options['filetypes'] = [('Image file', '.jpg'), ('Image file', '.jpeg')]
        options['defaultextension'] = "jpg"
        options['title'] = "Choose ref image"

        filename = askopenfilename(**options)

        if filename:
             
            self.ref_image = cv2.imread(filename, -1)
            h = int(self.ref_image.shape[0] * self.scale_factor)
            w = int(self.ref_image.shape[1] * self.scale_factor)
            self.ref_image = cv2.resize(self.ref_image, (w, h))
            
            return True
        
        else:
            return False

    def show_image(self):

        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('image', 0, 0)
        cv2.setMouseCallback('image', self.set_points_callback)
                    
        self.start_processing()

        cv2.destroyAllWindows()
        self.opencv_thread = None

    def start_processing(self):
        
        if self.image is not None:
                        
            try:                    
                trackbar_name = 'Alpha x %d' % 100
                cv2.createTrackbar(trackbar_name, 'image' , int(self.alpha * 100), 100, self.on_trackbar_alpha)
                
                trackbar_name2 = 'Width x %d' % 100
                cv2.createTrackbar(trackbar_name2, 'image' , self.scale_width, 500, self.on_trackbar_size)
                
                trackbar_name3 = 'Rotate x %d' % 100
                cv2.createTrackbar(trackbar_name3, 'image' , self.rotate_grade, 360, self.on_trackbar_rotate)

                trackbar_name4 = 'Move x %d' % 100
                cv2.createTrackbar(trackbar_name4, 'image' , self.move_x, 2000, self.on_trackbar_move_x)
                
                trackbar_name5 = 'Move y %d' % 100
                cv2.createTrackbar(trackbar_name5, 'image' , self.move_y, 1000, self.on_trackbar_move_y)                                   
                
                self.draw()
                
                cv2.waitKey(0)
                

            except Exception:

                #continue
                e = sys.exc_info()[0]
                messagebox.showinfo("Error processing file", e)
                raise
    
    def set_points_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.refPt:
                self.refPt.append((x, y))
            else:
                self.refPt = [(x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            if self.refPt[-1] != (x, y):
                self.refPt.append((x, y))
                image = self.put_points_and_line_on_image(self.image,
                    self.refPt[self.inter_line_counter], self.refPt[self.inter_line_counter + 1])
                cv2.imshow("image", image)
                self.pixel_points.append(self.refPt[self.inter_line_counter])
                self.fix_points.append(self.refPt[self.inter_line_counter + 1])
                self.inter_line_counter += 2
                self.num_selected_points += 1
            else:
                self.refPt.pop()

    def put_points_and_line_on_image(self, img, point1, point2):

        cv2.line(img, point1, point2, (0, 255, 0), 2)
        cv2.putText(img,
                    "p1" + str(point1),
                    (point1[0] - 60, point1[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 255),
                    1)
        cv2.putText(img,
                    "p2" + str(point2),
                    (point2[0] - 60, point2[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 255),
                    1)
        
        return img
                        
    def do_tps_trans_and_warp(self):        
        
        if len(self.pixel_points) == len(self.fix_points) and len(self.pixel_points) >= 3:   
            
            matches = list()
            for i in range(0, len(self.pixel_points)):            
                matches.append(cv2.DMatch(i,i,0))
            
            pixel_points = np.float32(self.pixel_points).reshape(1, -1, 2)
            fix_points = np.float32(self.fix_points).reshape(1, -1, 2)
            
            tps_transformer = cv2.createThinPlateSplineShapeTransformer(0)
                                        
            tps_transformer.estimateTransformation(pixel_points, fix_points, matches)        
            ret, output = tps_transformer.applyTransformation(pixel_points)
            image = tps_transformer.warpImage(self.image) #, cv2.INTER_CUBIC, cv2.BACK_WARP)
            
            cv2.imshow("image", image)
        
        else:
            messagebox.showinfo("Not enough points!", "Please set at least 3 point pairs!")
        
                       
    def save_image(self):

        if self.image is not None:

            options = {}
            options['filetypes'] = [('Image file', '.jpg'), ('Image file', '.jpeg')]
            options['defaultextension'] = "jpg"
            options['initialfile'] = "counting_result_image.jpg"
            options['title'] = "Where to save?"

            filename = asksaveasfilename(**options)

            if filename:

                outputjpgfile = filename

                try:

                    cv2.imwrite(outputjpgfile, self.image)

                except Exception:

                    e = sys.exc_info()[0]
                    messagebox.showinfo("Error saving file", e)
                    raise

        else:
            messagebox.showinfo("No image selected", "Please select an image first!")

    def draw(self):
        
        w = int(round(self.ref_image.shape[1] * (self.scale_width / 100), 0))
        ref_image = imutils.resize(self.ref_image, w)
        ref_image = imutils.rotate(ref_image, angle=int(self.rotate_grade))                    
        image = App.overlay_transparent(self.clone, ref_image, self.move_x, self.move_y, alpha=self.alpha)        
                            
        cv2.imshow("image", image)

    def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None, alpha=0.1):
        
        output = background_img.copy()
        overlay = background_img.copy()
                
        ho, wo, _ = output.shape
        h, w, _ = img_to_overlay_t.shape
                
        output[y:y+h, x:x+w] = img_to_overlay_t[0:h, 0:w]
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
            
        return output
    
     
    def on_trackbar_alpha(self, val):
        
        self.alpha = round((val / 100), 1)
        self.draw()
    
    def on_trackbar_size(self, val):
        
        self.scale_width = val
        self.draw()
        
    def on_trackbar_rotate(self, val):
        
        self.rotate_grade = val
        self.draw()
    
    def on_trackbar_move_x(self, val):
        
        self.move_x = val
        self.draw()
    
    def on_trackbar_move_y(self, val):
        
        self.move_y = val
        self.draw()
        
    def on_closing():

        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            sys.exit()

if __name__ == '__main__':
    #    import sys

    App().run()
