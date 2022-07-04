from enum import Enum
import cv2
import numpy as np
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import time
import datetime

import tkinter as tk
from tkinter import *
import os
window=Tk()
MODEL_PATH = "hand_model_gray.hdf5"

face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

fe={
    '':0,
    'Angry':0,
    'Disgust':1,
    'Neutral':3,
    'Happy':4,
    'Surprise':5,
    'Sad':2,
    'Fear':3,
}
hg={
    '':0,
    'fist':0,
    'five':5,
    'point':1,
    'swing':3
}
class GestureRecognizer:
    """
    Class for tracking the hand and labeling the gesture.
    """

    # Mapping between the output of our model and a textual representation
    CLASSES = {
        0: 'fist',
        1: 'five',
        2: 'point',
        3: 'swing'
    }

    positions = {
        'hand_pose': (15, 40), # hand pose text
        'fps': (15, 20), # fps counter
        'null_pos': (200, 200) # used as null point for mouse control
    }

    def __init__(self, capture=0, model_path=MODEL_PATH):
        self.bg = None
        self.frame = None # current frame
        self.tracker = cv2.TrackerKCF_create()
        self.kernel = np.ones((3, 3), np.uint8)
        self.recognizer = load_model(model_path)
        self.is_tracking = False
        self.hand_bbox = (116, 116, 170, 170)

        # Begin capturing video
        self.video = cv2.VideoCapture(capture)
        width  = self.video.get(3)  # float `width`
        height = self.video.get(4)  # float `height`
        # print('width, height:', width, height)
        if not self.video.isOpened():
            print("Could not open video")

    # def __del__(self):
    #     cv2.destroyAllWindows()
    #     self.video.release()

    # Helper function for applying a mask to an array
    def mask_array(self, array, imask):
        if array.shape[:2] != imask.shape:
            raise Exception("Shapes of input and imask are incompatible")
        output = np.zeros_like(array, dtype=np.uint8)
        for i, row in enumerate(imask):
            output[i, row] = array[i, row]
        return output

    def extract_foreground(self):
        # Find the absolute difference between the background frame and current frame
        diff = cv2.absdiff(self.bg, self.frame)
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Threshold the mask
        th, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

        # Opening, closing and dilation
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel)
        img_dilation = cv2.dilate(closing, self.kernel, iterations=2)

        # Get mask indexes
        imask = img_dilation > 0

        # Get foreground from mask
        foreground = self.mask_array(self.frame, imask)

        return foreground, img_dilation

    def run(self):
        # Capture, process, display loop 
        f=0 
        g=""
        e=""
        s=45
        arrm=[]
        while True:
            # Read a new frame
            ok, self.frame = self.video.read()
            display = self.frame.copy() # Frame we'll do all the graphical drawing to
            data_display = np.zeros_like(display, dtype=np.uint8) # Black screen to display data
            if not ok:
                break

            # Start FPS timer
            timer = cv2.getTickCount()

            if self.bg is None:
                cv2.putText(display,
                            "Press 'r' to reset the foreground extraction.", 
                            self.positions['hand_pose'], 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.75, (0, 127, 64), 2)
                # cv2.imshow("display", display)

                k = cv2.waitKey(1) & 0xff
                if k == 27: break # ESC pressed
                elif k == 114 or k == 108:
                    f=1 
                    # r pressed
                    self.bg = self.frame.copy()
                    self.hand_bbox = (116, 116, 170, 170)
                    self.is_tracking = False
            else:
                # Extract the foreground
                foreground, mask = self.extract_foreground()
                foreground_display = foreground.copy()

                # Get hand from mask using the bounding box
                hand_crop = mask[int(self.hand_bbox[1]):int(self.hand_bbox[1]+self.hand_bbox[3]), 
                                 int(self.hand_bbox[0]):int(self.hand_bbox[0]+self.hand_bbox[2])]

                # Update tracker
                if self.is_tracking:
                    tracking, self.hand_bbox = self.tracker.update(foreground)

                try:
                    # Resize cropped hand and make prediction on gesture
                    hand_crop_resized = np.expand_dims(cv2.resize(hand_crop, (54, 54)), axis=0).reshape((1, 54, 54, 1))
                    prediction = self.recognizer.predict(hand_crop_resized)
                    predi = prediction[0].argmax() # Get the index of the greatest confidence
                    gesture = self.CLASSES[predi]
                    
                    for i, pred in enumerate(prediction[0]):
                        # Draw confidence bar for each gesture
                        barx = self.positions['hand_pose'][0]
                        bary = 60 + i*60
                        bar_height = 20
                        bar_length = int(400 * pred) + barx # calculate length of confidence bar
                        
                        # Make the most confidence prediction green
                        if i == predi:
                            colour = (0, 255, 0)
                        else:
                            colour = (0, 0, 255)
                        
                        cv2.putText(data_display, 
                                    "{}: {}".format(self.CLASSES[i], pred), 
                                    (self.positions['hand_pose'][0], 30 + i*60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.75, (255, 255, 255), 2)
                        cv2.rectangle(data_display, 
                                      (barx, bary), 
                                      (bar_length, 
                                      bary - bar_height), 
                                      colour, 
                                      -1, 1)
                    g=format(gesture)
                    cv2.putText(display, 
                                "hand pose: {}".format(gesture), 
                                self.positions['hand_pose'], 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.75, (0, 0, 255), 2)
                    # cv2.putText(foreground_display, 
                    #             "hand pose: {}".format(gesture), 
                    #             self.positions['hand_pose'], 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 
                    #             0.75, (0, 0, 255), 2)
                except Exception as ex:
                    cv2.putText(display, 
                                "hand pose: error", 
                                self.positions['hand_pose'], 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.75, (0, 0, 255), 2)
                    # cv2.putText(foreground_display, 
                    #             "hand pose: error", 
                    #             self.positions['hand_pose'], 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 
                    #             0.75, (0, 0, 255), 2)    
                
                # Draw bounding box
                p1 = (int(self.hand_bbox[0]), int(self.hand_bbox[1]))
                p2 = (int(self.hand_bbox[0] + self.hand_bbox[2]), int(self.hand_bbox[1] + self.hand_bbox[3]))
                # cv2.rectangle(foreground_display, p1, p2, (255, 0, 0), 2, 1)
                cv2.rectangle(display, p1, p2, (255, 0, 0), 2, 1)

                # Calculate difference in hand position
                hand_pos = ((p1[0] + p2[0])//2, (p1[1] + p2[1])//2)
                mouse_change = ((p1[0] + p2[0])//2 - self.positions['null_pos'][0], self.positions['null_pos'][0] - (p1[1] + p2[1])//2)

                # Draw hand moved difference
                cv2.circle(display, self.positions['null_pos'], 5, (0,0,255), -1)
                cv2.circle(display, hand_pos, 5, (0,255,0), -1)
                cv2.line(display, self.positions['null_pos'], hand_pos, (255,0,0),5)

                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

                # Display FPS on frame
                # cv2.putText(foreground_display, "FPS : " + str(int(fps)), self.positions['fps'], cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 170, 50), 2)
                cv2.putText(display, "FPS : " + str(int(fps)), self.positions['fps'], cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 170, 50), 2)
                # print(self.positions['fps'])r
                # Display pause command text
                cv2.putText(display, 
                            "hold 'r' to recalibrate until the screen is black", 
                            (15, 400), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 0, 255), 2)
                cv2.putText(display, 
                            "to recalibrate", 
                            (15, 420), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.65, (0, 0, 255), 2)
                cv2.putText(display, 
                            "press 'p' to return to paused state", 
                            (15, 440), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.65, (0, 0, 255), 2)
                cv2.putText(display, 
                            "press 'ESC' to exit", 
                            (15, 465), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.65, (0, 0, 255), 2)

                # Display foreground_display
                # cv2.imshow("foreground_display", foreground_display)
                # cv2.imshow("display", display)
                cv2.imshow("hand_crop", hand_crop)
                # Display result
                # cv2.imshow("data", data_display)

                k = cv2.waitKey(1) & 0xff
                if k == 27: break # ESC pressed
                elif k == 114 or k == 108: # r pressed
                    self.bg = self.frame.copy()
                    self.hand_bbox = (116, 116, 170, 170)
                    self.is_tracking = False
                elif k == 116: # t pressed
                    # Initialize tracker with first frame and bounding box
                    self.is_tracking = True
                    self.tracker = cv2.TrackerKCF_create()
                    self.tracker.init(self.frame, self.hand_bbox)
                elif k == 112: # p pressed
                    # Reset to paused state
                    self.is_tracking = False
                    self.bg = None
                    cv2.destroyAllWindows()
                elif k != 255: print(k)

            # _, frame = cap.read()
            labels = []
            gray = cv2.cvtColor(display,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)
            if f==1:
                for (x,y,w,h) in faces:
                    if w>=110:
                        cv2.rectangle(display,(x,y),(x+w,y+h),(0,255,255),2)
                        roi_gray = gray[y:y+h,x:x+w]
                        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



                        if np.sum([roi_gray])!=0:
                            roi = roi_gray.astype('float')/255.0
                            roi = img_to_array(roi)
                            roi = np.expand_dims(roi,axis=0)

                            prediction = classifier.predict(roi)[0]
                            label=emotion_labels[prediction.argmax()]
                            label_position = (x,y)
                            e=label
                            cv2.putText(display,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                        else:
                            cv2.putText(display,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                z=(hg[g]+fe[e])/2
                arrm.append(z)
                cv2.putText(display, 
                            "Stars: "+ str(z), 
                            (15,65), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.75, (0, 0, 255), 2)
                s=s-1
                cv2.putText(display,'time left-'+str(s),(15,95),cv2.FONT_HERSHEY_SIMPLEX,0.9,(50,170,50),2)
                if s==0:
                    cv2.destroyAllWindows()
                    break
            cv2.imshow('Display',display)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break  
        if len(arrm)>0:
            arrm.sort()      
            z1=len(arrm)//2
            window1=Tk()
            z2=Label(window1, text="Rating: "+str(arrm[z1]), font=('Copperplate Gothic Bold',20), foreground="gold").place(x=220,y=180)
            window1.title("Results")
            window1.geometry("640x480")
            window1.tk.call('wm', 'iconphoto', window1._w, tk.PhotoImage(file=r'result.png'))
            window1.mainloop()
            # window.print(arrm[z1])

def open_prompt():
    window.destroy()
    recognizer = GestureRecognizer(0)
    recognizer.run()

z3=Label(window, text='Click Here to start!!', font=('Copperplate Gothic Bold',20)).place(x=170,y=180)
tk.Button(window, text= "Start", command= open_prompt,font=('Copperplate Gothic Bold',20), background="cyan").place(x=270,y=240)
# btn=Button(window, text="Start", fg='green',align="center")
# btn.place(x=80, y=100)
window.title("Start")
window.geometry("640x480")
window.tk.call('wm', 'iconphoto', window._w, tk.PhotoImage(file=r'start.png'))
window.mainloop()
# if __name__ == "__main__":
#     recognizer = GestureRecognizer(0)
#     recognizer.run()