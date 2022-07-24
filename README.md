# Automated-Rating-System-Using-Hand-gestures-and-Face-recoginition

INTRODUCTION

It was a team project, my contribution was that I combined both models of hand gesture and face recognition using opencv for getting inputs of both hand and face simultaneously and tkinter for creating the UI. Main Window runs for window of 45 secs and median of average ratings has been displayed.

WORKING

There are no existing systems that provide rating points based on recognizing facial expressions and hand gestures by taking in the video feed. This project does exactly that. Both things are taken into account simultaneously. That is, the user gives facial expressions and hand gestures together in a single feed. Both of these are read by the system and averaged out into one single rating. 

When the software starts a window comes up which is the UI. A calibration instruction is provided which asks the user to hit button ‘R’ for calibration of the background for hand gesture recognition. Once that happens successfully, another window opens up. Here, the video feed is taken in as input. Bounding boxes appear on the screen and a calibration button is also provided for gesture recognition setup. The user has to give expressions and show gestures and these boxes will capture the same and display the instantaneous individual output as to what these readings are actually interpreted. The user can see how with changing expressions and changing gestures, the individual outputs are changing. This accounts for transparency in the process. UI has been created using Tkinter of python whereas the backend has been built using python and OpenCV.

The window for taking input stays on for 45 seconds. This is because this is a feasible time for a user to hold the expression and gesture. The median of all the ratings is taken into account and the output is displayed in another window. The output happens to be the averaged-out number from both the readings. The UI that has been provided is interactive and engaging for the customers so that they don’t get bored while giving ratings.


OUTPUT

Start Screen

![image](https://user-images.githubusercontent.com/69521280/180649955-2f204141-492b-4186-8d78-965ec2c0ea06.png)

Capturing Face expression and hand gesture window

![image](https://user-images.githubusercontent.com/69521280/180650012-4685db5e-cb3d-4587-bac2-07ae630aad96.png)

Results Window

![image](https://user-images.githubusercontent.com/69521280/180650046-1d917f87-c0b0-4720-aeed-8623dfb9a6b8.png)
