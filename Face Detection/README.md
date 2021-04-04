#Face Detection

##Purpose
This program was created to explore object detection and centroid extraction.

##Description
The program will execute by opening a video feed and detect all available faces.
The program can track unique faces and re-establish tracking of a existing face
as long as the face has not been undetected for 50 frames.

##Libraries
*imutils / imutils.video
*numpy
*argparse
*time
*cv2 (opencv-python)

##Execution
Once all required libraries have been installed, execute this script "faceDetection.py"
from the terminal/command prompt using python3. You may also optionally pass a
confidence value using '-c' argument followed by a value.
