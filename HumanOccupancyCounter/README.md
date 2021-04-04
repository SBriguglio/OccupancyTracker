# Human Occupancy Counter

## Purpose
This program was created to count the occupancy of an establishment by monitoring
the entrance and exit where the count begins.

## Description
The program will execute by opening a video feed. The program can track unique
bodies across the frame. Once the object(person) crosses the designated threshold
they either increment the counter or decrement the count depending on the direction
of crossing.

## Libraries
* imutils / imutils.video
* numpy
* argparse
* time
* cv2 (opencv-python)
* dlib

## Execution
Once all required libraries have been installed, execute this script "occupancyDetection.py"
from the terminal/command prompt using python3. You may also optionally pass in
a few arguments:
*-i : you can supply a file path for an input video file (Will use webcam by default)
*-o : you can supply a path for an output video file
*-c : you can supply a confidence value
*-s : you can provide a value for how many frames to skip
