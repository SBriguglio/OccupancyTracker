from CentroidTracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

def main():
    #used to set a confidence contstraint, has a defualt 0.5 value
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    #initialize the centroid tracker
    ct = CentroidTracker()
    #initialize the frame size
    (H, W) = (None, None)

    #Load model
    net = cv2.dnn.readNetFromCaffe("deploy.prototext", "res10_300x300_ssd_iter_140000.caffemodel")

    print("[LAUNCHING VIDEO STREAM...]")
    #start the video stream
    if VideoStream(src=1) is None:
        vs = VideoStream(src=1).start()
    else:
        vs = VideoStream(src=0).start()
    time.sleep(2.0)

    #loop over the frames from the video stream
    while True:
        #read the next frame from video stream
        frame = vs.read()
        #resize the frame
        frame = imutils.resize(frame, width=800)
        #if the frame dimensions don't exist, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        #construct a blob from the frame
        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),(104.0, 177.0, 123.0))
        #Pass through
        net.setInput(blob)
        #obtain detections
        detections = net.forward()
        rects = []
        
        #loop over hte detections
        for i in range(0, detections.shape[2]):
            #Filter out weak detections based on the confidence threshold
            if detections[0, 0, i, 2] > args["confidence"]:
                #Compute the bounding box coordinares
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                #update the bounding box list
                rects.append(box.astype("int"))
                #Draw the counding box
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY),(255, 0, 0), 2)

        #Update the centroids
        objects = ct.update(rects)
        
        #loop over the tracked objects
        for (objectID, centroid) in objects.items():
            #Print the ID
            text = "PERSON {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #Print a dot for the centroid
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)

        #show the output frame
        cv2.imshow("Face Detection Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        #press 'q' key to exit
        if key == ord("q"):
                break

    cv2.destroyAllWindows()
    vs.stop()
    
if __name__ == '__main__':
    main()

    
