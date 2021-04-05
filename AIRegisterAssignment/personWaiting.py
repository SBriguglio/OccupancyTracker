from CentroidTracker import CentroidTracker
from TrackableObject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import graph

def main():
    #This will parse the arguements for execution
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str,
        help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
        help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
        help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
        help="# of skip frames between detections")
    args = vars(ap.parse_args())

    #Initialize the list of class labels of trained models
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
        
    #Load the specialized model from disk
    print("LAUNCHING...")
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

    #If an input video has not been provided via --input then use webcam
    if not args.get("input", False):
        print("[LAUNCHING VIDEO STREAM...]")
        if VideoStream(src=1) is None:
            vs = VideoStream(src=1).start()
        else:
            vs = VideoStream(src=0).start()
        time.sleep(2.0)
    #if input was specified then load the video
    else:
        print("[LOADING VIDEO FILE...]")
        vs = cv2.VideoCapture(args["input"])

    writer = None
    #Initialize the frame dimensions
    W = None
    H = None

    #initiate the centroid tracker
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    #create a list of trackers
    trackers = []
    #Create a set of trackable objects
    trackableObjects = {}

    #counts the total number of frames processed
    totalFrames = 0
    #total number of objects that moved down
    totalDown = 0
    #total number of objects that moved up
    totalUp = 0
    #tracks total occupancy
    totalOccupancy = 0
    #start the frames per second throughput estimator
    fps = FPS().start()

    #loop through each frame in the video stream
    while True:
        #grab a frame from the video stream
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame

        #check for EOF of vidoe
        if args["input"] is not None and frame is None:
                break
                
        #resize the frame
        frame = imutils.resize(frame, width=500)
        #convert frame to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #if the frame dimensions are empty, set them
        if W is None or H is None:
                (H, W) = frame.shape[:2]
                
        #if output was specified then intialize it
        if args["output"] is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

        status = "Waiting"
        rects = []
        
        #check to see if we need more computations
        if totalFrames % args["skip_frames"] == 0:
                
                #set the status and initialize our new set of object trackers
                status = "Detecting"
                trackers = []
                #conver to blob and process
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
                net.setInput(blob)
                #obtain detectiojs
                detections = net.forward()
                
                #Loop through detections
                for i in np.arange(0, detections.shape[2]):
                    #obtain confidence
                    confidence = detections[0, 0, i, 2]
                    #remove weak detections
                    if confidence > args["confidence"]:
                        #extract the index of the class label
                        idx = int(detections[0, 0, i, 1])
                        #if the class label is not a person, ignore it
                        if CLASSES[idx] != "person":
                            continue
                        #Compute the coordinates of the boundary box
                        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        (startX, startY, endX, endY) = box.astype("int")
                        
                        #Construct a boundary box and dlib tracker
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        tracker.start_track(rgb, rect)
                        #add it to the list of trackers
                        trackers.append(tracker)
                        
        else:
            #loop through trackers
            for tracker in trackers:
                    status = "Tracking"
                    
                    #update the tracker
                    tracker.update(rgb)
                    #obtain new positions
                    pos = tracker.get_position()
                    
                    #Extract the obtained positions
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())
                    
                    #add the coordinates to the rect list
                    rects.append((startX, startY, endX, endY))

        #Creates the threshold line
        #(img, (point 1), (point 2), color, thicknesspx)
        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
        objects = ct.update(rects)

        #loop through tracked objects
        for (objectID, centroid) in objects.items():
            #check if the object is a currently existing object
            to = trackableObjects.get(objectID, None)
            #if it does not exist, create it
            if to is None:
                to = TrackableObject(objectID, centroid)
            #if it does exist then use it
            else:
                #check which direction the object is moving
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)
                
                #see if the object has been counted or not
                if not to.counted:
                    #If the direction is negative, object is moving up
                    #if centroid is above line
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        #updates the total occupancy
                        if totalOccupancy > 0:
                            totalOccupancy -= 1
                        to.counted = True
                    #If the directio is positive, object is moving down
                    #If centroid is below line
                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        #update the total occupancy but dont go below 0
                        totalOccupancy += 1
                        to.counted = True
            # store the trackable object in our dictionary
            trackableObjects[objectID] = to
            
            #print the id
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #print the centroid dot
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        
        #information for the frame
        info = [
            ("Current Occupancy", totalOccupancy),
            ("Status", status),
        ]
        
        #print the inormation
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        #write to disk if specified
        if writer is not None:
            writer.write(frame)
            
        #show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        #press 'q' key to exit
        if key == ord("q"):
            break
        
        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1
        fps.update()

    fps.stop()
    #Print fps information
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    #check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()

    #if we are not using a video file, stop the camera video stream
    if not args.get("input", False):
        vs.stop()

    #otherwise, release the video file pointer
    else:
        vs.release()
        
    # close any open windows
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
