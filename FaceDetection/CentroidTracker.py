from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    #maxDisappeared states the number of consecutive frames the object has to be lost to be deleted
    def __init__(self, maxDisappeared=50):
        #counter used to assign new ID's
        self.nextObjectID = 0
        #Object dictionary that store the objects: objectID = key, cnetroid = value
        self.objects = OrderedDict()
        #Tracks how many consecutive frames have passed without the object
        self.disappeared = OrderedDict()
        #Store the number of consecutive frames the object has to be lost to be deleted
        self.maxDisappeared = maxDisappeared

    #Reccords a new object(centroid)
    def register(self, centroid):
        #store the centroid
        self.objects[self.nextObjectID] = centroid
        #set this objects consecutive frames to 0
        self.disappeared[self.nextObjectID] = 0
        #incremenet the objectID counter
        self.nextObjectID += 1

    #removes an object that has "disappeared"
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    #Updates the centroid of existing objects
    def update(self, rects):
        #rects = list of boundary boxes
        #structure: (startX, startY, endX, endY)
        if len(rects) == 0:
            #if there are no detections
            #Mark all existing objects that have dissappeared
            for objectID in list(self.disappeared.keys()):
                #Here is where we mark that, this will keep track of the amount of frames the objects has been lost for
                self.disappeared[objectID] += 1
                #Once the objects has been missing for more than 50 Frames, delete the object
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        #Iitialize a numpy array to store the centroid of each rect above
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        #loop through the boundary boxes (rects) and calculate their centorids
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            #Calcualte the center of the x coordiantes
            cX = int((startX + endX) / 2.0)
            #Calculate the center of the y coordinates
            cY = int((startY + endY) / 2.0)
            #Store the centroid coordinates
            inputCentroids[i] = (cX, cY)

        #If there are no objects, register the new object
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        #Update existing object
        else:
            #Obtain object id's
            objectIDs = list(self.objects.keys())
            #obtain object centroids
            objectCentroids = list(self.objects.values())
            #Calcualte the distance between the obejct centroids and the new inputs
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            '''
            (1) Find the smallest value in each row
            (2) sort the row indexes based on their minimum values at the front (ascending)
            '''
            rows = D.min(axis=1).argsort()
            '''
            (1) Find the smallest value in each column
            (2) sort the row indexes based on their minimum values at the front (ascending)
            '''
            cols = D.argmin(axis=1)[rows]
            #Keep track of the of the rows and cols examined
            usedRows = set()
            usedCols = set()
            
            #Loops over combination of the (row,column) index
            for (row, col) in zip(rows, cols):
                #If the column or row has been examined, skip
                if row in usedRows or col in usedCols:
                    continue
                    
                #Grab the object in the row
                objectID = objectIDs[row]
                #set its new centroid
                self.objects[objectID] = inputCentroids[col]
                #Reset its dissappeared frames
                self.disappeared[objectID] = 0
                #indicate that we have visited each of the rows and columns
                usedRows.add(row)
                usedCols.add(col)
                
            #Compute row index we havent visited
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            #Compute column index we havent visited
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            #If the number of centroids is greater than the current number of input centroids 
            if D.shape[0] >= D.shape[1]:
                #loop over the unused row index
                for row in unusedRows:
                    #get the object ids
                    objectID = objectIDs[row]
                    #increment the dissappearing counter
                    self.disappeared[objectID] += 1
                    
                    #if the number of "lost" frames exceeds 50
                    if self.disappeared[objectID] > self.maxDisappeared:
                        #remove the object
                        self.deregister(objectID)
            #If the number of inpout centroids is greater than the existing centroids
            else:
                #input them
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects
