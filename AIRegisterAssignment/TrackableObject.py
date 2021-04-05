class TrackableObject:
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        #previous and current centroids
        self.centroids = [centroid]
        #Tracks if the object has been counted, False = not counted, True = counted
        self.counted = False
