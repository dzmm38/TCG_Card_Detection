##### --------------------------------------------------------------- #####
##### --------------------------- Imports --------------------------- #####
##### --------------------------------------------------------------- #####
import cv2
import time
import threading
import numpy as np
from webcam import WebcamStream
from imageProcessor import ImageProcessor
from cardDataLoader import CardDataLoader
from imageClassifierFeatureDetection import ImageClassifierFeatureDetection

##### --------------------------------------------------------------- #####
##### -------------------------- Parameter -------------------------- #####
##### --------------------------------------------------------------- #####
featureDetectionIntervall = 5   # how often the feature detection should run in seconds
developer = False               # if the developer windows should be shown (grayscale, contours etc.)
blurIntensity = 7 
focus = 10
threshold1 = 95
threshold2 = 58
minArea = 25000

##### --------------------------------------------------------------- #####
##### --------------------- Variables & Objects --------------------- #####
##### --------------------------------------------------------------- #####
webcam = WebcamStream(webcamWidth=640, webcamHeight=360, developer=developer, focus=focus)
processor = ImageProcessor(developer=developer)
dataLoader = CardDataLoader()


##### --------------------------------------------------------------- #####
##### --------------------------- Execute --------------------------- #####
##### --------------------------------------------------------------- #####
if __name__ == "__main__":   
    ##### ------- Threading für Feature Extraction & Matching ------- #####
    #thread = threading.Thread(target=wiederholte_ausfuehrung, args=(featureDetectionIntervall,))
    #thread.daemon = True    # Der Thread wird als Hintergrundthread ausgeführt
    #thread.start()
    ##### ----------------------------------------------------------- #####

    dataLoader.loadData()
    

    # Das Hauptprogramm kann hier weiterlaufen
    while True:
        stream = webcam.getVideoStreamImages()
        imageOriginal = stream.copy()
        imageContour = stream.copy()
        imageBoundingBox = stream.copy()
        if developer == True:
            threshold1 = cv2.getTrackbarPos('Threshold1', 'Parameters')
            threshold2 = cv2.getTrackbarPos('Threshold2', 'Parameters')
        
        streamGreyScale, streamCanny, streamDilation = processor.preProcessImage(stream=stream, blurIntensity=blurIntensity, 
                                                                                 threshold1=threshold1, threshold2=threshold2)
        processor.processImage(imageDilation=streamDilation, imageContour=imageContour, 
                               imageBoundingBox=imageBoundingBox, imageOriginal=imageOriginal, 
                               dataLoader=dataLoader, minArea=minArea)

        if developer:
            image_stack = webcam.stackImages(0.8, ([stream, streamGreyScale, streamCanny], 
                                                   [streamDilation, imageContour, imageBoundingBox]))
            cv2.imshow('CardDetector_Dev', image_stack)
        else:
            cv2.imshow('CardDetector', imageBoundingBox)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

            

        
