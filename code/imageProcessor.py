import cv2
import numpy as np
import datetime

from scipy.spatial.distance import euclidean
from imageClassifierFeatureDetection import ImageClassifierFeatureDetection

class ImageProcessor():

    def __init__(self, developer = False, featureDetectionIntervall = 5) -> None:
        self.developer = developer
        self.classifier = ImageClassifierFeatureDetection()
        self.startTime = datetime.datetime.now() 
        self.cardNameMain = ''
        self.cardNameSupport = ''
        self.featureDetectionIntervall = featureDetectionIntervall
        print('Image Processor erstellt um: ', self.startTime)

        if developer:
            cv2.namedWindow('Parameters')
            cv2.resizeWindow('Parameters', 600, 120)
            cv2.createTrackbar('Threshold1', 'Parameters', 95, 255, self.empty)
            cv2.createTrackbar('Threshold2', 'Parameters', 58, 255, self.empty)
            cv2.createTrackbar('MinArea', 'Parameters', 60000, 100000, self.empty)

        self.classifier.importImages()
        self.classifier.findDesOfAllImages()
        

    def empty(self, a):
        pass


    def preProcessImage(self, stream, blurIntensity, threshold1, threshold2):
        streamBlur = cv2.GaussianBlur(stream, (blurIntensity,blurIntensity), 1)
        streamGreyScale = cv2.cvtColor(streamBlur, cv2.COLOR_BGR2GRAY)
        streamCanny = cv2.Canny(streamGreyScale, threshold1, threshold2)
        kernel = np.ones((4,4))
        streamDilation = cv2.dilate(streamCanny, kernel, iterations=1)

        return streamGreyScale, streamCanny, streamDilation


    def __getCutoutViews__(self, approx_corners, imageOriginal, flip=True):
        cardCutoutWidth, cardCutoutHeight = 250,350 # die größe des cutouts der gesamten karte
        #verticalRangeName = list(range(184, 221)) #221 aber 222 damit auch 221 in der liste ist
        #horizontalRangeName = list(range(0, 250)) #250 aber 251 damit auch 250 in der liste ist
        verticalRangeArt = (15,178)   #178 aber 179 damit auch 178 in der liste ist
        horizontalRangeArt = (12,233)  #233 aber 234 damit auch 233 in der liste ist

        cornerPointsDetected = approx_corners.astype(np.float32) # der erfassten karte
        correctedOrientationList = self.__findCorrectOrientation__(cornerPointsDetected)
        cornerPointsDetected = correctedOrientationList

        cornerPointsTarget = np.float32([[cardCutoutWidth,cardCutoutHeight],[cardCutoutWidth,0],[0,0],[0, cardCutoutHeight]]) # des output bildes
        perspectiveMatrix = cv2.getPerspectiveTransform(cornerPointsDetected,cornerPointsTarget)
            
        cardImageView = cv2.warpPerspective(imageOriginal, perspectiveMatrix, (cardCutoutWidth,cardCutoutHeight))   # erstellt eine vogelperspektive
        if flip :
            cardImageView = cv2.rotate(cardImageView, cv2.ROTATE_180)   # dreht das bild um 180 Grad
        #cardNameView = cv2.cvtColor(cardImageView[verticalRangeName, horizontalRangeName], cv2.COLOR_BGR2GRAY)  # gray scale of cutout (name) / da ich dieses aktuell nicht verwende
        cardArtView = cv2.cvtColor(cardImageView[verticalRangeArt[0]:verticalRangeArt[1], 
                                                 horizontalRangeArt[0]:horizontalRangeArt[1]], 
                                                 cv2.COLOR_BGR2GRAY)     # gray scale of cutout (art)

        return cardImageView, cardArtView   #,cardNameView # cardNameView wird nicht erstellt da dieses aktuell nicht verwendet / gebraucht wird


    def __findCorrectOrientation__(self, approx_corners):
        referencePoint = approx_corners[0][0]
        distancesToReferencePoint = [0]

        for i in range(1, len(approx_corners)):
            point = approx_corners[i][0]
            distance = euclidean(referencePoint, point)
            distancesToReferencePoint.append(distance)

        ersterPunkt = approx_corners[0]
        approx_corners = np.delete(approx_corners, 0, axis=0)
        distancesToReferencePoint.pop(0)

        index = distancesToReferencePoint.index(max(distancesToReferencePoint))
        weitestWeg = approx_corners[index]
        approx_corners = np.delete(approx_corners, index, axis=0)
        distancesToReferencePoint.pop(index)

        index = distancesToReferencePoint.index(max(distancesToReferencePoint))
        zweitNeachster = approx_corners[index]
        naechster = approx_corners[distancesToReferencePoint.index(min(distancesToReferencePoint))]

        if naechster[0][0] > ersterPunkt[0][0]:
            correctedOrientationList = np.float32([ersterPunkt,zweitNeachster,weitestWeg,naechster])
        else:
            correctedOrientationList = np.float32([naechster,weitestWeg,zweitNeachster,ersterPunkt])
            
        return correctedOrientationList



    def __drawBoundingBox__(self, contour, imageBoundingBox):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(imageBoundingBox, (x,y),(x+w, y+h), (0,255,0), 2)
        return x, y


    def __showCornerPoints__(self, approx_corners, imageContour):
        for corner in approx_corners:
            x, y = corner[0]
            cv2.circle(imageContour, (x, y), 10, (255, 0, 0), -1)
            cv2.putText(imageContour, "Point: " + str(corner[0]), (x, y-20),cv2.FONT_HERSHEY_COMPLEX, .5, (0, 255, 0),1)
    

    def processImage(self, imageDilation, imageContour, imageBoundingBox, imageOriginal, dataLoader, minArea):
        contours, _ = cv2.findContours(imageDilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # finden der Konturen
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.developer:
                threshold_area = cv2.getTrackbarPos('MinArea', 'Parameters')
            else:
                threshold_area = minArea

            if area > threshold_area:
                largest_contour = max(contours, key=cv2.contourArea) # hier die größe kontur --> sollte im normalfall immer die karte sein 
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)   # holen der Ecken!

                if len(approx_corners) == 4:
                    # hier wird sichergestellt das eine gewisse area gegeben ist sowie das es sich um ein rechteck handel und somit um eine Karte
                    cv2.drawContours(imageContour, contour, -1, (255, 0, 0), 7)
                    
                    # beinhaltet getCutoutViews und die Klassifikation
                    self.doTimedFunction(dataLoader=dataLoader, approx_corners=approx_corners, imageOriginal=imageOriginal)

                    x, y = self.__drawBoundingBox__(contour, imageBoundingBox)
                    cv2.putText(imageBoundingBox, self.cardNameMain, (x,y-20), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0),1)
                    cv2.putText(imageBoundingBox, self.cardNameSupport, (x,y-5), cv2.FONT_HERSHEY_COMPLEX, .4, (0, 255, 0),1)
                    
                    if self.developer:
                        cv2.putText(imageBoundingBox, "Area: " + str(int(area)), (x, y-35),cv2.FONT_HERSHEY_COMPLEX, .5, (0, 255, 0),1)
                        self.__showCornerPoints__(approx_corners, imageContour)

                    if self.developer:
                        cv2.putText(imageBoundingBox, "Area: " + str(int(area)), (x, y-35),cv2.FONT_HERSHEY_COMPLEX, .5, (0, 255, 0),1)


    def doTimedFunction(self, dataLoader, imageOriginal, approx_corners):
        currentTime = datetime.datetime.now()
        timeDifference = (currentTime - self.startTime).total_seconds()

        if timeDifference > self.featureDetectionIntervall:
            self.startTime = currentTime
            cardImageView, cardArtView = self.__getCutoutViews__(approx_corners=approx_corners, imageOriginal=imageOriginal)
            self.classification(dataLoader=dataLoader, cardArtView=cardArtView)

            if self.cardNameMain == 'Unknown Card':     # hier dann noch eine klassifikation mit einem umgedrehten bild um auch karten auf dem kopf zu erkennen!
                cardImageView, cardArtView = self.__getCutoutViews__(approx_corners=approx_corners, imageOriginal=imageOriginal, flip=False)
                self.classification(dataLoader=dataLoader, cardArtView=cardArtView)


            if self.developer:
                cv2.imshow('Cutout_Dev', cardImageView)
                #cv2.imshow('CardName_Dev', card_name)
                cv2.imshow('Card_Art_Grey_Dev', cardArtView)




    def classification(self, dataLoader, cardArtView):
        #print('Neue Klassifikation')
        card_id = self.classifier.findCardId(cardImage=cardArtView)
        self.cardNameMain, self.cardNameSupport = dataLoader.getCardNames(card_id=card_id)

        
        return self.cardNameMain, self.cardNameSupport