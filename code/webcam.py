import cv2
import numpy as np

class WebcamStream():

    def __init__(self, webcamWidth = 480, webcamHeight = 320, developer=False, focus = 20) -> None:
        self.developer = developer
        self.focus = focus
        print('WebcamStram erstellt.')

        if self.developer:
            cv2.namedWindow('WebcamProperties')
            cv2.resizeWindow('WebcamProperties', 600, 40)
            cv2.createTrackbar('Focus', 'WebcamProperties', 20, 250, self.empty)
        
        self.stream = cv2.VideoCapture(0)    
        self.stream.set(3, webcamWidth)
        self.stream.set(4, webcamHeight)


    def getVideoStreamImages(self):
        sucess, image = self.stream.read()
        image = cv2.rotate(image, cv2.ROTATE_180)
        self.focus = cv2.getTrackbarPos('Focus', 'WebcamProperties') if self.developer else self.focus
        self.stream.set(cv2.CAP_PROP_FOCUS, self.focus)
        return image
    

    def empty(self, a):
        pass


    def stackImages(self, scale, imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range ( 0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank]*rows
            hor_con = [imageBlank]*rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor= np.hstack(imgArray)
            ver = hor
        return ver