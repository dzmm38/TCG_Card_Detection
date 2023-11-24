from imageClassifierFeatureDetection import ImageClassifierFeatureDetection
from cardDataLoader import CardDataLoader

import cv2
import numpy as np
import time

webcamWidth = 800 #640 
webcamHeight = 800  #360

cam = cv2.VideoCapture(0)
cam.set(3, webcamWidth)     # 3 = width
cam.set(4, webcamHeight)    # 4 = height
classifier = ImageClassifierFeatureDetection()
classifier.importImages()
classifier.findDesOfAllImages()
loader = CardDataLoader()
loader.loadData()

def empty(a):
    pass

cv2.namedWindow('Parameters')
cv2.resizeWindow('Parameters', 640, 240)
cv2.createTrackbar('Threshold1', 'Parameters', 95, 255, empty)
cv2.createTrackbar('Threshold2', 'Parameters', 58, 255, empty)
cv2.createTrackbar('Area', 'Parameters', 60000, 100000, empty)
cv2.createTrackbar('MaxArea', 'Parameters', 95000, 100000, empty)
cv2.createTrackbar('Focus', 'Parameters', 20, 250, empty)


def stackImages(scale,imgArray):
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

def getContours(image, image_Contour, image_bounding_box, image_clear):
    contours, hierachy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)

        threshold_area = cv2.getTrackbarPos('Area', 'Parameters')
        threshold_Maxarea = cv2.getTrackbarPos('MaxArea', 'Parameters')

        if area > threshold_area: #and area < threshold_Maxarea

            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)
            cv2.drawContours(image, [approx_corners], -1, (0, 255, 0), 2)

            for corner in approx_corners:
                x, y = corner[0]
                cv2.circle(image_Contour, (x, y), 10, (255, 0, 0), -1)
                cv2.putText(image_Contour, "Point: " + str(corner[0]), (x, y-20),cv2.FONT_HERSHEY_COMPLEX, .5, (0, 255, 0),1)

            if len(approx_corners) == 4:
                width, height = 250,350
                
                pts1 = approx_corners.astype(np.float32)
                #[[366,394]]#[[365,163]]#[[204,159]]#[[197,388]]            [204,159]
                #pts1 = np.float32([[204,159],[365,163],[197,388],[366,394]])
                #pts2 = np.float32([[0,0], [width,0], [0, height], [width,height]])
                pts2 = np.float32([[width,height],[width,0],[0,0],[0, height]])
                matrix = cv2.getPerspectiveTransform(pts1,pts2)
                imgOutput = cv2.warpPerspective(image_clear, matrix, (width,height))
                imgOutput = cv2.rotate(imgOutput, cv2.ROTATE_180)
                #cv2.imshow('Cutout_Dev', imgOutput)
                
                x1, y1 = 0,184
                x2, y2 = 250,184
                x3, y3 = 0,221
                x4, y4 = 250,221
                card_name = imgOutput[y1:y3, x1:x2]
                #cv2.imshow('CardName_Dev', card_name)

                x1_art, y1_art = 12,15
                x2_art, y2_art = 233,15
                x3_art, y3_art = 12,178
                x4_art, y4_art = 233,178
                card_art = imgOutput[y1_art:y3_art, x1_art:x2_art]
                #cv2.imshow('Card_Art_Dev', card_art)

                card_art_grey = cv2.cvtColor(card_art, cv2.COLOR_BGR2GRAY)
                #cv2.imshow('Card_Art_Grey_Dev', card_art_grey)

                card_id = classifier.findCardId(cardImage=card_art_grey)
                cardNameMain, cardNameSupport = loader.getCardNames(card_id=card_id)

            cv2.drawContours(image_Contour, contour, -1, (255,0,255), 7)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            #print(len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(image_bounding_box, (x,y),(x+w, y+h), (0,255,0), 2)

            #cv2.putText(image_bounding_box, "Area: " + str(int(area)), (x, y-20),cv2.FONT_HERSHEY_COMPLEX, .5, (0, 255, 0),1)
            if len(approx_corners) == 4:
                cv2.putText(image_bounding_box, cardNameMain, (x,y-20), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0),1)
                cv2.putText(image_bounding_box, cardNameSupport, (x,y-5), cv2.FONT_HERSHEY_COMPLEX, .4, (0, 255, 0),1)


while True:
    focus = cv2.getTrackbarPos('Focus', 'Parameters')
    cam.set(cv2.CAP_PROP_FOCUS, focus)

    sucess, image = cam.read()
    image = cv2.rotate(image, cv2.ROTATE_180)
    img_Contour = image.copy()
    image_bounding_box = image.copy()
    image_clear = image.copy()
    
    image_blur = cv2.GaussianBlur(image, (7,7), 1)
    image_grey = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)

    threshold1 = cv2.getTrackbarPos('Threshold1', 'Parameters')
    threshold2 = cv2.getTrackbarPos('Threshold2', 'Parameters')
    image_Canny = cv2.Canny(image_grey,threshold1,threshold2)
    kernel = np.ones((4,4))
    image_dilation = cv2.dilate(image_Canny, kernel, iterations=1)

    getContours(image_dilation, image_Contour=img_Contour, image_bounding_box=image_bounding_box, image_clear=image_clear)

    image_stack = stackImages(0.8, ([image, image_grey, image_Canny],
                                    [image_dilation, img_Contour, image_bounding_box]))
    cv2.imshow('CardDetector_Dev', image_stack)
    
    cv2.imshow('CardDetector', image_bounding_box)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('d'):
        cv2.imshow('CardDetector_Dev', image_stack)
