import cv2
import os

class ImageClassifierFeatureDetection():

    def __init__(self) -> None:
        self.set1Path = '../data/set_1/'
        self.images = []
        self.classNames = []
        self.desList = []
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher()
        print('Klassifikator erstellt.')


    def importImages(self):
        myList = os.listdir(self.set1Path)
        for image_class in myList: 
            current_image = cv2.imread(f'{self.set1Path}/{image_class}', 0) # 0 for greyscale
            self.images.append(current_image)
            self.classNames.append(os.path.splitext(image_class)[0])

        print('Anzahl an verfügbaren Klassen: ', len(myList))


    def findDesOfAllImages(self):
        for image in self.images:
            kp, des = self.orb.detectAndCompute(image, None)
            self.desList.append(des)
        
        print('Anzahl an Klassen in Description List: ', len(self.desList))


    def findCardId(self, cardImage, threshold=15):
        kp, des_current = self.orb.detectAndCompute(cardImage, None)
        matchList = []
        card_id = -1
        try:
            for des in self.desList:
                matches = self.bf.knnMatch(des, des_current, k=2)
                goodMatches = []
                for m,n in matches:
                    if m.distance < 0.75 * n.distance:     # number can be changed
                        goodMatches.append([m])
            
                matchList.append(len(goodMatches)) 
        except:
            pass

        if len(matchList) != 0:
            if max(matchList) > threshold:
                card_id = matchList.index(max(matchList))
        
        card_id = self.__getCardDataName__(card_id)
        return card_id
    

    def __getCardDataName__(self, card_id):
        cardName = 'Unknown Card'
        if card_id != -1:
            cardName = self.classNames[card_id]
        return cardName
