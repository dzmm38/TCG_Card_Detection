import pandas as pd

class CardDataLoader():
    
    def __init__(self) -> None:
        self.dataListPath = '../data/card_list_excel.xlsx'
        self.cardData = []
        print('Data Loader erstellt.')


    def loadData(self):
        dfData = pd.read_excel(self.dataListPath)
        dfData = dfData.fillna('')  # für songs und action karten
        self.cardData = dfData
        print('Karten daten sind geladen.')
        print('Anzahl an verfügbaren Klassen: ', dfData.shape[0])

    
    def getCardNames(self, card_id):
        if card_id != 'Unknown Card':
            card_data_row = self.__getDataRow__(card_id)
            mainName = card_data_row['Main-Name'].values[0]
            supportName = card_data_row['Support-Name'].values[0]
        else:
            mainName = 'Unknown Card'
            supportName = ''
        return mainName, supportName


    def __getDataRow__(self, card_id):
        row = self.cardData[self.cardData['Card-Id'] == card_id]
        return row