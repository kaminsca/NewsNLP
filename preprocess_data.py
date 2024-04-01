import nltk
from nltk.corpus import stopwords
import csv
import re
from constants import EXTRACTED_DATA_PATH, PROCESSED_DATA_PATH

class DataProcessor:
    def __init__(self, extracted_data_path=EXTRACTED_DATA_PATH):
        self._extracted_data_path = extracted_data_path
        self._rows_processed = 0
        self._processed_data = []
        self.__apply_filters()

    def __apply_filters(self):
        with open(EXTRACTED_DATA_PATH, mode='r', newline='', encoding='utf-8') as file:
            creader = csv.reader(file, delimiter='|')
            c = 0
            self._processed_data.append(next(creader))
            for row in creader:
                if len(row) < 2:
                    continue
                title, county = row[0], row[1]
                no_punctuation = self.clean_punctuation(title)
                no_stopwords = self.remove_stopwords(no_punctuation, county.lower())
                #take most recent transformation
                new_row = [no_stopwords]
                new_row.extend([itm for itm in row[1:]])
                if no_stopwords:
                    self._processed_data.append(new_row)
                c +=1
            self._rows_processed = c
        file.close()

    def save_processed_data(self):
        with open(PROCESSED_DATA_PATH, mode='w', newline='', encoding='utf-8') as file:
            cwriter = csv.writer(file, delimiter='|')
            cwriter.writerows(self._processed_data)

    #https://swatimeena989.medium.com/beginners-guide-for-preprocessing-text-data-f3156bec85ca
    def clean_punctuation(self, sentence):
        return re.sub(r'[^\w\s]','',sentence)

    def remove_stopwords(self, sentence, county):
        stopwords_list = stopwords.words('english')
        stopwords_list.append(county)
        stopwords_list.extend(['january','february','march','april','may','june','july','august','september','november','december'])
        return " ".join([word.lower() for word in sentence.split() if word.lower() not in stopwords_list and not word.isdigit()])
