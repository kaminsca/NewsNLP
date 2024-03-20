import nltk
from nltk.corpus import stopwords
import csv
import re
#https://swatimeena989.medium.com/beginners-guide-for-preprocessing-text-data-f3156bec85ca
def clean_punctuation(sentence):
    return re.sub(r'[^\w\s]','',sentence)

def remove_stopwords(sentence,county):
    stopwords_list = stopwords.words('english')
    stopwords_list.append(county)
    stopwords_list.extend(['january','february','march','april','may','june','july','august','september','november','december'])
    stopwords_removed = [word.lower() for word in sentence.split() if word.lower() not in stopwords_list and not word.isdigit()]
    #print(stopwords_removed)

if __name__ == "__main__":
    with open('./output/master_data_no_content.csv', mode='r', newline='') as file:
        creader = csv.reader(file, delimiter='|')
        c = 0
        for row in creader:
            c +=1
            title, county = row[0], row[1]
            no_punctuation = clean_punctuation(title)
            no_stopwords = remove_stopwords(no_punctuation,county.lower())
        print(c)
        