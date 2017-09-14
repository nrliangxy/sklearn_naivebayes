import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
def getTrainData(path,num):
    df = pd.read_csv(path)
    data_list = []
    class_list = []
    for i in df['des']:
        tken = [word for word in nltk.word_tokenize(i)]
        tags = [i[0] for i in nltk.pos_tag(tken) if (("NN" in i[1]) or ("J" in i[1]) or ("RB" in i[1]))]
        filtered_0 = [word.lower() for word in tags if word not in stopwords.words("english")]
        filtered_1 = [word for word in filtered_0 if not any(char.isdigit() for char in word)]
        filtered = [nltk.stem.SnowballStemmer("english").stem(word) for word in filtered_1]
        features = []