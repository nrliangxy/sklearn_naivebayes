import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import pickle
def getTrainData(path,num):
    df=pd.read_csv(path)
    trainingset=[]
    for line in df['des']:
        if num//10 == 0:labl="0Agriculture, Forestry & Fishing"
        elif 15>num>9: labl="1Mining"
        elif 18>num>14: labl="2Construction"
        elif 40>num>19: labl="3Manufacturing"
        elif num//10 == 4: labl="4Transportation, Communications, Electric, Gas & Sanitary Services"
        elif 52>num>49 : labl="5Wholesale Trade"
        elif 60>num>51: labl="6Retail Trade"
        elif 68>num>59: labl="7Finance, Insurance & Real Estate"
        elif 90>num>69: labl="8services"
        elif 99>num>90: labl="9Public Administration"
        trainingset.append((line,labl))
    return trainingset
train_set=[]
test_set=[]
for i in range(100):
    try:
        train_test_all=getTrainData("/home/lxy/Downloads/category/category%.2d.csv"%i,i)
        if len(train_test_all)>10:
            train_set.extend(train_test_all[:-5])
            test_set.extend(train_test_all[-5:])
            print (i,len(train_set),len(test_set))
    except:continue
train_data_list, train_class_list = zip(*train_set)
test_date_list, test_class_list = zip(*test_set)
def remove_noise(document):
    noise_pattern = re.compile("|".join(["http\S+","\@\w+","\#\w+"]))
    clean_text = re.sub(noise_pattern,"",document)
    return clean_text.strip()
vec = CountVectorizer(
    lowercase=True, # lowercase the text
    analyzer='char_wb', # tokenise by character ngrams
    ngram_range=(1,2), # use ngrams or size 1 and 2
    max_features=1000, # keep the most common 1000 ngrams
    preprocessor=remove_noise
)
vec.fit(train_data_list)
def get_features(x):
    vec.transform(x)
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(vec.transform(train_data_list), train_class_list)
with open('/home/lxy/Documents/industry_divide.pickle', 'wb') as fw:
    pickle.dump(classifier, fw)
print(classifier.score(vec.transform(test_data_list), test_class_list))
