path = '/home/lxy/Downloads/nlp_corpus/Lecture_2/Lecture_2/Language-Detector/data.csv'
in_f = open(path)
lines = in_f.readlines()
in_f.close()
dataset = [(line.strip()[:-3], line.strip()[-2:]) for line in lines]
#print(dataset[:5])
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
import pickle
x, y = zip(*dataset)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
#print(len(x_train))
def remove_noise(document):
    noise_pattern = re.compile("|".join(["http\S+","\@\w+","\#\w+"]))
    clean_text = re.sub(noise_pattern,"",document)
    return clean_text.strip()
#print(remove_noise("Trump images are now more popular than cat gifs. @trump #trends http://www.trumptrends.html"))
vec = CountVectorizer(
    lowercase=True, # lowercase the text
    analyzer='char_wb', # tokenise by character ngrams
    ngram_range=(1,2), # use ngrams or size 1 and 2
    max_features=1000, # keep the most common 1000 ngrams
    preprocessor=remove_noise
)
vec.fit(x_train)
def get_features(x):
    vec.transform(x)
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(vec.transform(x_train), y_train)
with open('/home/lxy/Documents/language_detection_1.pickle', 'wb') as fw:
    pickle.dump(classifier, fw)
print(classifier.score(vec.transform(x_test), y_test))