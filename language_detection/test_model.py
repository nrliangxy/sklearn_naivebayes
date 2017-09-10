from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
import pickle
# x, y = zip(*dataset)
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
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
#vec.fit(x_train)
def get_features(x):
    vec.transform(x)

with open('/home/lxy/Documents/language_detection_1.pickle','rb') as fr:
    new_nbs = pickle.load(fr)
    x_train = 'Wave Money , a joint venture between mobile operator Telenor and Yoma Bank, is looking to expand its digital payment services in Myanmar. Telenor and Yoma Bank formed the mobile financial services firm in 2016, by taking a 51 per cent and 49 per cent stake in it respectively. After having spread its agents’ network in 255 out of the 330 townships in Myanmar, Wave Money would further expand into untouched markets in the country, possibly by 2019. “Counting into 2019 that (digital payments) will be definitely something that we will look at, expanding online e-commerce space as well,” Brad Jones, chief executive officer of Wave Money said.'
    print(new_nbs.predict(vec.transform(x_train), ['en']))