import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
class LanguageDetector():
    def __init__(self, classifier=MultinomialNB()):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(ngram_range=(1,2), max_features=1000, preprocessor=self._remove_noise)
    def _remove_noise(self, document):
        noise_pattern = re.compile("|".join(["http\S+", "\@\w+", "\#\w+"]))
        clean_text = re.sub(noise_pattern,"",document)
        return clean_text
    def features(self, x):
        return self.vectorizer.transform(x)
    def fit(self, x, y):
        self.vectorizer.fit(x)
        self.classifier.fit(self.features(x), y)
    def predict(self, x):
        return self.classifier.predict(self.features([x]))
    def score(self, x, y):
        return self.classifier.score(self.features(x), y)

path = '/home/lxy/Downloads/nlp_corpus/Lecture_2/Lecture_2/Language-Detector/data.csv'
>>>>>>> d2b9c7706f1e1c68d3f0bbf7093636d6b9218871
in_f = open(path)
lines = in_f.readlines()
in_f.close()
dataset = [(line.strip()[:-3], line.strip()[-2:]) for line in lines]
x, y = zip(*dataset)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
language_detector = LanguageDetector()
language_detector.fit(x_train, y_train)
print(language_detector.predict('Bonjour, je vais réussir'))
print(language_detector.score(x_test, y_test))
