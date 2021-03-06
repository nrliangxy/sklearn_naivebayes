import os
import codecs
import jieba
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pylab
import matplotlib.pyplot as plt
def make_word_set(word_file):
    """
    :param word_file: txt
    :return: remove duplicate
    """
    word_set = set()
    with open(word_file, 'r') as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word) > 0 and word not in word_set:
                word_set.add(word)
    return word_set

def text_processing(folder_path, test_size=0.2):
    """
    :param folder_path:
    :param test_size:
    :return: text processing
    """
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path,folder)
        files = os.listdir(new_folder_path)
        for file in files:
            with open(os.path.join(new_folder_path, file), 'r') as fp:
                raw = fp.read()
            # try:
            #     with codecs.open(os.path.join(new_folder_path,file), 'r', 'GB18030') as fp:
            #         raw = fp.read()
            # except UnicodeDecodeError:
            #     pass
            jieba.enable_parallel(2)
            word_cut = jieba.cut(raw,cut_all=False)
            word_list = list(word_cut)
            #print(word_list)
            jieba.disable_parallel()
            data_list.append(word_list)
            class_list.append(folder)

    train_data_list, test_data_list, train_class_list, test_class_list = train_test_split(data_list, class_list, test_size=test_size)
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict:
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    all_words_tuple_list = sorted(all_words_dict.items(),key=lambda f:f[1], reverse=True)
    all_words_list = list(zip(*all_words_tuple_list))[0]
    #return all_words_list
    return all_words_list,train_data_list,test_data_list,train_class_list,test_class_list
def words_dict(all_words_list,deleteN,stopwords_set=set()):
    """
    :param all_words_list: Key words sorted by word frequency
    :param deleteN: Randomly set parameters
    :param stopwords_set: stopwords
    :return: feature words
    """
    feature_words = []
    n = 1
    for t in range(deleteN,len(all_words_list), 1):
        if n > 1000:
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
            n += 1
    return feature_words
def text_features(train_data_list,test_data_list,feature_words,flag='sklearn'):
    """
    :param train_data_list: train data [['','',..],['','',..]....]
    :param test_data_list: test data [['','',..],['','',..],....]
    :param feature_words: feature words ['','',...]
    :param flag:
    :return: train feature and test feature [[1,0,1..],[0,0,1..],...]
    """
    def text_features(text,feature_words):
        text_words = set(text)
        if flag == 'nltk':
            features = {word:1 if word in text_words else 0 for word in feature_words}
        elif flag == 'sklearn':
            features = [1 if word in text_words else 0 for word in feature_words]
        else:
            features = []
        return features
    train_feature_list = [text_features(text,feature_words) for text in train_data_list]
    test_feature_list = [text_features(text,feature_words) for text in test_data_list]
    return train_feature_list,test_feature_list
def text_classifier(train_feature_list, test_feature_list,train_class_list,test_class_list,flag='sklearn'):
    """
    :param train_feature_list: train feature [[1,0,1..],[0,0,1..],...]
    :param test_feature_list: test feature [[1,0,1..],[0,0,1..],...]
    :param train_class_list: train class ['tech','financial','political',....]
    :param test_class_list: test class ['tech','financial','political',....]
    :param flag:
    :return: test accuracy [0.6,0.8,0.7,.....]
    """
    if flag == 'nltk':
        train_flist = zip(train_feature_list,train_class_list)
        test_flist = zip(test_feature_list,test_class_list)
        classifier = nltk.classify.NaiveBayesClassifier.train(train_flist)
        test_accuracy = nltk.classify.accuracy(classifier,test_flist)
    elif flag == 'sklearn':
        classifier = MultinomialNB().fit(train_feature_list, train_class_list)
        test_accuracy = classifier.score(test_feature_list,test_class_list)
    else:
        test_accuracy = []
    return test_accuracy
print('start')
folder_path = '/home/lxy/Downloads/nlp_corpus/Lecture_2/Lecture_2/Naive-Bayes-Text-Classifier/Database/SogouC/Sample'
all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = text_processing(folder_path,test_size=0.2)
stopwords_file = '/home/lxy/Downloads/nlp_corpus/Lecture_2/Lecture_2/Naive-Bayes-Text-Classifier/stopwords_cn.txt'
stopwords_set = make_word_set(stopwords_file)
flag = 'sklearn'
deleteNs = range(0,1000,20)
test_accuracy_list = []
for deleteN in deleteNs:
    feature_words = words_dict(all_words_list,deleteN,stopwords_set)
    train_feature_list, test_feature_list = text_features(train_data_list,test_data_list,feature_words,flag)
    test_accuracy = text_classifier(train_feature_list,test_feature_list,train_class_list,test_class_list,flag)
    test_accuracy_list.append(test_accuracy)
print(test_accuracy_list)
print(len(test_accuracy_list))
plt.plot(deleteNs,test_accuracy_list)
plt.title('Relationship of deleteNs and test_accuracy')
plt.xlabel('deleteNs')
plt.ylabel('test_accuracy')
plt.show()
print('finished')


