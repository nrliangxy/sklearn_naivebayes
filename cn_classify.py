#coding: utf-8
import os
import time
import random
import jieba
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
#print(make_word_set('/home/lxy/Downloads/nlp/Lecture_2/Lecture_2/Naive-Bayes-Text-Classifier/Database/SogouC/Sample/C000010/10.txt'))
#text processing
def text_processing(folder_path, test_size=0.2):
    """
    :param folder_path:
    :param test_size:
    :return: text processing
    """
    folder_list =  os.listdir(folder_path)
    data_list = []
    class_list = []
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path,folder)
        files = os.listdir(new_folder_path)
        for file in files:
            with open(os.path.join(new_folder_path,file), 'r') as fp:
                raw = fp.read()
            jieba.enable_parallel(2)
            word_cut = jieba.cut(raw,cut_all=False)
            word_list = list(word_cut)
            #print(word_list)
            jieba.disable_parallel()
            data_list.append(word_list)
            class_list.append(folder)

    # data_class_list = zip(data_list, class_list)
    # random.shuffle(data_class_list)
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
    return all_words_list
    #return all_words_list,train_data_list,test_data_list,train_class_list,test_class_list
def words_dict(all_words_list,deleteN,stopwords_set=set()):
    feature_words = []
    n = 1
    for t in range(deleteN,len(all_words_list), 1):
        if n > 1000:
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
            n += 1
    return feature_words
def text_features(train_data_list,test_data_list,feature_words,flag='nltk'):
    def text_features(text,feature_words):
        text_words = set(text)
        if flag == 'nltk':
            features = {word:1 if word in text_words else 0 for word in feature_words}
        elif flag == 'sklearn':
            features = [1 if word in text_words else o for word in feature_words]
        else:
            features = []
        return features
    train_feature_list = [text_features(text,feature_words) for text in train_data_list]
    test_feature_list = [text_features(text,feature_words) for text in test_data_list]
    return train_feature_list,test_feature_list


    #print(data_class_list)
    # random.shuffle(data_class_list)
    # index = int(len(data_class_list)*test_size) + 1
    # train_list = data_class_list[index:]
    # test_list = data_class_list[:index]
    # train_data_list, train_class_list = zip(*train_list)
    # test_data_list, test_class_list = zip(*test_list)
    #
    # all_words_dict = {}
    # for word_list in train_data_list:
    #     for word in word_list:
    #         if all_words_dict.has_key(word):
    #             all_words_dict[word] += 1
    #         else:
    #             all_words_dict[word] = 1
    # all_words_tuple_list = sorted(all_words_dict.items(),key=lambda f:f[1], reverse=True)
    # all_words_dict = list(zip(*all_words_tuple_list)[0])
    # return all_words_list, train_data_list, test_data_list, train_data_list, test_data_list


folder_path = '/home/lxy/Downloads/nlp_corpus/Lecture_2/Lecture_2/Naive-Bayes-Text-Classifier/Database/SogouC/Sample'
print(words_dict(text_processing(folder_path),1))

