#coding: utf-8
import os
import time
import random
import jieba
import sklearn
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
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []
    # traverse the folder
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path,folder)
        files = os.listdir(new_folder_path)
        # read file
        j = 1
        for file in files:
            if j > 100: #limit its number
                break
            with open(os.path.join(new_folder_path,file), 'r') as fp:
                raw = fp.read()
            jieba.enable_parallel(2)
            word_cut = jieba.cut(raw,cut_all=False)
            word_list = list(word_cut)
            jieba.disable_parallel()
            data_list.append(word_list)
            class_list.append(folder)
            j += 1
    data_class_list = zip(data_list, class_list)
    random.shuffle(data_class_list)
    index = int(len(data_class_list)*test_size) + 1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)

    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if all_words_dict.has_key(word):
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    all_words_tuple_list = sorted(all_words_dict.items(),key=lambda f:f[1], reverse=True)
    all_words_dict = list(zip(*all_words_tuple_list)[0])
    return all_words_list, train_data_list, test_data_list, train_data_list, test_data_list
