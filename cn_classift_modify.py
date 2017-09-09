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

    # folder_list = os.listdir(folder_path)
    # data_list = []
    # class_list = []
    # # traverse the folder
    # for folder in folder_list:
    #     new_folder_path = os.path.join(folder_path,folder)
    #     files = os.listdir(new_folder_path)
    #     # read file
    #     j = 1
    #     for file in files:
    #         if j > 100: #limit its number
    #             break
    #         with open(os.path.join(new_folder_path,file), 'r') as fp:
    #             raw = fp.read()
                #print(raw)
folder_path = '/home/lxy/Downloads/nlp_corpus/Lecture_2/Lecture_2/Naive-Bayes-Text-Classifier/Database/SogouC/Sample'
print(text_processing(folder_path))