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