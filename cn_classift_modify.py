import os
def text_processing(folder_path, test_size=0.2):
    """
    :param folder_path:
    :param test_size:
    :return: text processing
    """
    for path,d,filelist in os.walk(folder_path):
        for filename in filelist:
            direct = os.path.join(path,filename)
            with open(direct, 'r') as fp:
                raw = fp.read()
                print(raw)
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
folder_path = '/home/lxy/Downloads/nlp/Lecture_2/Lecture_2/Naive-Bayes-Text-Classifier/Database/SogouC/Sample'
text_processing(folder_path)