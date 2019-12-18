# @Author  : GentleCP
# @Email   : 574881148@qq.com
# @File    : naive_bayes.py
# @Item    : PyCharm
# @Time    : 2019-08-09 08:59
# @WebSite : https://www.gentlecp.com

import os,shutil
import jieba
import warnings
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

root_path = 'K:/学习资料/研究生学习/课内学习/nlp/数据集/THUCNews/test'

def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def file_split(root_dir):
    mkdir(root_dir+'/train')
    mkdir(root_dir+'/test')
    for class_dir in ['体育','娱乐','家居','彩票','房产','教育','时尚','时政','星座','游戏','社会','科技','股票','财经']:
        train_dir = root_dir+'/train'+'/'+class_dir
        test_dir = root_dir+'/test'+'/'+class_dir
        mkdir(train_dir)
        mkdir(test_dir)
        source_dir = root_dir+'/'+class_dir
        files = os.listdir(source_dir)

        train_files = files[0:int(0.8*len(files))]   # 将数据集中80%用于训练
        test_files = files[int(0.8*len(files)):]    # 20%用于测试

        for file in train_files:
            shutil.move(source_dir+'/'+file,train_dir+'/'+file)
        for file in test_files:
            shutil.move(source_dir+'/'+file,test_dir+'/'+file)
        os.rmdir(source_dir)  # 移除现有的目录

def cut_words(file_path):
    '''
    对文本切词, 就是把中文的词语放到一块
    :param file_path: 文件路径
    :return: 用空格分词的字符串
    '''
    text_with_spaces = ''
    with open(file_path,'r', encoding='gb18030') as f:
        text = f.read()
        textcut = jieba.cut(text)
        for word in textcut:
            text_with_spaces += word + ' '
    return text_with_spaces


def load_file(file_dir, label):
    '''
    加载目录文件
    :param file_dir: 文件目录
    :param label: 文档标签
    :return: 分词后的文档列表和标签
    '''
    file_list = os.listdir(file_dir)
    words_list = []
    labels_list = []
    for file in file_list:
        file_path = file_dir + '/' + file
        words_list.append(cut_words(file_path))
        labels_list.append(label)
    return  words_list, labels_list

def train_test():
    l_train_words = []
    l_train_labels = []
    l_test_words = []
    l_test_labels = []
    # for klass in ['体育','娱乐','家居','彩票','房产','教育','时尚','时政','星座','游戏','社会','科技','股票','财经']:
    #     print("\r","当前加载训练数据:{}".format(klass),end='',flush=True)
    #     words, labels= load_file(root_path+'/train/'+klass, klass)
    #     l_train_words.extend(words)
    #     l_train_labels.extend(labels)


    for klass in ['体育']:
        print("\r","当前加载测试数据:{}".format(klass),end='',flush=True)
        words, labels= load_file(root_path+'/train/'+klass, klass)
        l_test_words.extend(words)
        l_test_labels.extend(labels)


    print("加载停用词")
    stop_words = open('stopword.txt', encoding='utf-8').read()
    stop_words = stop_words.encode('utf-8').decode('utf-8-sig')
    stop_words = stop_words.split('\n')

    tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)

    print("提取特征")
    # train_features = tf.fit_transform(l_train_words)
    test_features = tf.transform(l_test_words)

    print("开始训练..")
    with open('model.pkl','r') as f:
        clf = pickle.load(f)

    # clf = MultinomialNB(alpha=0.001).fit(train_features, l_train_labels)
    predicted_labels=clf.predict(test_features)

    # 计算准确率
    print('准确率为：', metrics.accuracy_score(l_test_labels, predicted_labels))
    # print("保存模型...")
    # with open('model.pkl','wb') as f:
    #     pickle.dump(clf,f)


def predict(file_path):
    text = cut_words(file_path)
    print(text)
    print("加载停用词")
    stop_words = open('stopword.txt', encoding='utf-8').read()
    stop_words = stop_words.encode('utf-8').decode('utf-8-sig')
    stop_words = stop_words.split('\n')

    tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
    features = tf.transform([text])
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
    predict_label = clf.predict(features)
    print(predict_label)
    
if __name__ == '__main__':
    # file_split(root_path)
    # train_test()
    predict('test.txt')