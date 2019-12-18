import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class PreProcesser:

    def __init__(self):
        self._train_path = 'data/train'
        self._test_path = 'data/test'

    def _load_stop_words(self):
        with open('stopword.txt', 'r',encoding='utf-8') as f:
            return f.read().split('\n')

    def __cut_words(self, file_path):
        '''
        读取单个文件，用jieba分词，返回结果
        :param file_path:
        :return:
        '''
        with open(file_path, 'r',encoding='gb18030') as f:
            cut_res = list(jieba.cut(f.read()))
            return cut_res

    def __load_data(self,file_dir, label):
        '''
        读取目录下文件，进行分词，打上标签
        :param file_dir: 文件目录
        :param label:
        :return: 分词列表，标签列表
        '''
        l_files = os.listdir(file_dir)  # 目录下的所有文件名
        l_words = []
        l_labels = []
        for file in l_files:
            file_path = file_dir + '/' + file
            l_words.extend(self.__cut_words(file_path))
            l_labels.extend(label)

        return l_words,l_labels

    def _load_datas(self,root_path):
        '''
        对每个类别分别进行load操作
        :param root_path:
        :return:
        '''
        all_words = []
        all_labels = []
        for label in ["体育","女性","文学","校园"]:
            l_words, l_labels = self.__load_data(root_path+'/'+ label,label)
            all_words += l_words
            all_labels += l_labels

        return all_words,all_labels

    def _get_words_features(self, all_words, l_stop_words):
        tf = TfidfVectorizer(stop_words=l_stop_words,max_df=0.5)
        train_features = tf.fit_transform(all_words)
        return train_features

    def _train(self,train_features,train_labels):
        clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)
        return clf

    def _test(self,clf, test_features,test_labels):
        predicted_labels = clf.predict(test_features)
        count = 0
        for predict,real in zip(predicted_labels,test_labels):
            if predict == real:
                count +=1
        print('准确率：{}'.format(count/len(predicted_labels)))

    def run(self):
        l_stop_words = self._load_stop_words()
        train_words, train_labels = self._load_datas(self._train_path)
        train_features = self._get_words_features(train_words, l_stop_words)
        test_words, test_labels = self._load_datas(self._test_path)
        test_features  = self._get_words_features(test_words, l_stop_words)
        clf = self._train(train_features,train_labels)
        self._test(clf,test_features,test_labels)

if __name__ == '__main__':
    pp = PreProcesser()
    pp.run()
