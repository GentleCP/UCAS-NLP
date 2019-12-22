import json
import jieba
import collections
import os
import pickle
import logging
from collections import  Counter

def load_stop_words(stop_words_path='stopword.txt',stop_signals={'\n','\u3000','\xa0',' ',}):
    if not os.path.exists(stop_words_path):
        stop_words_path = 'NaiveBayes/stopword.txt'
    with open(stop_words_path,encoding='UTF-8-sig') as f:
        s_stop_words = set()
        for line in f:
            s_stop_words.add(line.strip())
        s_stop_words.update(stop_signals)
        return s_stop_words



def fit_transform(features, words_counter):
    '''
    将输入的文档依据特征集合转换为向量形式
    '''
    doc_feature_vector = []
    for feature in features:
        if words_counter.get(feature):
            doc_feature_vector.append(words_counter.get(feature))
        else:
            doc_feature_vector.append(0)
    return doc_feature_vector


def transform(features, file_path):
    '''
    接受特征字典{'feature':weight}和文件（训练或测试），将数据转换为向量表示返回
    '''
    d_categories = {}  # 统计每个类的数量信息
    l_category_vectors = []  # 统计每个类和特征向量信息
    with open(file_path, encoding='UTF-8-sig') as f:
        for line in f:
            label, content = line.strip().split(' ', 1)
            words_counter = collections.Counter(content.split(' '))
            doc_feature_vector = fit_transform(features, words_counter)
            d_category_vector = {
                'category': label,
                'vector': doc_feature_vector
            }
            l_category_vectors.append(d_category_vector)
            try:
                d_categories[label] += 1
            except KeyError:
                d_categories[label] = 1

        return {
            'd_categories': d_categories,  # 每个类别的统计信息，用于计算先验概率
            'l_category_vectors': l_category_vectors  # 由类别和特征向量组成的字典列表
        }

def load_model(file_path='bayes_model.pkl'):
    '''
    导入训练好的bayes模型
    :param file_path:
    '''
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logging.error('模型文件不存在，请先训练模型并存储！')
        exit(404)


class NaiveBayes:

    def __init__(self):
        self._features = self.__load_features()
        self._prior_prb = {}  # 各个分类的先验概率
        self._likelihood_prb = {}  # 各个特征在不同分类下的概率
        self._l_categories = ['财经', '彩票', '房产', '股票', '家居', '教育', '科技', '社会', '时尚', '时政', '体育', '星座', '游戏', '娱乐']

        self._train_path = 'train.txt'
        self._test_path = 'test.txt'

        self._s_stop_words = None


    def __load_features(self):
        with open('features.json', encoding='utf-8') as f:
            features = json.loads(f.read())
        return features

    def _tokenization(self, text):
        return jieba.cut(text)

    def _fit_transform(self, words_counter):
        '''
        将输入的文档依据特征集合转换为向量形式
        '''
        doc_feature_vector = []
        for feature in self._features:
            if words_counter.get(feature):
                doc_feature_vector.append(words_counter.get(feature))
            else:
                doc_feature_vector.append(0)
        return doc_feature_vector

    def _text_preprocess(self, text):
        '''
        预处理数据，返回文本的特征向量表达形式
        '''
        if not self._s_stop_words:
            self._s_stop_words = load_stop_words()
        token_res = []
        for word in self._tokenization(text):  # 分词
            if word not in self._s_stop_words:  # 去停用词
                token_res.append(word)
        words_counter = Counter(token_res)
        return self._fit_transform(words_counter)


    def predict(self, text):
        vector = self._text_preprocess(text)
        max_predict_prb = 0
        for category in self._l_categories:
            # 计算该文本在各个分类下的概率
            prod_likelihood_prb = 1
            for i, v in enumerate(vector):
                # 对向量中每一个不为0的分量求概率
                if v != 0:
                    prod_likelihood_prb *= self._likelihood_prb[(i, category)]
            predict_prb = self._prior_prb[category] * prod_likelihood_prb
            #             print('category:{},prb:{}'.format(category, predict_prb))
            if predict_prb > max_predict_prb:
                predict_category = category  # 更新概率最大类别
                max_predict_prb = predict_prb
        return predict_category

TEST_TEXT ="""
婚恋心理：婆媳聊天必知的潜规则(图)
　　婆媳矛盾是我们中华民族的千古矛盾，一直都得不到缓解。这个社会的人都有两面性，大家嘴上说着一套一套的漂亮话，但是实际上所作所为，又是另外一回事。而婆媳关系也有两套规则，一套是明规则，还有一套潜规则，利用好了，这个千古矛盾对你来说将不再是难题。
　　婆媳相处如何妙用“潜规则”
　　可是，我们当中有多少人是口含银匙而生呢？多少人是公主下嫁招驸马的童话呢我们当中的大多数，不都是要为柴米油盐生计而喜怒哀乐吗？不都要正视如何和婆家人相处——我们不想可又不得不去做吗？
　　首先，我建议婆媳之间不要直接交流，有什么相左的意见应该通过先生缓冲一下，他是“汉奸”——会和皇军交流，也懂八路的心思。
　　其次，如果直接交流受阻，一定要先自省：自己冲不冲动，有没有言语不当的地方，对婆婆有没有肢体冲撞，自己如果和妈妈这么说，妈妈怎么反应如果这些都自省过了，没有问题，那就要和先生说，实事求是，注意方式，不要动怒说粗，宁可哭，不可以骂人。如果是自己做的过分，有形式上的不当之处，但是内容没有错，就得避重就轻一点了，但是要提。记住：提前招认，绝对好过后来被盘问不得不招。如果你有重大错误，对不起，我也不知道怎么办了。因为我从来不和婆婆正面交流不同意见。请其她姐妹指点吧。
　　总之，要做和先生解释说明冲突的第一人，要尽量心平气和，决不能搞人身攻击，婆婆丰满说人家是吹了气的青蛙，公公苗条说人家是榨干的甘蔗。要会做人，尤其是有外人在的场合，要表现的温和有礼，听话勤快，既让婆婆有面子，也可以请外人给你制造舆论。
　　婆媳相处，要善于利用“潜规则”
　　婆媳交流，要注意不能乱用潜规则，尽量说漂亮的官话。哪怕虚伪点，也不能来个赤裸裸的大实话，起码，不能首先使用大实话。聪明的妈妈会教女儿嘴巴要甜，说白了就是要会说官话。
　　当然，官话不仅仅是说话，还包括行动。例如一个五十多岁的媳妇得到了众人的赞扬，说她有媳妇相，自己都是有媳妇的人了，还那么孝顺婆婆。那她是怎么做的呢有客人来了，她贴身伺候婆婆，给婆婆拿着热水袋，香烟火柴，站在婆婆身边伺候着。其实，她这是在监视婆婆，让她没法说坏话，要说，只能说好话——这样，她的好名声就得到最权威的认可了。
　　如果娘家和婆家势力悬殊，或是先生靠着爸爸提携，你就不用担心什么婆媳关系了，婆婆哪还敢说你坏话她得为儿子好啊。这种情况下，媳妇若是为长久计，就要锦上添花，待公婆好一些，省得老公翅膀硬了老爹退休了，公婆甚至老公一口恶气吐到脸上来。如果不想费力气，那也不用做什么，大家场面上过的去就行了。
"""

if __name__ == '__main__':
    nb = load_model(file_path="bayes_model.pkl")
    print(nb.predict(TEST_TEXT))