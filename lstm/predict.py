# coding=utf8
# predict
from keras.models import load_model
from keras.preprocessing import sequence
import pickle
import jieba
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pre_process import load_data

label, data, tokenize, length = load_data('sub-THUCNews.csv')
input_data, input_label, _, _ = train_test_split(tokenize, label, test_size=0.9)

text = """
习近平指出，5年来，在崔世安行政长官带领下，澳门行政、立法、司法机关严格依照宪法和基本法办事，认真履职尽责，付出了辛勤劳动，取得了累累硕果，交出了一份让中央满意、让澳门居民满意的答卷。中央政府对大家的工作是充分肯定、高度评价的。
"""


def predict(text):
    max_len = 500
    stopwords = [i.strip() for i in open("stopwords.txt", encoding='u8').read()]

    token = " ".join([i for i in jieba.cut(text) if i not in stopwords])
    print(token)

    model = load_model('model/lstm.h5')
    tok = pickle.load(open('model/token.pickle', 'rb'))

    test_seq = tok.texts_to_sequences([token])
    test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)

    """
    le = LabelEncoder()
    le_label_test = le.fit_transform(input_label).reshape(-1,1)
    
    # 分类标签转为 one hot
    ohe = OneHotEncoder()
    le_label_test = ohe.fit_transform(le_label_test).toarray()
    
    
    score, ac = model.evaluate(test_seq_mat, le_label_test, batch_size=128)
    print(score, ac)
    """
    l = ['体育', '娱乐', '家居', '彩票', '房产', '教育', '时尚', '时政', '星座', '游戏', '社会', '科技', '股票', '财经']
    confidence = dict()

    pre = model.predict(test_seq_mat)
    for i, j in enumerate(l):
        # print(i, j)
        confidence[j] = pre[0][i]

    # print(pre)
    # print(confidence)

    max_index = np.argmax(pre)
    # print(max_index)
    max_type = l[max_index]
    # print(f"{max_type}: {confidence[max_type]}")
    confidence = {k: v for k, v in sorted(confidence.items(), key=lambda item: item[1], reverse=True)}
    return max_type, confidence


if __name__ == "__main__":
    t = predict(text)
    print(t)
