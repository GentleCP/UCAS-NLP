import os
import jieba
import pandas as pd
from pathlib import Path


def read_file(p: Path) -> (list, list):
    stopwords = [i.strip() for i in open("stopwords.txt", encoding='u8').readlines()]

    _labels = os.listdir(p)
    _data = dict()
    _data_list = list()
    # _data = list()
    data_input = []
    data_label = []
    data_tokenlize = []
    data_length = []
    for label in _labels:
        for file in os.listdir(p / label):
            con = open(p / label / file, encoding='u8').read()
            con.replace(u'\u3000', ' ')
            con.replace(u'\xa0', ' ')

            j = jieba.cut(con)
            j = " ".join([i for i in j if i not in stopwords])
            # j = list(j)

            data_input.append(con)
            data_tokenlize.append(j)
            data_label.append(label)
            data_length.append(len(j))

            if len(data_input) > 100:
                continue
                print(data_tokenlize)
                print(data_input)
                print(data_label)
                print(data_length)
                break

        # break

    return data_label, data_input, data_tokenlize, data_length


def token():
    path = Path("sub-THUCNews")

    labels, inputs, tokenize, length = read_file(path)

    df = pd.DataFrame([labels, inputs, tokenize, length]).T
    df = df.rename(columns={0: "labels", 1: 'inputs', 2: "tokenize", 3: 'length'})
    print(df)
    df.to_csv(f'sub-THUCNews.csv', )
    return df


def load_data(filename):
    df = pd.read_csv(filename)
    return df.labels, df.inputs, df.tokenize, df.length


if __name__ == "__main__":
    d = token()
    d = load_data("sub-THUCNews.csv")
    print(d)
