import os
import zipfile
import lzma
import jieba
import pandas as pd
from pathlib import Path

data_path = "../../THUCNews/THUCNews"
csv_path = data_path + ".csv"

test_set_path = "test"


def get_test_set(s: Path, d: Path):
    files = os.listdir(s)
    print(files)
    for folder in files:
        # os.mkdir(d/folder)
        test_set = os.listdir(s / folder)
        print(folder,len(test_set), len(test_set) * 0.2)


    pass


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
        print(label)
        for file in os.listdir(p / label):
            con = open(p / label / file, encoding='u8').read()
            # con = con.encode('gbk')
            con = con.replace('\u3000', ' ')
            con = con.replace('\xa0', ' ')
            con = con.replace('\n', ' ')

            j = jieba.cut(con)
            j = " ".join([i for i in j if i not in stopwords])
            # j = list(j)
            # print(j)

            # exit()
            data_input.append(con)
            data_tokenlize.append(j)
            data_label.append(label)
            data_length.append(len(j))

            if not len(data_input) % 100:
                print(len(data_input))
                continue
                # print(data_tokenlize)
                # print(data_input)
                # print(data_label)
                # print(data_length)
                break

        # break

    return data_label, data_input, data_tokenlize, data_length


def token():
    path = Path(data_path)

    labels, inputs, tokenize, length = read_file(path)

    df = pd.DataFrame([labels, inputs, tokenize, length]).T
    df = df.rename(columns={0: "labels", 1: 'inputs', 2: "tokenize", 3: 'length'})
    print(df)
    df.to_csv(csv_path)
    return df


def load_data(filename):
    xzfilename = f"{filename}.csv.xz"
    datafilename = f"{filename}.csv"
    print(f"unzip {xzfilename} to {datafilename}")

    # z = zipfile.ZipFile(zipfilename)
    # z.extract(datafilename, '.')
    # lzma.decompress(xzfilename)

    # df = pd.read_csv(datafilename)
    df = pd.read_csv(lzma.open(xzfilename))
    print(f"load {filename} finish.")
    return df.labels, df.inputs, df.tokenize, df.length


if __name__ == "__main__":
    f = "THUCNews"
    # d = token()
    # d = load_data(f)
    # print(d)
    get_test_set(Path(data_path), Path(test_set_path))
