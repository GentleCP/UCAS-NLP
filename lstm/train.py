import os
import time
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, LSTM, Activation, Dropout, Input
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

try:
    from lstm.pre_process import load_data
except ImportError:
    from .pre_process import load_data

max_word = 3000  # 词语的最多数量
max_len = 1000
datafile = "THUCNews"
testfile = "sub-THUCNews"

print(f"init at: {time.ctime()}")
# def read_flag():


labels, data, tokenize, length = load_data(datafile)
label_test, _, input_test, _ = load_data(testfile)

# x: input y:label

# 划分测试集训练集
input_train, input_valid, label_train, label_valid = train_test_split(tokenize, labels, test_size=0.3)
# input_train, input_test, label_train, label_test = train_test_split(input_train, label_train, test_size=0.2)
print(f"data num: {len(tokenize)}, train num {len(input_train)}, valid num: {len(input_valid)}")

# 对分类的标签编码
le = LabelEncoder()
le_label_train = le.fit_transform(label_train).reshape(-1, 1)
le_label_valid = le.fit_transform(label_valid).reshape(-1, 1)
le_label_test = le.fit_transform(label_test).reshape(-1, 1)
# 分类标签转为 one hot
ohe = OneHotEncoder()
le_label_train = ohe.fit_transform(le_label_train).toarray()
le_label_valid = ohe.fit_transform(le_label_valid).toarray()
le_label_test = ohe.fit_transform(le_label_test).toarray()

# input_train = le.fit_transform(input_train).reshape(-1, 1)
# input_valid = le.fit_transform(input_valid).reshape(-1, 1)

# one hot
# ohe = OneHotEncoder()
# input_train = ohe.fit_transform(le_label_train).toarray()
# input_valid = ohe.transform(le_label_valid).toarray()  # ?

# 使用tokenizer 建立词组和向量的词典 因为程序只能处理数字
tok = Tokenizer(num_words=max_word)
tok.fit_on_texts(input_train)

# print(input_train)
train_seq = tok.texts_to_sequences(input_train)
train_seq_mat = sequence.pad_sequences(train_seq, maxlen=max_len)

valid_seq = tok.texts_to_sequences(input_valid)
valid_seq_mat = sequence.pad_sequences(valid_seq, maxlen=max_len)

test_seq = tok.texts_to_sequences(input_test)
test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)

print(f"shape: train_seq: {train_seq_mat.shape}, valid_seq: {valid_seq_mat.shape}, test_seq: {test_seq_mat.shape}")

# print(train_seq_mat.shape)

# set model
"""
model = Sequential()

hidden_size = 128
model.add(Input(name="input", shape=[max_len]))
model.add(Embedding(max_len, hidden_size, input_length=max_len))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(Activation('softmax'))
# model.add(Dropout(rate=0.5))
model.add(Dense(10))
model.summary()
"""

# epochs=20时 正确率70%
inputs = Input(name="inputs", shape=[max_len])
layer = Embedding(max_word + 1, 128, input_length=max_len)(inputs)
layer = LSTM(128)(layer)
layer = Dense(128, activation='relu', name="FC1")(layer)
layer = Dropout(0.5)(layer)
layer = Dense(14, activation='softmax', name="FC2")(layer)
model = Model(inputs=inputs, outputs=layer)

"""
# 最简单的网络 准确率 23%
max_fetures = 20000
model = Sequential()
model.add(Embedding(max_fetures, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(14, activation='sigmoid'))
"""

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
# model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['categorical_accuracy'])
# checkpointer = ModelCheckpoint(filepath=Path(".") / "model" / "{epoch:02d}.hdf5", verbose=1)

model.fit(
    train_seq_mat,
    le_label_train,
    batch_size=128,
    epochs=50,
    validation_data=(valid_seq_mat, le_label_valid),  # turn to mat
    # callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)],
)
model.save("model/lstm.h5")

with open('model/token.pickle', 'wb') as f:
    pickle.dump(tok, f, protocol=pickle.HIGHEST_PROTOCOL)
# model.save(f"model/{int(time.time())}.h5")

score, acc = model.evaluate(test_seq_mat, le_label_test, batch_size=128)
print(f"score: {score}, acc: {acc}")

#
# text = '北京时间12月18日，2019年东亚杯冠军产生，韩国队1-0击败日本队，以3战全胜的战绩历史上第5次获得东亚杯的冠军，成为首支东亚杯3连冠球队！虽然世界杯和亚洲杯的表现不如对手，但这一次韩国队找回场子，此外在去年亚运会以及今年世青赛，韩国国奥与韩国国青都曾击败日本队，3条战线都击败对手。'
# t = "北京 时间 12 月 18 2019 东亚 杯 冠军 产生 韩国队 1 0 击败 日本队 3 战全胜 战绩 历史 5 获得 东亚 杯 冠军 成为 首支 东亚 杯 3 连冠 球队 虽然 世界杯 亚洲杯 表现 不如 对手 一次 韩国队 找回 场子 此外 去年 亚运会 以及 今年 世青赛 韩国 国奥 韩国 国青 都 曾 击败 日本队 3 条 战线 都 击败 对手"
#
# seq = tok.texts_to_sequences([t])
# seq_mat = sequence.pad_sequences(seq, maxlen=max_len)
#
# c = model.predict(seq_mat)
# print(c)
