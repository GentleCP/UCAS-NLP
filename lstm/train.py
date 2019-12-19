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

from pre_process import load_data

print("init")
# def read_flag():
flags = tf.app.flags
flags.DEFINE_string('data', 'sub-THUCNews', "path of data")
flags.DEFINE_string('model', 'lstm', "type of mode, include [lstm,]")
flags.DEFINE_float('test_size', 0.2, "size of test dataset")

FLAGS = flags.FLAGS

path = Path(FLAGS.data)

labels, inputs, tokenize, length = load_data("sub-THUCNews.csv")

# x: input y:label

# 划分测试集训练集
input_train, input_valid, label_train, label_valid = train_test_split(tokenize, labels, test_size=FLAGS.test_size)
print(f"data num: {len(tokenize)}, train num {len(input_train)}, valid num: {len(input_valid)}")

# 对分类的标签编码
le = LabelEncoder()
le_label_train = le.fit_transform(label_train).reshape(-1, 1)
le_label_valid = le.fit_transform(label_valid).reshape(-1, 1)

# 分类标签转为 one hot
ohe = OneHotEncoder()
le_label_train = ohe.fit_transform(le_label_train).toarray()
le_label_valid = ohe.fit_transform(le_label_valid).toarray()

# input_train = le.fit_transform(input_train).reshape(-1, 1)
# input_valid = le.fit_transform(input_valid).reshape(-1, 1)

# one hot
# ohe = OneHotEncoder()
# input_train = ohe.fit_transform(le_label_train).toarray()
# input_valid = ohe.transform(le_label_valid).toarray()  # ?

# 使用tokenizer 建立词组和向量的词典 因为程序只能处理数字
max_word = 1000  # 词语的最多数量
max_len = 500
tok = Tokenizer(num_words=max_word)
tok.fit_on_texts(input_train)

# print(input_train)
train_seq = tok.texts_to_sequences(input_train)
train_seq_mat = sequence.pad_sequences(train_seq, maxlen=max_len)

valid_seq = tok.texts_to_sequences(input_valid)
valid_seq_mat = sequence.pad_sequences(valid_seq, maxlen=max_len)
print(f"shape: train_seq: {train_seq_mat.shape}, valid_seq: {valid_seq_mat.shape}")

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
# inputs = Input(name="inputs", shape=[max_len])
# layer = Embedding(max_word+1, 128, input_length=max_len)(inputs)
# layer = LSTM(128)(layer)
# layer = Dense(128, activation='relu', name="FC1")(layer)
# layer = Dropout(0.5)(layer)
# layer = Dense(14, activation='softmax', name="FC2")(layer)
# model = Model(inputs=inputs, outputs=layer)

max_fetures = 20000
model = Sequential()
model.add(Embedding(max_fetures, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(14, activation='sigmoid'))


model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
# model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['categorical_accuracy'])
# checkpointer = ModelCheckpoint(filepath=Path(".") / "model" / "{epoch:02d}.hdf5", verbose=1)

model.fit(
    train_seq_mat,
    le_label_train,
    batch_size=128,
    epochs=10,
    validation_data=(valid_seq_mat, le_label_valid),  # turn to mat
    callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)],
)
model.save("model/lstm.h5")
# model.save(f"model/{int(time.time())}.h5")
