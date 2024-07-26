# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:09:42 2024

@author: zhaodf
"""
#更新了Python3.9和tf2.10

#采用Transformer对IMDB数据集进行分类
#IMDB数据集是一组电影评论（英语），用于情感分析任务，数据集包含了50,000个电影评论
#每个评论都有一个标签，表示评论的情感是正面的(positive)还是负面的(negative)
#该数据集是一个非常常见的自然语言处理任务，用于训练模型对文本进行情感分析

#1.数据不平衡：负面评论比正面评论多，可能导致模型偏向于预测负面情感
#2.语言噪声：评论中可能包含拼写错误、语法错误和语义歧义
#3.多样性：评论中可能包含不同的话题、语言风格和情感表达方式

#该示例Keras官方来源 https://keras.io/examples/nlp/text_classification_with_transformer/

import tensorflow as tf
import keras
from keras import layers
# from  keras.utils.data_utils import pad_sequences

## Download the data 数据加载
vocab_size = 20000  # Only consider the top 20k words
maxlength = 200  # Only consider the first 200 words of each movie review
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)#完成加载及Token等操作
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.utils.pad_sequences(x_train, maxlen=maxlength)
x_val = keras.utils.pad_sequences(x_val, maxlen=maxlength)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

## 1st layer for tokens
## 2nd layer for positions (index)

class TokenAndPositionEmbedding(layers.Layer):
  def __init__(self, maxlen, vocab_size, embed_dim):
    super().__init__()
    self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
    self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

  def call(self, x):
    maxlen = tf.shape(x)[-1]
    positions = tf.range(start=0, limit=maxlen, delta=1)
    positions = self.pos_emb(positions)
    x = self.token_emb(x)
    return x+positions 


## Define the model 建立模型
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlength,))
embedding_layer = TokenAndPositionEmbedding(maxlength, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# ## Train and evaluate the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_val, y_val))

####绘图#################################
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

import matplotlib.pyplot as plt

figsize = 7,5
figure, ax = plt.subplots(figsize=figsize)
plt.tick_params(labelsize=12)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.plot(history.history['accuracy'],'-r',label='training acc',linewidth=1.5)
plt.plot(history.history['val_accuracy'],'-b',label='val acc',linewidth=1.5)
plt.title('model accuracy',font2)
plt.ylabel('accuracy',font2)
plt.xlabel('epoch',font2)
plt.legend(loc='lower right',prop=font2)

figsize = 7,5
figure, ax = plt.subplots(figsize=figsize)
plt.tick_params(labelsize=12)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.plot(history.history['loss'],'-r',label='training loss',linewidth=1.5)
plt.plot(history.history['val_loss'],'-b', label='val loss',linewidth=1.5)
plt.title('model loss',font2)
plt.ylabel('loss',font2)
plt.xlabel('epoch',font2)
plt.legend(loc='upper right',prop=font2)
