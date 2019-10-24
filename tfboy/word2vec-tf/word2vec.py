# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import math
import collections
import pickle as pkl
import re
import jieba
import os
import os.path as path
import codecs,sys
import uniout

class word2vec():
    def __init__(self,vocab_list = None,embedding_size =200,win_len = 3,learning_rate = 1,num_sample =100):
        self.batch_size = None
        assert type(vocab_list) == list
        self.vocab_list = vocab_list
        self.vocab_size = vocab_list.__len__()
        self.win_len = win_len
        self.learning_rate = learning_rate
        self.num_sample = num_sample
        self.embedding_size = embedding_size
        self.word2id = {}
        for i in range(self.vocab_size):
            self.word2id[self.vocab_list[i]] = i
        self.train_words_num = 0#训练了多少个词
        self.train_sentence_num = 0
        self.train_times = 0
        self.bulid_graph()
        self.sess = tf.Session()

    def bulid_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32,shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32,shape=[self.batch_size,1])
            self.embedding_dict = tf.Variable(tf.truncated_normal(shape=[self.vocab_size,self.embedding_size]))
            self.nec_weight = tf.Variable(tf.truncated_normal(shape=[self.vocab_size,self.embedding_size]))#类别，维度
            self.bias = tf.Variable(tf.zeros([self.vocab_size]))

            embed = tf.nn.embedding_lookup(self.embedding_dict,self.train_inputs)

            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=self.nec_weight,biases=self.bias,inputs = embed,labels = self.train_labels,num_sampled= self.num_sample,
                               num_classes=self.vocab_size)
            )
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.test_word_id = tf.placeholder(tf.int32,shape=[None])
            voc_l2_model = tf.sqrt(tf.reduce_sum(tf.square(self.embedding_dict),1))#报错
            self.normaled_embedding = self.embedding_dict/voc_l2_model
            test_embed = tf.nn.embedding_lookup(self.embedding_dict, self.test_word_id)
            self.similarity = tf.matmul(test_embed,self.normaled_embedding)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def train_by_sent(self,input_sentence=[]):
        sent_num = input_sentence.__len__()
        batch_inputs = []
        batch_labels = []
        for sent in input_sentence:
            for i in range(sent.__len__()):
                start = max(0,i-self.win_len)
                end = min(sent.__len__(),i+self.win_len)
                for index in range(start,end):
                    if index == i:
                        continue
                    else:
                        input_id = self.word2id.get(sent[i])
                        label_id = self.word2id.get(sent[index])
                        batch_inputs.append(input_id)
                        batch_labels.append(label_id)
        batch_inputs = np.array(batch_inputs,dtype=np.int32)
        batch_labels = np.array(batch_labels,dtype=np.int32)
        batch_labels = np.reshape(batch_labels,[batch_labels.__len__(),1])
        feed_dict = {
            self.train_inputs:batch_inputs,
            self.train_labels:batch_labels
        }
        loss = self.sess.run(self.train_op,feed_dict=feed_dict)
        self.train_words_num += batch_inputs.__len__()
        self.train_sentence_num +=input_sentence.__len__()
        self.train_times += 1

    def cal_similarity(self,test_word_id):
        sim_matrix = self.sess.run(self.similarity,feed_dict={self.test_word_id:test_word_id})
        test_words = []
        near_words = []
        for i in range(test_word_id.__len__()):
            test_word.append(self.vocab_list[test_word_id][i])
            nearest_id = [sim_matrix[i,:].argsort()[1:10]]
            nearest_word = [self.vocab_list[x] for x in nearest_id]
        return test_word,near_words


if __name__ == '__main__':
    # reload(sys)
    # sys.setdefaultencoding("utf8")
    stop_words = []
    with codecs.open('/Users/xuyang/Desktop/data mining/stopwords.txt') as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)

    raw_word_list = []
    sentence_list = []
    # with codecs.open('/Users/xuyang/Desktop/data mining/AI培训资料/0.流失预警案例预习材料/Sophon数据操作说明.txt') as f:
    with codecs.open('/Users/xuyang/Desktop/data mining/280.txt') as f:
        line = f.readline()
        while line:
            while '\n' in line:
                line = line.replace('\n','')
            if len(line)>0:
                raw_words = list(jieba.cut(line))
                dealed_words = []
                for word in raw_words:
                    if word not in stop_words:
                        raw_word_list.append(word)
                        dealed_words.append(word)
                sentence_list.append(dealed_words)
            line = f.readline()
    word_count = collections.Counter(raw_word_list)
    word_count = word_count.most_common(30000)
    word_list = [x[0] for x in word_count]

    w2v = word2vec(vocab_list = word_list,embedding_size =200,learning_rate = 1,num_sample =100)
    num_steps = 1000
    for i in range(num_steps):
        sent = sentence_list(i)
        w2v.train_by_sent(sentence_list)
    # w2v.save_model(save_path)
    test_word = ['剑气','无敌']
    test_id = [word_list.index(x) for x in test_word]
    test_word, near_words = w2v.cal_similarity(test_id)
    print(test_word, near_words)

