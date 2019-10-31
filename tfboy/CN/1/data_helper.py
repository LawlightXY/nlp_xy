# -*- coding:utf-8 -*-
import re
import jieba
import numpy as np
import uniout
import sys
import codecs
reload(sys)
sys.setdefaultencoding('utf-8')
def clean_str(string):
    """
    该函数的作用是去掉一个字符串中的所有非中文字符
    :param string:
    :return: 返回处理后的字符串
    """
    string.strip('\n')
    pattern = re.compile(ur'[^\u4e00-\u9fa5]')
    string = re.sub(pattern, " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def cut_line(line):
    """
    该函数的作用是 先清洗字符串，然后分词
    :param line:
    :return: 分词后的结果，如 ：     衣带  渐宽  终  不悔
    """
    line = clean_str(line)
    seg_list = jieba.cut(line)
    cut_words = " ".join(seg_list)
    return cut_words


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    按行载入数据，然后分词。同时构造标签
    :param positive_data_file:
    :param negative_data_file:
    :return:  分词后的结果和标签
    x_text:   ['衣带 渐宽 终 不悔',' 为 伊 消得 人憔悴']
    y: [[1 0],[ 1 0]]
    """
    # positive = []
    # negative = []
    # for line in open(positive_data_file, "rb").read().decode('utf-8'):
    #     positive.append(cut_line(line))
    # for line in open(negative_data_file, "rb").read().decode('utf-8'):
    #     negative.append(cut_line(line))

    positive_examples = codecs.open(positive_data_file, "r").readlines()
    positive_examples = [s.strip().decode('utf-8') for s in positive_examples]
    negative_examples = codecs.open(negative_data_file, "r").readlines()
    negative_examples = [s.strip().decode('utf-8') for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [cut_line(sent) for sent in x_text]

    # positive = open(positive_data_file, "rb").read().decode('utf-8')
    # negative = open(negative_data_file, "rb").read().decode('utf-8')
    #
    # positive_examples = positive.split('\n')[:-1]
    # negative_examples = negative.split('\n')[:-1]
    #
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = [s.strip() for s in negative_examples]
    # x_text = positive_examples + negative_examples
    # x_text = [cut_line(sent) for sent in x_text]
    positive_label = [[0, 1] for _ in positive_examples]  # 构造one-hot 标签[[0, 1], [0, 1], [0, 1], [0, 1],....]
    negative_label = [[1, 0] for _ in negative_examples]

    y = np.concatenate([positive_label, negative_label], axis=0)

    return x_text, y


def gen_batch(x_train, y_train, begin, batch_size):
    data_size = len(y_train)
    start = (begin * batch_size) % data_size
    end = min(start + batch_size, data_size)
    x = x_train[start:end]
    y = y_train[start:end]
    return x, y


if __name__ == '__main__':
    positive_data_file = '../data/ham_5000.utf8'
    negative_data_file = '../data/spam_5000.utf8'
    x_text, y = load_data_and_labels(positive_data_file, negative_data_file)

    print(x_text)
