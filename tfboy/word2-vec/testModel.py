# -*- coding:utf-8 -*-
from gensim.models import Word2Vec
import sys
import uniout

wiki_model = Word2Vec.load("/Users/xuyang/PycharmProjects/untitled/tfboy/word2-vec/wiki_fc.model")
print(wiki_model)
testWords = [u'苹果',u'数学',u'学术',u'白痴',u'篮球'];
for i in range(5):
    res = wiki_model.most_similar(testWords[i])
    print(testWords[i])
    print(res)

