from gensim.models import word2vec
import logging
import codecs,sys

logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)

raw_sentences = ['the quick brown fox jumps over the lazy dogs','yoyoyo you go home now to sleep']
sentences = [s.split() for s in raw_sentences]
print(sentences)
model =  word2vec.Word2Vec(sentences,min_count=1)
sim = model.similarity('dogs','you')
print(sim)
print(model['the'])

# f = codecs.open('/Users/xuyang/PycharmProjects/untitled/tfboy/word2-vec/wiki_fc.txt','r',encoding='utf-8')
# line = f.readline()
# print(line)