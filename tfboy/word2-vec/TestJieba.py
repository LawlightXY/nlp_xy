import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs,sys


f = codecs.open('/Users/xuyang/PycharmProjects/untitled/tfboy/word2-vec/wiki_zhs.text', 'r', encoding='utf-8')
target = codecs.open('/Users/xuyang/PycharmProjects/untitled/tfboy/word2-vec/wiki_fc.txt', 'w', encoding='utf-8')
print('open files')
line_num = 1
line = f.readline()
while line:
    print('processing',line_num,'article')
    line_seg = " ".join(jieba.cut(line))
    target.writelines(line_seg)
    line_num += 1
    line = f.readline()
f.close()
target.close()
exit()