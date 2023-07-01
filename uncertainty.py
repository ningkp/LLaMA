# -*- coding: utf-8 -*-            
# @Time : 2023/7/1 20:23
# @Author: LeoN

import csv
from nltk import word_tokenize

def algo1_score(fp):
    reader=csv.reader(fp)
    result=[]
    for num,line in enumerate(reader):
        temp_set=set()
        count=0
        idx=line[0]
        for item in line[2:]:
            words=word_tokenize(item)
            # words=set(words)
            count+=len(words)
            temp_set.update(words)

        result.append((idx,len(temp_set)/count))
    result.sort(key=lambda x:x[1], reverse=True)

    return result

if __name__ == '__main__':
    with open('result/generate/result.csv','r',encoding='utf-8') as f:
        result=algo1_score(f)

    with open('result/score/wikisql_algo1.csv','w',encoding='utf-8',newline='') as f:
        csv.writer(f).writerows(result)