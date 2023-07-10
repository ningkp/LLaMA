# -*- coding: utf-8 -*-            
# @Time : 2023/7/1 20:03
# @Author: LeoN

import csv

def merge(directory,num):
    idx_dict={}
    for i in range(num):
        path='{}/result{}.csv'.format(directory,i)
        with open(path,'r',encoding='utf-8') as f:
            reader=csv.reader(f)
            for idx,question,answer in reader:
                if idx not in idx_dict:
                    idx_dict[idx]=[question]
                idx_dict[idx].append(answer.strip())

    result_list=[]
    for k,v in idx_dict.items():
        if len(v)==11:
            result_list.append([k]+v)
    result_list.sort(key=lambda x:int(x[0]))

    result_path='{}/result.csv'.format(directory)
    with open(result_path,'w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f)
        writer.writerows(result_list)

if __name__ == '__main__':
    merge('result/generate/samsum',10)