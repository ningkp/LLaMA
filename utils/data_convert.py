# -*- coding: utf-8 -*-            
# @Time : 2023/6/28 16:37
# @Author: LeoN

from fastparquet import ParquetFile
import pandas as pd
from prompter import wikisql_prompt
import os,csv,json

# pf=ParquetFile('wikisql-test.parquet')
# df=pf.to_pandas()
# df.to_csv('dataset/wikisql/test.csv',encoding='utf-8')
# df=pd.read_csv('dataset/wikisql/train.csv')


def json2csv(path,name):
    json_name=name+'.json'
    csv_name=name+'.csv'
    json_path=os.path.join(path,json_name)
    csv_path=os.path.join(path,csv_name)
    with open(json_path,'r',encoding='utf-8') as f:
        json_reader=json.load(f)
        res=[]
        for line in json_reader:
            res.append([
                int(line['id'].replace('-','')),
                line['dialogue'].replace('\r\n','\n').strip(),
                line['summary'].replace('\r\n','\n').strip()
            ])
    with open(csv_path,'w',encoding='utf-8',newline='') as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(res)

if __name__ == '__main__':
    base_dir= '../dataset/SAMSum'
    for name in ['train','val','test']:
        json2csv(base_dir,name)