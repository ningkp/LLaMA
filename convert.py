# -*- coding: utf-8 -*-            
# @Time : 2023/7/8 16:41
# @Author: LeoN

import os,csv,json

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
    base_dir='dataset/SAMSum'
    for name in ['train','val','test']:
        json2csv(base_dir,name)