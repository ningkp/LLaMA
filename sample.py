# -*- coding: utf-8 -*-            
# @Time : 2023/7/1 23:11
# @Author: LeoN

import csv,random,datasets
from transformers import LlamaTokenizer
from prompter import wikisql_prompt

random.seed(425)
base_model='../../../share/LLaMA-hf/7B'
# base_model='../../../Data/model/llama-7b-hf'
cutoff_len=200
tokenizer = LlamaTokenizer.from_pretrained(base_model,padding_side="left")
tokenizer.pad_token_id=0

def wiki_random_sampler(fp,size):
    data=list(csv.reader(fp))[1:]
    select_data=random.sample(data,size)
    sample_result=[]
    for line in select_data:
        prompt = wikisql_prompt(line[2].replace('\xa0',' '), line[3], line[6])
        sql=line[12]
        sample_result.append({
                "prompt":prompt,
                "sql":sql
            })

    train_dataset=datasets.Dataset.from_list(sample_result)
    return train_dataset

def wiki_active_sampler(fp,idx_list,size):
    reader=csv.reader(fp)
    reader.__next__()
    select_idx=idx_list[:size]
    select_idx.sort()
    sample_result=[]
    count=0
    for line in reader:
        if int(line[0])==select_idx[count]:
            prompt=wikisql_prompt(line[2].replace('\xa0',' '),line[3],line[6])
            sql = line[12]
            sample_result.append({
                "prompt": prompt,
                "sql": sql
            })
            count+=1
        if count==size:
            break

    train_dataset=datasets.Dataset.from_list(sample_result)
    return train_dataset

def wiki_test_sampler(fp):
    reader=csv.reader(fp)
    reader.__next__()
    sample_result = []
    for line in reader:
        prompt = wikisql_prompt(line[2].replace('\xa0',' '), line[3], line[6])
        sql = line[12]
        sample_result.append({
            "prompt": prompt,
            "sql": sql
        })

    test_dataset = datasets.Dataset.from_list(sample_result)
    return test_dataset

def tokenize(item):
    full_prompt = item["prompt"]+item["sql"]
    result = tokenizer(full_prompt, padding=True, truncation=True, max_length=cutoff_len)
    if len(result["input_ids"]) < cutoff_len:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    input_result = tokenizer(item["prompt"], truncation=True, max_length=cutoff_len)
    input_len = len(input_result["input_ids"])
    result["labels"] = [-100] * input_len + result["input_ids"][input_len:]
    return result

if __name__ == '__main__':
    with open('result/score/wikisql_algo1.csv','r',encoding='utf-8') as f:
        reader=csv.reader(f)
        idx_list=[]
        for line in reader:
            idx_list.append(int(line[0]))
    with open('dataset/wikisql/train.csv','r',encoding='utf-8') as f:
        d=wiki_active_sampler(f,idx_list,10).shuffle().map(tokenize)