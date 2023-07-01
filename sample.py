# -*- coding: utf-8 -*-            
# @Time : 2023/7/1 23:11
# @Author: LeoN

import csv,random

import datasets
from transformers import LlamaTokenizer
from prompter import wikisql_prompt

random.seed(425)
base_model='../../../share/LLaMA-hf/7B'
cutoff_len=200
tokenizer = LlamaTokenizer.from_pretrained(base_model,padding_side="left")
tokenizer.pad_token_id=0

def wiki_random_sampler(fp,size):
    data=list(csv.reader(fp))
    select_data=random.sample(data,size)
    sample_result=[]
    for line in select_data:
        prompt = wikisql_prompt(line[2], line[3], line[6])
        sql=line[12]
        sample_result.append({
                "prompt":prompt,
                "sql":sql
            })

    train_dataset=datasets.Dataset.from_list(sample_result)
    return train_dataset

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
    with open('dataset/wikisql/train.csv','r',encoding='utf-8') as f:
        d=wiki_random_sampler(f,10).shuffle().map(tokenize)
        print(d.__dict__)