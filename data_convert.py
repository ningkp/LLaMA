# -*- coding: utf-8 -*-            
# @Time : 2023/6/28 16:37
# @Author: LeoN

from fastparquet import ParquetFile
import pandas as pd
from prompter import wikisql_prompt

# pf=ParquetFile('wikisql-test.parquet')
# df=pf.to_pandas()
# df.to_csv('dataset/wikisql/test.csv',encoding='utf-8')
df=pd.read_csv('dataset/wikisql/train.csv')
# print(df.columns)
for index,line in df.iterrows():
    # print(line['question'],line['table.header'],line['table.types'],line['sql.human_readable'])
    prompt=wikisql_prompt(line['question'],line['table.header'],line['table.types'],line['sql.human_readable'])
    print(prompt)
    break