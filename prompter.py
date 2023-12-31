# -*- coding: utf-8 -*-            
# @Time : 2023/6/25 18:23
# @Author: LeoN

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

def wikisql_prompt(question,header,table_type):
    header=header[1:-1].replace("'","`")
    table_type=table_type[1:-1].replace("'","`")
    question=question.strip()
    prompt=\
'''There is a table.
The table header:
{}
The type of the table header:
{}
Question: `{}`
If I want to query the problem, write the SQL language.
### the answer is pure SQL in a  line.
Answer:'''\
    .format(header,table_type,question)
    return prompt

def samsum_prompt(dialogue):
    word_list=dialogue.split(' ')
    if len(word_list)>100:
        dialogue=''.join(word_list[:100])
    prompt=\
'''Below is a dialogue:
{}
{}
{}
###Please summary this dialogue in one line briefly.
Summary: '''\
    .format('{',dialogue,'}')
    return prompt

def chanel_prompt(batch_title):
    content = ""
    for i in range(len(batch_title)):
        content += str(i)+"、"+batch_title[i]+"\n"
    SYS = "\n### 请你扮演一个关键词提取专家。我将会提供大量用户有关香奈儿（Chanel）的内容，输出格式上要求关键词之间使用“|”分割，且不包含其它任何内容\n" \
          "关键词："
    prompt = "以下为一批用户评论：\n" + content + SYS
    print(prompt)
    return prompt