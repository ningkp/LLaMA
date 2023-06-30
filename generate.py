# -*- coding: utf-8 -*-            
# @Time : 2023/6/25 18:24
# @Author: LeoN

import os, sys, csv, fire, torch, argparse
import pandas as pd
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from prompter import Prompter, wikisql_prompt

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=int)
args = parser.parse_args()
device_num=args.device

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

model_weights_path='../../../share/LLaMA-hf/7B'
df=pd.read_csv('dataset/wikisql/train.csv')
prompter=Prompter('alpaca')
cutoff_len = 256
res_path='result'+str(device_num)+'.csv'

def main(
    load_8bit: bool = False,
    base_model: str = model_weights_path,
    lora_weights: str = "",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "127.0.0.1",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model,padding_side="left")

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )


    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half() # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    prompts=[]
    temp_rows=[]
    for index, line in df.iterrows():
        if 28694<=index<=28699 or 35771<=index<35775:
            continue

        prompt = wikisql_prompt(line['question'], line['table.header'], line['table.types'], line['sql.human_readable'])
        prompts.append(prompt)
        row = [line['idx'],prompt]
        temp_rows.append(row)
        if len(prompts)==5:
            input_ids=tokenizer(prompts, return_tensors="pt",padding=True,max_length=cutoff_len).input_ids
            if device == 'cuda':
                input_ids=input_ids.cuda()

            generate_ids = model.generate(input_ids, max_new_tokens=30, num_beams=3, do_sample=True)
            generate_ids = generate_ids[:,input_ids.shape[1]:]
            res=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for i in range(5):
                temp_rows[i].append(res[i])
            with open(res_path,'a',newline='') as f:
                csv.writer(f).writerows(temp_rows)
            torch.cuda.empty_cache()
            prompts=[]
            temp_rows=[]



    # generate_ids=model.generate(inputs.input_ids)
    # print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])


if __name__ == "__main__":
    fire.Fire(main)
