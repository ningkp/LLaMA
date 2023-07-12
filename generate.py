# -*- coding: utf-8 -*-            
# @Time : 2023/6/25 18:24
# @Author: LeoN

import os, sys, csv, fire, torch, argparse
import pandas as pd
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from prompter import wikisql_prompt, samsum_prompt, chanel_prompt
from tqdm import tqdm

def print_log(log):
    print("*" * 20)
    print(log)
    print("*" * 20)

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str, default="0")
parser.add_argument('--model',type=str, default="LLaMA13B")
parser.add_argument('--dataset',type=str, default="Chanel")
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

if args.model == "LLaMA7B":
    model_weights_path='../../share/LLaMA-hf/7B'
elif args.model == "LLaMA13B":
    model_weights_path='../../share/LLaMA-hf/13B'
cutoff_len = 256
print_log("loading dataset...")

if args.dataset == "SAMSum":
    with open('dataset/SAMSum/train.csv','r',encoding='utf-8') as f:
        rows=list(csv.reader(f))
    res_path = 'result/generate/samsum/result.csv'
elif args.dataset == "Chanel":
    table = pd.read_excel('dataset/Chanel/chanel_new.xlsx', keep_default_na=False)
    print(table)
    res_path = 'result/generate/Chanel/result.csv'
    # print(ca)

def main(
    load_8bit: bool = False,
    base_model: str = model_weights_path,
    lora_weights: str = "",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "127.0.0.1",  # Allows to listen on all interfaces by providing 0.
    share_gradio: bool = False,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

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

    print_log("loading model...")
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )

    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )

    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
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

    result=[]
    prompts=[]
    temp_rows=[]
    print_log("inference...")
    if args.dataset == "Chanel":
        comments = table["标题"][:1000]
        batch = 4
        for i in range(0, len(comments), batch):
            prompt = chanel_prompt(comments[i:i+batch])
            input_ids = tokenizer(prompt, return_tensors="pt", padding=True, max_length=cutoff_len).input_ids
            if device == 'cuda':
                input_ids = input_ids.cuda()

            generate_ids = model.generate(input_ids, max_new_tokens=100, num_beams=1, do_sample=True)
            generate_ids = generate_ids[:, input_ids.shape[1]:]
            res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print(res)
            result.append(res)
            torch.cuda.empty_cache()

    elif args.dataset == "SAMSum":
        for idx,dialogue,summary in tqdm(rows):
            prompt = samsum_prompt(dialogue)
            prompts.append(prompt)
            row =[idx,prompt]
            temp_rows.append(row)
            if len(prompts)==5:
                input_ids=tokenizer(prompts, return_tensors="pt",padding=True,max_length=cutoff_len).input_ids
                if device == 'cuda':
                    input_ids=input_ids.cuda()

                generate_ids = model.generate(input_ids, max_new_tokens=40, num_beams=1, do_sample=True)
                generate_ids = generate_ids[:,input_ids.shape[1]:]
                res=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for i in range(5):
                    temp_rows[i].append(res[i])
                    result.append(temp_rows[i])
                torch.cuda.empty_cache()
                prompts=[]
                temp_rows=[]
    with open(res_path,mode='w',encoding='utf-8',newline='') as fp:
        csv.writer(fp).writerows(result)


if __name__ == "__main__":
    main()
