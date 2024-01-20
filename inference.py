from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
from tqdm import tqdm
import torch
import pandas as pd
import re
import csv
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.distributed
import transformers
import pandas as pd
from datasets import Dataset
import re
import subprocess
import ast

os.environ['STARCODER_AUTH_TOKEN'] = ""


prompt = '''
Provide qualitative severity ratings of CVSS v2.0 for the vulnerable C code snippet. Please output only High or Medium or Low, no need to output any other characters.
// Code Start
{}
// Code End

// Assessment
'''

def build_instruction_prompt(source, task):
    if task == "fewshot":
        with open(f"./fewshot/prompt_assessment.txt", 'r') as file:
            prompt_ = file.read()
        prompt_ = prompt_.format(function=source)
        return prompt_
    else:
        return prompt.format(source.strip()).lstrip() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model to use for vulnerability task.', required=True, type=str)
    parser.add_argument('--task', help='model to use for vulnerability task. should be one of [fewshot, finetune]', type=str, default="finetune")
    
    args = parser.parse_args()
    
    if args.task == "fewshot":
        if args.model == "codellama":
            model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf", trust_remote_code=True, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf", trust_remote_code=True)
        elif args.model == "codellama_34b":
            model = AutoModelForCausalLM.from_pretrained("../../codellm/CodeLlama-34b-Instruct-hf", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained("../../codellm/CodeLlama-34b-Instruct-hf", trust_remote_code=True)
        elif args.model == "deepseek":
            model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
        elif args.model == "deepseek_33b":
            model = AutoModelForCausalLM.from_pretrained("../../codellm/deepseek-coder-33b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained("../../codellm/deepseek-coder-33b-instruct", trust_remote_code=True)
        elif args.model == "starcoder":
            model = AutoModelForCausalLM.from_pretrained("bigcode/starcoderbase-7b", trust_remote_code=True, device_map='auto', token=os.environ['STARCODER_AUTH_TOKEN'])
            tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-7b", trust_remote_code=True, token=os.environ['STARCODER_AUTH_TOKEN'])
        elif args.model == "starcoder_15b":
            model = AutoModelForCausalLM.from_pretrained('bigcode/starcoder', use_auth_token=os.environ['STARCODER_AUTH_TOKEN'], trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained('bigcode/starcoder', use_auth_token=os.environ['STARCODER_AUTH_TOKEN'])
        elif args.model == "mistral":
            model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", trust_remote_code=True, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", trust_remote_code=True)
        elif args.model == "wizardcoder":
            model = AutoModelForCausalLM.from_pretrained("WizardLM/WizardCoder-Python-7B-V1.0", trust_remote_code=True, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained("WizardLM/WizardCoder-Python-7B-V1.0", trust_remote_code=True)
        elif args.model == "wizardcoder_34b":
            model = AutoModelForCausalLM.from_pretrained("WizardLM/WizardCoder-Python-34B-V1.0", trust_remote_code=True, device_map='auto', torch_dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained("WizardLM/WizardCoder-Python-34B-V1.0", trust_remote_code=True)
        elif args.model == "phi":
            model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    else:
        if args.model == "codellama":
            model = AutoModelForCausalLM.from_pretrained("./output/assessment/CodeLlama-7b-Instruct-hf",  torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf", trust_remote_code=True)
        elif args.model == "deepseek":
            model = AutoModelForCausalLM.from_pretrained("./output/assessment/deepseek-coder-6.7b-instruct", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained("./output/assessment/deepseek-coder-6.7b-instruct", trust_remote_code=True)
        elif args.model == "phi":
            model = AutoModelForCausalLM.from_pretrained("./output/assessment/phi-2", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        elif args.model == "mistral":
            model = AutoModelForCausalLM.from_pretrained("./output/assessment/Mistral-7B-Instruct-v0.1", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", trust_remote_code=True)
        elif args.model == "wizardcoder":
            model = AutoModelForCausalLM.from_pretrained("./output/assessment/WizardCoder-Python-7B-V1.0", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained("./output/assessment/WizardCoder-Python-7B-V1.0", trust_remote_code=True)
        elif args.model == "starcoder":
            model = AutoModelForCausalLM.from_pretrained("./output/assessment/starcoderbase-7b", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto', token=os.environ['STARCODER_AUTH_TOKEN'])
            tokenizer = AutoTokenizer.from_pretrained("./output/assessment/starcoderbase-7b", trust_remote_code=True, token=os.environ['STARCODER_AUTH_TOKEN'])

    if tokenizer.pad_token == None: 
        tokenizer.pad_token = tokenizer.eos_token
            
    with open("./dataset/Big-Vul/test.jsonl", 'r') as file:
        lines = file.readlines()
        
    df = pd.DataFrame([eval(line) for line in lines])
    preds = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating predictions"):
        code = row["code"]
        if row["assessment_label"] == "" or row["label"] == 0: 
            preds.append("")
            continue
        input = tokenizer.encode(code, return_tensors="pt", padding=True, truncation=True, max_length=1024, add_special_tokens=False).to(model.device)
        code = tokenizer.decode(input[0])
        input_text = build_instruction_prompt(code, args.task)
        input = tokenizer.encode(input_text, return_tensors="pt", padding=True).to(model.device)
        
        while True:
            output = model.generate(input, pad_token_id=tokenizer.eos_token_id, max_new_tokens=3, do_sample=True, top_p=0.95, top_k=50, temperature=1)
            pred = tokenizer.decode(output[0][len(input[0]): ], skip_special_tokens=True).strip()
            if pred[:4] == "High":
                preds.append("High")
                break
            elif pred[:6] == "Medium":
                preds.append("Medium")
                break
            elif pred[:3] == "Low":
                preds.append("Low")
                break
            print(pred)

    df_preds = pd.DataFrame({'assessment': preds})
    df_preds.to_csv(f'./result/assessment/{args.model}_{args.task}_assessment.csv', index=False)