from __future__ import absolute_import, division, print_function

import os
import re
import json
import random
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from transformers import (AutoTokenizer, AutoModelForCausalLM)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


no_deprecation_warning=True
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


prompt = '''
I will provide you a vulnerable C code snippet and please locate vulnerable lines.
// Code Start
void WaitForCallback() {{
if (!use_audio_thread_) {{
base::RunLoop().RunUntilIdle();
return;
}}
media::WaitableMessageLoopEvent event;
audio_thread_.task_runner()->PostTaskAndReply(
FROM_HERE, base::Bind(&base::DoNothing), event.GetClosure());
event.RunAndWait();
base::RunLoop().RunUntilIdle();
}}
// Code End

// Location
[5]

I will provide you a vulnerable C code snippet and please locate vulnerable lines.
// Code Start
static void build_l4proto_dccp(const struct nf_conntrack *ct, struct nethdr *n)
{{
ct_build_group(ct, ATTR_GRP_ORIG_PORT, n, NTA_PORT,
sizeof(struct nfct_attr_grp_port));
if (!nfct_attr_is_set(ct, ATTR_DCCP_STATE))
return;
ct_build_u8(ct, ATTR_DCCP_STATE, n, NTA_DCCP_STATE);
ct_build_u8(ct, ATTR_DCCP_ROLE, n, NTA_DCCP_ROLE);
}}
// Code End

// Location
[2, 3]

I will provide you a vulnerable C code snippet and please locate vulnerable lines.
// Code Start
{}
// Code End

// Location
'''


def build_instruction_prompt(code):
    return prompt.format(code.strip()).lstrip() 


def convert_examples_to_features(code, tokenizer, args):
    """convert examples to token ids"""
    input_ids = tokenizer.encode(
        code, 
        add_special_tokens=False,
        truncation=True,
        max_length=args.max_length, 
        return_tensors="pt"
    )
    decoded_code = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    nl_len = decoded_code.count('\n') + 1

    input_ids = tokenizer.encode(
        build_instruction_prompt(decoded_code), 
        return_tensors="pt"
    )

    return input_ids, nl_len


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(test_dataset, model, tokenizer):
    """ Evaluate the model """

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(test_dataset))

    model.eval()
    labels = []
    preds = []

    for batch in tqdm(test_dataset, total=len(test_dataset)):
        input_ids = batch[1].to(model.device)
        nl_len = batch[2]
        label = batch[3]

        tensor = torch.zeros(max(nl_len, label[-1]+1), dtype=torch.int64)
        tensor[label] = True
        label = tensor
    
        with torch.no_grad():
            while True:
                output = model.generate(input_ids, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_p=0.95, top_k=50, temperature=1)
                pred = tokenizer.decode(output[0][len(input_ids[0]): ], skip_special_tokens=True).strip()                
                pattern = r'\[\s*(\d+\s*(?:,\s*\d+\s*)*)'
                match = re.match(pattern, pred)
                if match:
                    numbers = match.group(1).split(',')
                    pred = [int(num) for num in numbers]
                    pred.sort()
                    tensor = torch.zeros(max(nl_len, pred[-1]+1), dtype=torch.int64)
                    tensor[pred] = True
                    pred = tensor
                    break
        preds.append(pred[:nl_len])
        labels.append(label[:nl_len])

    preds = np.concatenate(preds,0)
    labels = np.concatenate(labels,0)

    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)
    conf_matrix = confusion_matrix(labels, preds)
    tn, fp, _, _ = conf_matrix.ravel()
    false_positive_rate = fp / (fp + tn)

    results = {
        "acc": float(acc),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "fpr": float(false_positive_rate)
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--max_length", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--nl_ids", type=int, default=185,
                        help="")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization.")   
    
          
    # Print arguments
    args = parser.parse_args()
    
    # Set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )

    
    # Set seed
    set_seed(args.seed)
    
    # Build model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)

    if tokenizer.pad_token == None: 
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')

    test_dataset = []
    with open(args.test_data_file) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            if js["detection_label"] == 1 and js["location_label"] != []:
                input_ids, nl_len = convert_examples_to_features(js["func_before"], tokenizer, args)
                test_dataset.append((js["index"], input_ids, nl_len, js["location_label"]))

    result = evaluate(test_dataset, model, tokenizer)
    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key]*100 if "map" in key else result[key],4)))       


if __name__ == "__main__":
    main()