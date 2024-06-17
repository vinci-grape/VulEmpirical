from __future__ import absolute_import, division, print_function

import os
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
If this C code snippet has vulnerabilities, output Yes; otherwise, output No. Please output only Yes or No, no need to output any other characters.
// Code Start
int main(int argc, const char **argv)
{{
DCRaw *d = new DCRaw;
return d->main(argc, argv);
}}
// Code End

// Detection
Yes

If this C code snippet has vulnerabilities, output Yes; otherwise, output No. Please output only Yes or No, no need to output any other characters.
// Code Start
static int jas_iccgetuint64(jas_stream_t *in, jas_iccuint64_t *val)
{{
ulonglong tmp;
if (jas_iccgetuint(in, 8, &tmp))
return -1;
*val = tmp;
return 0;
}}
// Code End

// Detection
No

If this C code snippet has vulnerabilities, output Yes; otherwise, output No. Please output only Yes or No, no need to output any other characters.
// Code Start
{}
// Code End

// Detection
'''


def build_instruction_prompt(code):
    return prompt.format(code.strip()).lstrip() 


def convert_examples_to_features(code, tokenizer, args):
    """convert examples to token ids"""
    input = tokenizer.encode(
        code, 
        add_special_tokens=False,
        truncation=True, 
        max_length=args.max_length,
        return_tensors="pt", 
    )
    code = tokenizer.decode(input[0])
    input_ids = tokenizer.encode(
        build_instruction_prompt(code),
        return_tensors="pt"
    )

    return input_ids


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
        labels.append(batch[2])
        with torch.no_grad():
            while True:
                output = model.generate(input_ids, max_new_tokens=3, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_p=0.95, top_k=50, temperature=1)
                pred = tokenizer.decode(output[0][len(input_ids[0]): ], skip_special_tokens=True).strip()
                if pred[:3] == "Yes" or pred[:3] == "YES":
                    preds.append(1)
                    break
                elif pred[:2] == "No" or pred[:2] == "NO":
                    preds.append(0)
                    break

    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)
    conf_matrix = confusion_matrix(labels, preds)
    tn, fp, _, _ = conf_matrix.ravel()
    false_positive_rate = fp / (fp + tn)

    results = {
        "f1": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "acc": float(acc),
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
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization.")   
    
          
    # Print arguments
    args = parser.parse_args()
    
    # Set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )

    
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
            test_dataset.append((js["index"], convert_examples_to_features(js["code"], tokenizer, args), js["detection_label"]))

    result = evaluate(test_dataset, model, tokenizer)
    logger.info("***** Test results *****")
    for key in result.keys():
        logger.info("  %s = %s", key, str(round(result[key]*100 if "map" in key else result[key],4)))       


if __name__ == "__main__":
    main()