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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix


no_deprecation_warning=True
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


prompt = '''
Provide qualitative severity ratings of CVSS v2.0 for the vulnerable C code snippet. Please output only High or Medium or Low, no need to output any other characters.
// Code Start
iakerb_gss_export_sec_context(OM_uint32 *minor_status,
gss_ctx_id_t *context_handle,
gss_buffer_t interprocess_token)
{{
OM_uint32 maj;
iakerb_ctx_id_t ctx = (iakerb_ctx_id_t)context_handle;
if (!ctx->established)
return GSS_S_UNAVAILABLE;
maj = krb5_gss_export_sec_context(minor_status, &ctx->gssc,
interprocess_token);
if (ctx->gssc == GSS_C_NO_CONTEXT) {{
iakerb_release_context(ctx);
*context_handle = GSS_C_NO_CONTEXT;
}}
return maj;
}}
// Code End

// Assessment
High

Provide qualitative severity ratings of CVSS v2.0 for the vulnerable C code snippet. Please output only High or Medium or Low, no need to output any other characters.
// Code Start
void SendStatus(struct mg_connection* connection,
const struct mg_request_info* request_info,
void* user_data) {{
std::string response = "HTTP/1.1 200 OK\r\n"
"Content-Length:2\r\n\r\n"
"ok";
mg_write(connection, response.data(), response.length());
}}
// Code End

// Assessment
Medium

Provide qualitative severity ratings of CVSS v2.0 for the vulnerable C code snippet. Please output only High or Medium or Low, no need to output any other characters.
// Code Start
MagickExport int LocaleUppercase(const int c)
{{
if (c < 0)
return(c);
#if defined(MAGICKCORE_LOCALE_SUPPORT)
if (c_locale != (locale_t) NULL)
return(toupper_l((int) ((unsigned char) c),c_locale));
#endif
return(toupper((int) ((unsigned char) c)));
}}
// Code End

// Assessment
Low

Provide qualitative severity ratings of CVSS v2.0 for the vulnerable C code snippet. Please output only High or Medium or Low, no need to output any other characters.
// Code Start
{}
// Code End

// Assessment
'''

def number_to_rating(number):
    if number <= 3.9 and number >= 0.0:
        return 0
    elif number <= 6.9:
        return 1
    else:
        return 2
    

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
                pred = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True).strip()
                if pred[:4] == "High":
                    preds.append(2)
                    break
                elif pred[:6] == "Medium":
                    preds.append(1)
                    break
                elif pred[:3] == "Low":
                    preds.append(0)
                    break
    
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')

    results = {
        "f1": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "acc": float(acc),
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
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )

    
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
            if js["detection_label"] == 1 and js["assessment_label"] != "":
                test_dataset.append((js["index"], convert_examples_to_features(js["func_before"], tokenizer, args), number_to_rating(js["assessment_label"])))

    result = evaluate(test_dataset, model, tokenizer)
    logger.info("***** Test results *****")
    for key in result.keys():
        logger.info("  %s = %s", key, str(round(result[key]*100 if "map" in key else result[key],4)))       


if __name__ == "__main__":
    main()