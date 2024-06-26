from __future__ import absolute_import, division, print_function

import os
import json
import random
import torch
import logging
import argparse
import numpy as np
import sys
sys.path.append("..")
from tqdm import tqdm
from model import Model
from torch.optim import AdamW
from Utils.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, PhiForSequenceClassification)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


no_deprecation_warning=True
logger = logging.getLogger(__name__)
early_stopping = EarlyStopping()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['STARCODER_AUTH_TOKEN'] = "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"


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_ids,
                 attention_mask,
                 nl_indices,
                 label,
                 index
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.nl_indices = nl_indices
        self.label = label
        self.index = index


def convert_examples_to_features(js, tokenizer, args):
    """convert examples to token ids"""
    source = tokenizer.encode_plus(
        js['func_before'] + '\n',
        add_special_tokens=True,
        padding='max_length', 
        truncation=True, 
        max_length=args.max_length, 
        return_tensors="pt"
    )

    input_ids = source["input_ids"]
    attention_mask = source["attention_mask"]
    nl_indices = torch.where(input_ids == args.nl_ids)[1].numpy()

    return InputFeatures(input_ids, attention_mask, nl_indices, js['location_label'], js['index'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                data.append(js)
        for js in data:
            if js["detection_label"] == 1 and js["location_label"] != []:
                self.examples.append(convert_examples_to_features(js,tokenizer,args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        input_ids = self.examples[i].input_ids
        attention_mask = self.examples[i].attention_mask
        nl_indices = self.examples[i].nl_indices
        label = self.examples[i].label
        index = self.examples[i].index
        return (input_ids, attention_mask, torch.tensor(nl_indices), torch.tensor(label), index)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataloader, eval_dataloader, model):
    """ Train the model """    
    args.max_steps = args.num_train_epochs * len(train_dataloader)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu )
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", args.max_steps)

    losses, best_f1 = [], 0
    
    model.zero_grad()
    for idx in range(args.num_train_epochs): 
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input_ids = batch[0].to(args.device).squeeze(1)
            attention_mask = batch[1].to(args.device).squeeze(1)
            nl_indices = batch[2].to(args.device).squeeze(0)
            labels = batch[3].to(args.device)

            loss, _, _ = model(input_ids, attention_mask, nl_indices, labels)
            
            if args.n_gpu > 1:
                loss = loss.mean()
                                
            loss.backward()       
                    
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    
            losses.append(loss.item())

            if (step+1)% 100==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(np.mean(losses[-100:]),4)))

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        results, eval_loss = evaluate(args, eval_dataloader, model)
        
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))                    
        
        if results['f1'] > best_f1:
            best_f1 = results['f1']
            logger.info("  "+"*"*20)  
            logger.info("  Best f1:%s",round(best_f1,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-f1'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            output_dir = os.path.join(output_dir, 'model.bin')
            model_to_save = model.module if hasattr(model,'module') else model
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

        early_stopping(eval_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def evaluate(args, eval_dataloader, model):
    """ Evaluate the model """

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    probs=[]
    labels=[]

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        input_ids = batch[0].to(args.device).squeeze(1)
        attention_mask = batch[1].to(args.device).squeeze(1)
        nl_indices = batch[2].to(args.device).squeeze(0)
        label = batch[3].to(args.device)
        
        with torch.no_grad():
            lm_loss, prob, label = model(input_ids, attention_mask, nl_indices, label)
            eval_loss += lm_loss.mean().item()
            probs.append(prob.detach().to(torch.float).cpu().numpy())
            labels.append(label.detach().to(torch.float).cpu().numpy())
        nb_eval_steps += 1

    probs = np.concatenate(probs,0)
    labels = np.concatenate(labels,0)
    preds = probs[:, 1] > 0.5
            
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)
    
    conf_matrix = confusion_matrix(labels, preds)
    tn, fp, fn, tp = conf_matrix.ravel()
    false_positive_rate = fp / (fp + tn)

    results = {
        "f1": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "acc": float(acc),
        "fpr": float(false_positive_rate),
    }
    return results, eval_loss


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a jsonl file).")    
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--max_length", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test on the dev set.")     
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    if "models--microsoft--phi-2" in args.model_name_or_path:
        model = PhiForSequenceClassification.from_pretrained(args.model_name_or_path, token=os.environ['STARCODER_AUTH_TOKEN'], trust_remote_code=True, output_hidden_states=True, torch_dtype=torch.bfloat16, device_map='auto')
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, token=os.environ['STARCODER_AUTH_TOKEN'], trust_remote_code=True, output_hidden_states=True, torch_dtype=torch.bfloat16, device_map='auto')

    model.config.pad_token_id = model.config.eos_token_id

    # Set device
    device = model.device
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    model = Model(model, config, args)
    
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)
        
        eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

        train(args, train_dataloader, eval_dataloader, model)
        
    # Testing
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir, map_location='cuda'))     

        test_dataset = TextDataset(tokenizer, args, args.test_data_file)    
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=6, pin_memory=True)

        result, _ = evaluate(args, test_dataloader, model)
        logger.info("***** Test results *****")
        for key in (result.keys()):
            logger.info("  %s = %s", key, str(round(result[key]*100 if "map" in key else result[key],4)))       
    

if __name__ == "__main__":
    main()