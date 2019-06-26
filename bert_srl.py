
# coding: utf-8

# In[1]:


import json
import dataio
import sys
sys.path.append('../')

import numpy as np

import torch
from torch import nn
from torch.optim import Adam
import glob
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertModel
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from tqdm import tqdm, trange
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score
from pprint import pprint
from datetime import datetime
start_time = datetime.now()


# In[2]:


MAX_LEN = 256
batch_size = 6

try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'


# In[2]:


def load_data():
    trn, tst = dataio.load_srl_data_for_bert()
    print('trn:', len(trn))
    print('tst:', len(tst))
    print('data example')
    print(trn[0])
    
    return trn, tst


# In[5]:


class for_BERT():
    
    def __init__(self, mode='training'):
        self.mode = mode
        
        with open(dir_path+'/data/tag2idx.json','r') as f:
            self.tag2idx = json.load(f)
            
        self.idx2tag = dict(zip(self.tag2idx.values(),self.tag2idx.keys()))
        
        # load pretrained BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        
        # load BERT tokenizer with untokenizing frames
        never_split_tuple = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
        added_never_split = []
        added_never_split.append('<tgt>')
        added_never_split.append('</tgt>')
        added_never_split_tuple = tuple(added_never_split)
        never_split_tuple += added_never_split_tuple
        vocab_file_path = dir_path+'/data/bert-multilingual-cased-dict-add-frames'
        self.tokenizer_with_frame = BertTokenizer(vocab_file_path, do_lower_case=False, max_len=256, never_split=never_split_tuple)
        
    def idx2tag(self, predictions):
        pred_tags = [self.idx2tag[p_i] for p in predictions for p_i in p]
        
        # bert tokenizer and assign to the first token
    def bert_tokenizer(self, text):
        orig_tokens = text.split(' ')
        bert_tokens = []
        orig_to_tok_map = []
        bert_tokens.append("[CLS]")
        for orig_token in orig_tokens:
            orig_to_tok_map.append(len(bert_tokens))
            bert_tokens.extend(self.tokenizer_with_frame.tokenize(orig_token))
        bert_tokens.append("[SEP]")

        return orig_tokens, bert_tokens, orig_to_tok_map
    
    def convert_to_bert_input(self, input_data):
        tokenized_texts, args = [],[]
        orig_tok_to_maps = []
        for i in range(len(input_data)):    
            data = input_data[i]
            text = ' '.join(data[0])
            orig_tokens, bert_tokens, orig_to_tok_map = self.bert_tokenizer(text)
            orig_tok_to_maps.append(orig_to_tok_map)
            tokenized_texts.append(bert_tokens)

            if self.mode == 'training':
                ori_args = data[2]
                arg_sequence = []
                for i in range(len(bert_tokens)):
                    if i in orig_to_tok_map:
                        idx = orig_to_tok_map.index(i)
                        ar = ori_args[idx]
                        arg_sequence.append(ar)
                    else:
                        arg_sequence.append('X')
                args.append(arg_sequence)

        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        orig_tok_to_maps = pad_sequences(orig_tok_to_maps, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post", value=-1)
        
        if self.mode =='training':
            arg_ids = pad_sequences([[self.tag2idx.get(ar) for ar in arg] for arg in args],
                                    maxlen=MAX_LEN, value=self.tag2idx["X"], padding="post",
                                    dtype="long", truncating="post")

        attention_masks = [[float(i>0) for i in ii] for ii in input_ids]    
        data_inputs = torch.tensor(input_ids)
        data_orig_tok_to_maps = torch.tensor(orig_tok_to_maps)
        data_masks = torch.tensor(attention_masks)
        
        if self.mode == 'training':
            data_args = torch.tensor(arg_ids)
            bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_args, data_masks)
        else:
            bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_masks)
        return bert_inputs


# In[6]:


# bert_io = for_BERT(mode='training')


# In[7]:


# trn_data = bert_io.convert_to_bert_input(trn)
# tst_data = bert_io.convert_to_bert_input(tst)


# In[10]:


def train():
    model_path = dir_path+'/models/'
    print('your model would be saved at', model_path)
    
    model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(bert_io.tag2idx))
    model.to(device);
    
    trn_data = bert_io.convert_to_bert_input(trn)
    sampler = RandomSampler(trn_data)
    trn_dataloader = DataLoader(trn_data, sampler=sampler, batch_size=batch_size)
    
    # load optimizer
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters()) 
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)
    
    
    # train 
    epochs = 10
    max_grad_norm = 1.0
    num_of_epoch = 0
    for _ in trange(epochs, desc="Epoch"):
        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(trn_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_orig_tok_to_maps, b_input_args, b_input_masks = batch            
            # forward pass
            loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_masks, labels=b_input_args)
            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            model.zero_grad()
#             break
#         break

        # print train loss per epoch
        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        model_saved_path = model_path+'ko-srl-epoch-'+str(num_of_epoch)+'.pt'        
        torch.save(model, model_saved_path)
        num_of_epoch += 1
    print('...training is done')


# In[11]:


# train()


# In[55]:


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def test():
    model_path = dir_path+'/models/'
    models = glob.glob(model_path+'*.pt')
    
    result_path = dir_path+'/results/'
    results = []
    
    for m in models:
        print('model:', m)
        model = torch.load(m)
        model.eval()
        
        tst_data = bert_io.convert_to_bert_input(tst)
        sampler = RandomSampler(tst_data)
        tst_dataloader = DataLoader(tst_data, sampler=sampler, batch_size=batch_size)
        
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        
        pred_args, true_args = [],[]
        for batch in tst_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_orig_tok_to_maps, b_input_args, b_input_masks = batch
            
            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                              attention_mask=b_input_masks, labels=b_input_args)
                logits = model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_masks)
                
            logits = logits.detach().cpu().numpy()
            
            b_pred_args = [list(p) for p in np.argmax(logits, axis=2)]
            b_true_args = b_input_args.to('cpu').numpy().tolist()
            
            
            eval_loss += tmp_eval_loss.mean().item()
            
            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
            
            for b_idx in range(len(b_true_args)):
                
                input_id = b_input_ids[b_idx]
                orig_tok_to_map = b_input_orig_tok_to_maps[b_idx]                
                pred_arg_bert = b_pred_args[b_idx]
                true_arg_bert = b_true_args[b_idx]

                pred_arg, true_arg = [],[]
                for tok_idx in orig_tok_to_map:
                    if tok_idx != -1:
                        tok_id = int(input_id[tok_idx])
                        if tok_id == 1:
                            pass
                        elif tok_id == 2:
                            pass
                        else:
                            pred_arg.append(pred_arg_bert[tok_idx])
                            true_arg.append(true_arg_bert[tok_idx])
                            
                pred_args.append(pred_arg)
                true_args.append(true_arg) 
            
#             break

        
        pred_arg_tags_old = [[bert_io.idx2tag[p_i] for p_i in p] for p in pred_args]
        
        pred_arg_tags = []
        for old in pred_arg_tags_old:
            new = []
            for t in old:
                if t == 'X':
                    new_t = 'O'
                else:
                    new_t = t
                new.append(new_t)
            pred_arg_tags.append(new)
            
        valid_arg_tags = [[bert_io.idx2tag[v_i] for v_i in v] for v in true_args]
        f1 = f1_score(pred_arg_tags, valid_arg_tags)
                
        print("Validation loss: {}".format(eval_loss/nb_eval_steps))
        print("Validation F1-Score: {}".format(f1_score(pred_arg_tags, valid_arg_tags)))
                
        result =  m+'\targid:'+str(f1)+'\n'
        results.append(result)
        
        epoch = m.split('-')[-1].split('.')[0]
        fname = result_path+str(epoch)+'-result.txt'
        
        with open(fname, 'w') as f:
            line = result
            f.write(line)
            line = 'gold'+'\t'+'pred'+'\n'
            f.write(line)
            
            for r in range(len(pred_arg_tags)):
                line = str(valid_arg_tags[r]) + '\t' + str(pred_arg_tags[r])+'\n'
                f.write(line)
                
    fname = result_path+'result.txt'
    with open(fname, 'w') as f:
        for r in results:
            f.write(r)
            
    print('result is written to',fname)


# In[56]:


# test()

