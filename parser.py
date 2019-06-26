
# coding: utf-8

# In[13]:


import json
import sys
sys.path.insert(0,'../')
from konlpy.tag import Kkma

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


from BERT_for_Korean_SRL import bert_srl
from BERT_for_Korean_SRL import preprocessor


# In[3]:


try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'


# In[21]:


class srl_parser():
    
    def __init__(self, model_dir=dir_path+'/model/model.pt', batch_size=1):
        try:
            self.model = torch.load(model_dir)
            self.model.to(device);
            self.model.eval()
        except KeyboardInterrupt:
            raise
        except:
            print('model dir', model_dir, 'is not valid ')
            
        self.bert_io = bert_srl.for_BERT(mode='test')
        self.batch_size = batch_size
        
    def ko_srl_parser(self, text):
        
        input_data = preprocessor.preprocessing(text)        
        input_tgt_data = preprocessor.data2tgt_data(input_data)        
        input_data_bert = self.bert_io.convert_to_bert_input(input_tgt_data)        
        input_dataloader = DataLoader(input_data_bert, batch_size=self.batch_size)
        
        pred_args = []
        for batch in input_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_orig_tok_to_maps, b_input_masks = batch
            
            with torch.no_grad():
                logits = self.model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_masks)
            logits = logits.detach().cpu().numpy()
            b_pred_args = [list(p) for p in np.argmax(logits, axis=2)]
            
            for b_idx in range(len(b_pred_args)):
                
                input_id = b_input_ids[b_idx]
                orig_tok_to_map = b_input_orig_tok_to_maps[b_idx]                
                pred_arg_bert = b_pred_args[b_idx]

                pred_arg = []
                for tok_idx in orig_tok_to_map:
                    if tok_idx != -1:
                        tok_id = int(input_id[tok_idx])
                        if tok_id == 1:
                            pass
                        elif tok_id == 2:
                            pass
                        else:
                            pred_arg.append(pred_arg_bert[tok_idx])                            
                pred_args.append(pred_arg)
                
        pred_arg_tags_old = [[self.bert_io.idx2tag[p_i] for p_i in p] for p in pred_args]
        
        result = []
        for b_idx in range(len(pred_arg_tags_old)):
            pred_arg_tag_old = pred_arg_tags_old[b_idx]
            pred_arg_tag = []
            for t in pred_arg_tag_old:
                if t == 'X':
                    new_t = 'O'
                else:
                    new_t = t
                pred_arg_tag.append(new_t)
                
            instance = []
            instance.append(input_data[b_idx][0])
            instance.append(input_data[b_idx][1])
            instance.append(pred_arg_tag)
            
            result.append(instance)
        
        return result
            


# In[22]:


# p = srl_parser(model_dir='/disk_4/BERT_for_Korean_SRL/models/ko-srl-epoch-4.pt')


# In[23]:


# text = '애플은 미국에서 태어난 스티브 잡스가 설립한 컴퓨터 회사이다.'

# d = p.ko_srl_parser(text)
# print(d)

