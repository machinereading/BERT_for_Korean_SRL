
# coding: utf-8

# In[1]:


import json


# In[70]:


def raw2tagseq(data):
    result = []
    
    conll = []
    for line in data:
        line = line.strip()
        if line != '':
            if line.startswith(';'):
                tokens, all_preds = [],[]               
            else:
                conll.append(line)
                token = line.split('\t')[1]
                pred = line.split('\t')[13]
                tokens.append(token)
                all_preds.append(pred)
        else:            
            p_ids = []
            for p_id in range(len(all_preds)):
                p = all_preds[p_id]
                if p != '_':
                    p_ids.append(p_id)
            for i in range(len(p_ids)):
                p_id = p_ids[i]                
                preds = ['_' for i in range(len(all_preds))]
                preds[p_id] = all_preds[p_id]                
                sent = []
                args = []
                for l in conll:
                    l = l.strip()
                    arg = l.split('\t')[14+i]
                    if arg == '_':
                        arg = 'O'
                    args.append(arg)                    
                sent.append(tokens)
                sent.append(preds)
                sent.append(args)
                result.append(sent)
            conll = []
    return result


# In[71]:


def load_srl_data():
    with open('./data/srl.train.formatted.conll') as f:
        d = f.readlines()    
    trn = raw2tagseq(d)
    with open('./data/srl.test.formatted.conll') as f:
        d = f.readlines()    
    tst = raw2tagseq(d)
    
    return trn, tst


# In[79]:


def data2tgt_data(input_data):
    result = []
    for item in input_data:
        ori_tokens, ori_preds, ori_args = item[0],item[1],item[2]
        for idx in range(len(ori_preds)):
            pred = ori_preds[idx]
            if pred != '_':
                if idx == 0:
                    begin = idx
                elif ori_preds[idx-1] == '_':
                    begin = idx
                end = idx
        tokens, preds, args = [],[],[]
        for idx in range(len(ori_preds)):
            token = ori_tokens[idx]
            pred = ori_preds[idx]
            arg = ori_args[idx]
            if idx == begin:
                tokens.append('<tgt>')
                preds.append('_')
                args.append('X')
                
            tokens.append(token)
            preds.append(pred)
            args.append(arg)
            
            if idx == end:
                tokens.append('</tgt>')
                preds.append('_')
                args.append('X')
        sent = []
        sent.append(tokens)
        sent.append(preds)
        sent.append(args)
        result.append(sent)
    return result 


# In[80]:


def load_srl_data_for_bert():
    trn_ori, tst_ori = load_srl_data()
    
    trn = data2tgt_data(trn_ori)
    tst = data2tgt_data(tst_ori)
    
    return trn, tst

