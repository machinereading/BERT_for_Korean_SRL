
# coding: utf-8

# In[13]:


import json
import sys
sys.path.insert(0,'../')
import jpype
from konlpy.tag import Kkma
kkma = Kkma()


# In[73]:


def pred_identifier(word):
    jpype.attachThreadToJVM()
    morps = kkma.pos(word)
    v = False
    result = []
    for m,p in morps:
        if p == 'XSV' or p == 'VV':
            v = True

    if v:
        for i in range(len(morps)):
            m,p = morps[i]
            if p == 'VA' or p == 'VV':
                if m[0] == word[0] and len(m) >= 1:
                    result.append(m)
                    break
            if i > 0 and p == 'XSV':
                r = morps[i-1][0]+m
                if r[0] == word[0]:
                    result.append(r)
            
    return result


# In[67]:


def basic_tokenizer(text):
    tokens = text.split(' ')
    idxs = []
    for i in range(len(tokens)):
        idxs.append(str(i))
    return idxs, tokens


# In[91]:


def data2tgt_data(input_data):
    result = []
    for item in input_data:
        ori_tokens, ori_preds = item[0],item[1]
        for idx in range(len(ori_preds)):
            pred = ori_preds[idx]
            if pred != '_':
                if idx == 0:
                    begin = idx
                elif ori_preds[idx-1] == '_':
                    begin = idx
                end = idx
        tokens, preds = [],[]
        for idx in range(len(ori_preds)):
            token = ori_tokens[idx]
            pred = ori_preds[idx]
            if idx == begin:
                tokens.append('<tgt>')
                preds.append('_')

            tokens.append(token)
            preds.append(pred)

            if idx == end:
                tokens.append('</tgt>')
                preds.append('_')
        sent = []
        sent.append(tokens)
        sent.append(preds)
        result.append(sent)
    return result 


# In[90]:


def preprocessing(text):
    result = []
    idxs, tokens = basic_tokenizer(text)
    for idx in range(len(tokens)):
        token = tokens[idx]
        verb_check = pred_identifier(token)
        
        if verb_check:
            preds = ['_' for i in range(len(tokens))]
            preds[idx] = verb_check[0]+'.v'
            instance = []            
#             instance.append(idxs)
            instance.append(tokens)
            instance.append(preds)

            result.append(instance)
    return result


# In[94]:


# text = '애플은 스티브 잡스와 스티브 워즈니악과 론 웨인이 1976년에 설립한 컴퓨터 회사이다.'
# text= '애플은 미국에서 태어난 스티브 잡스가 설립한 컴퓨터 회사이다.'

# d = preprocessing(text)
# print(d)

# z = data2tgt_data(d)
# print(z)


# In[61]:


# text = '설립한'
# d = pred_identifier(text)
# print(d)

