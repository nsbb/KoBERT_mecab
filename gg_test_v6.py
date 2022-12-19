import sys
import os
import pandas as pd
from konlpy.tag import Mecab
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import numpy
from tqdm.notebook import tqdm
from termcolor import colored
from time import time
from time import ctime
import subprocess

subprocess.call(['sh','/toy/logo.sh'])
print(colored("LG_NLP_Project...\nModel : koBERT + mecab\nLET's GO!!!\n",'cyan',attrs=['bold','blink']))

tt = ctime(time())
max_len = 64
batch_size = 64
 
bertmodel,vocab = get_pytorch_kobert_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#BERT Dataset class
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=44,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer,vocab,lower=False)

def new_softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = (exp_a/sum_exp_a) * 100
    return np.round(y, 3)

def predict(predict_sentence,state_path):
    model = torch.load('/toy/LG_model/kobert_model.pt',map_location=device)
    state='/toy/LG_model/state2/'+state_path
    model.load_state_dict(torch.load(state,map_location=device))

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)
#print(out)

        test_eval=[]
        for i in out:
            logits = i
#print(i,logits)
            logits = logits.detach().cpu().numpy()
            min_v = min(logits)
            total = 0
            probability = []
            logits = np.round(new_softmax(logits), 3).tolist()
            for logit in logits:
#print(logit)
                probability.append(np.round(logit, 3)) 
        res = np.argmax(probability)
#print(res)
    return res

def main():
    path = '/toy/LG_data/'
    file_list = os.listdir(path)
    d_test , d_result , d_target , d_target_num = [], [], [], []
    for file in file_list:
        xl = pd.ExcelFile(path+file)
        t_test=[]
        t_result=[]
        for i in range(1,len(xl.sheet_names)):
            t_test = xl.parse(xl.sheet_names[i])['개발자 TESTcase']
            t_result = xl.parse(xl.sheet_names[i])['개발자Result']
            for j in range(len(t_test)):
                if type(t_test[j])==str and type(t_result[j])==str:
                    d_test.append(t_test[j])
                    d_result.append(t_result[j])
                    d_target.append(xl.sheet_names[i])
    tar = list(set(d_target))
    tar.sort()
    for i in d_target:
        for j in range(len(tar)):
            if i == tar[j]:
                 d_target_num.append(j)
    for i in range(len(d_target_num)):
        if d_target[i] != tar[d_target_num[i]]:
            print(i)
    data_list=[]
    for q,label in zip(d_test,d_target_num):
        data=[]
        data.append(q)
        data.append(str(label))
        data_list.append(data)
    dataset_train,dataset_test = train_test_split(data_list, test_size = 0.2, random_state = 0)
    test_data = np.array(dataset_test)
    test_sen = list(test_data[:,0])
    test_label = list(test_data[:,1])

    #data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
    #data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

    #train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
    #test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

    #end = 1
    #while end == 1:
    #    sentence = input('input : ')
    #    if sentence ==0:
    #        break
    #    predict(sentence)
    #    print('\n')
    path = '/toy/LG_model/state2'
    file_list=os.listdir(path)
    for j in file_list:
        count = 0
        percent=0.0
        for i in range(len(test_sen)):
            flag = 0
            targ = tar[int(test_label[i])] #target number
            res = tar[predict(str(test_sen[i]),j)] #predict number
            i=str(i)
            i=i.rjust(3,'0')
            if res == targ:
                count+=1 
        slash=colored('/ ','yellow')
        percent = (count/len(test_sen))*100
        percent = round(percent,3)
        print(colored(j+' accuracy = '+str(percent)+'%','cyan',attrs=['bold']))
        os.rename(path+j,path+str(percent)+'% '+j)

if __name__ == '__main__':
    main()
