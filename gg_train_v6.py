import os
import pandas as pd
from konlpy.tag import Mecab
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
from sklearn.model_selection import train_test_split
from transformers import AdamW
from transformers import Adafactor
from transformers.optimization import get_cosine_schedule_with_warmup
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import numpy
from tqdm import tqdm
from time import time
from time import ctime
import subprocess

subprocess.call(['sh','/toy/logo.sh'])

tt = ctime(time())
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 30
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

bertmodel,vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()
m = Mecab(dicpath='/toy/mecab-ko-dic-2.1.1-20180720')
tok = nlp.data.BERTSPTokenizer(m,vocab,lower=False)

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

def main():
    # Setting parameters
    path = '/toy/LG_data/'
    file_list = os.listdir(path)
    d_test , d_result , d_target , d_target_num = [], [], [], []
    for file in file_list:
        if file[-1]=='x':
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
    d_concat=[]
    for i in range(len(d_test)):
        d_concat.append(d_test[i]+'\n'+d_result[i])
    data_list=[]
    for q,label in zip(d_test,d_target_num):
        data=[]
        data.append(q)
        data.append(str(label))
        data_list.append(data)
    for q,label in zip(d_result,d_target_num):
        data=[]
        data.append(q)
        data.append(str(label))
        data_list.append(data)
    for q,label in zip(d_concat,d_target_num):
        data=[]
        data.append(q)
        data.append(str(label))
        data_list.append(data)
    dataset_train,dataset_test = train_test_split(data_list, test_size = 0.2, random_state = 0)

    for max_len in range(61,65):
        for batch_size in range(60,70):
            for warmup_ratio in np.arange(0.1,0.3,0.1):
                test_acc=0.0
                data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
                data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

                train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
                test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

                #device = torch.device('cpu')
                model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

                no_decay = ['bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]

                optimizer = Adafactor(optimizer_grouped_parameters, lr=learning_rate)
                loss_fn = nn.CrossEntropyLoss()

                t_total = len(train_dataloader) * num_epochs
                warmup_step = int(t_total * warmup_ratio)
                scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
                cal_accuracy=0.0
                def calc_accuracy(X,Y):
                    max_vals, max_indices = torch.max(X, 1)
                    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
                    return train_acc
                train_dataloader

                for e in range(num_epochs):
                    train_acc = 0.0
                    test_acc = 0.0
                    model.train()
                    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
                        optimizer.zero_grad()
                        token_ids = token_ids.long().to(device)
                        segment_ids = segment_ids.long().to(device)
                        valid_length= valid_length
                        label = label.long().to(device)
                        out = model(token_ids, valid_length, segment_ids)
                        loss = loss_fn(out, label)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        train_acc += calc_accuracy(out, label)
                        if batch_id % log_interval == 0:
                            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
                    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
                
                    model.eval()
                    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
                        token_ids = token_ids.long().to(device)
                        segment_ids = segment_ids.long().to(device)
                        valid_length= valid_length
                        label = label.long().to(device)
                        out = model(token_ids, valid_length, segment_ids)
                        test_acc += calc_accuracy(out, label)
                    test_acc=test_acc/(batch_id+1)
                    #print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
                    print("epoch {} test acc {}".format(e+1, test_acc))
                m_path = '/toy/LG_model/state2'
                test_accs = str(round(test_acc,3))
                print('test_accs:',test_accs)
                print(test_accs+',opti='+str(optimizer_name)+',max_len='+str(max_len)+',batch_size='+str(batch_size)+',warmup_ratio='+str(warmup_ratio)
                if test_acc > 0.89:
                    torch.save(model.state_dict(),m_path+'kobert_model_state_'+test_accs+',opti='+str(optimizer_name)+',max_len='+str(max_len)+',batch_size='+str(batch_size)+',warmup_ratio='+str(warmup_ratio)+'.pt')

if __name__ == '__main__':
    main()
