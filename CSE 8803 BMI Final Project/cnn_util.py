import time
import torch
import gensim
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from util import save_model

class CNNBase1(nn.Module):
    '''
    Standard CNN text model (no spiking neurons). 
    Type 1  
    '''
    def __init__(self, embed_keys_path, null_word=None):
        '''
        embed_keys_path: path to KeyedVectors file containing embeddings
        '''
        super().__init__()
        
        wv = gensim.models.KeyedVectors.load(embed_keys_path, mmap='r')
        pad_idx = (wv.key_to_index.get(null_word, None))
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(wv.vectors), padding_idx=pad_idx, freeze=False)
        
        self.lstm1 = nn.LSTM(200, 25, batch_first=True)
        
        self.conv1 = nn.Conv2d(1, 16, (5, 25), bias=False)
        self.relu1 = nn.ReLU()
        self.flatten1 = nn.Flatten(start_dim=2)
        self.conv2 = nn.Conv1d(16, 16, 2, 2, bias=False)
        self.flatten2 = nn.Flatten()
        self.relu2 = nn.ReLU()
        self.linear = nn.Linear(5312, 91, bias=False)
        
    def forward(self, x, all_outputs_max=False):
        out1 = self.embedding(x)
        out2, _ = self.lstm1(out1)
        
        out3 = self.conv1(out2.unsqueeze(1))
        out4 = self.relu1(out3)
        out5 = self.flatten1(out4)
        out6 = self.conv2(out5)
        out7 = self.flatten2(out6)
        out8 = self.relu2(out7)
        out9 = self.linear(out8)
        
        if all_outputs_max:
            return out1.max(), out2.max(), out3.max(), out4.max(), out5.max(), out6.max(), out7.max(), out8.max(), out9.max()
        return out9
    
class CNNBase2(nn.Module):
    '''
    Standard CNN text model (no spiking neurons). 
    Type 2
    '''
    def __init__(self, embed_keys_path, null_word=None):
        
        super().__init__()

        # Get the word vectors; pad_idx is the index in wv of the null word (whose embedding vector is all 0's)
        wv = gensim.models.KeyedVectors.load(embed_keys_path, mmap='r')
        pad_idx = (wv.key_to_index.get(null_word, None))
        
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(wv.vectors), padding_idx=pad_idx, freeze=True)
                
        # Network architecture begins here:
        self.lstm1 = nn.LSTM(200, 50, batch_first=True)
        
        self.conv1 = nn.Conv2d(1, 16, (5, 50), bias=False)
        self.flatten1 = nn.Flatten(start_dim=2)
        
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(665, 300, bias=False)
        
        self.conv2 = nn.Conv1d(16, 16, 2, 2, bias=False)
        self.flatten2 = nn.Flatten()
        
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(2400, 91, bias=False)
     
    def forward(self, x, all_outputs_max=False):
        '''Complete a full forward pass for a single time step'''
        out1 = self.embedding(x)
        out2, _ = self.lstm1(out1)
        out3 = self.conv1(out2.unsqueeze(1))
        out4 = self.flatten1(out3)
        out5 = self.relu1(out4)
        out6 = self.linear1(out5)
        out7 = self.conv2(out6)
        out8 = self.flatten2(out7)
        out9 = self.relu2(out8)
        out10 = self.linear2(out9)

        if all_outputs_max:
            return (out1.max(), out2.max(), out3.max(), out4.max(), out5.max(), out6.max(), out7.max(), out8.max(), out9.max(), 
                    out10.max())
        
        return out10

class CNNBase3(nn.Module):
    '''
    Standard CNN text model (no spiking neurons). 
    Type 2
    '''
    
    def __init__(self, embed_keys_path, null_word=None):
        '''
        embed_keys_path: path to KeyedVectors file containing embeddings
        '''
        super().__init__()
        
        wv = gensim.models.KeyedVectors.load(embed_keys_path, mmap='r')
        pad_idx = (wv.key_to_index.get(null_word, None))
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(wv.vectors), padding_idx=pad_idx, freeze=False)
        
        self.lstm1 = nn.LSTM(200, 50, batch_first=True)
        
        self.conv1 = nn.Conv2d(1, 16, (5, 50), bias=False)
        self.relu1 = nn.ReLU()
        self.flatten1 = nn.Flatten(start_dim=2)
        
        self.linear1 = nn.Linear(665, 300, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 16, 2, 2, bias=False)
        
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv1d(16, 32, 5, bias=False)
        
        self.flatten2 = nn.Flatten()
        self.relu4 = nn.ReLU()
        self.linear2 = nn.Linear(4672, 91, bias=False)
        
    def forward(self, x, all_outputs_max=False):
        out1 = self.embedding(x)
        out2, _ = self.lstm1(out1)
        
        out3 = self.conv1(out2.unsqueeze(1))
        out4 = self.relu1(out3)
        out5 = self.flatten1(out4)
        
        out6 = self.linear1(out5)
        out7 = self.relu2(out6)
        out8 = self.conv2(out7)
        
        out9 = self.relu3(out8)
        out10 = self.conv3(out9)
        
        out11 = self.flatten2(out10)
        out12 = self.relu4(out11)
        out13 = self.linear2(out12)
        
        if all_outputs_max:
            return (out1.max(), out2.max(), out3.max(), out4.max(), out5.max(), 
                    out6.max(), out7.max(), out8.max(), out9.max(), out10.max(), 
                    out11.max(), out12.max(), out13.max())
        return out13


def train_model(model, optimizer, dataset, loss_fn, epochs, batch_size, save_freq=None, save_path=None, scheduler=None, device='cpu'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    
    print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
    print(f'Scheduler: {scheduler}' if scheduler else 'No learning rate scheduling!')
    print(f'Training for {epochs} epochs, with batch size={batch_size}')
    print(f'Using device: {device}')
    print(f'Saving model every {save_freq} epochs to {save_path}' if save_freq else 'WARNING: Will not save model!')

    for e in range(epochs):
        losses = []
        all_pred, all_true = [], []
        t = time.time()
        print(f'\n-----Epoch {e+1}/{epochs}-----')
        for i, (txt, labels) in enumerate(loader):
            labels = labels.to(device)
            txt = txt.to(device).squeeze()
            pred = model(txt)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            all_pred.append(pred.cpu())
            all_true.append(labels.cpu())

            if len(losses) == 150 or i == len(loader)-1:
                elapsed = time.time() - t
                t = time.time()
                print(f'Batch {i+1}/{len(loader)}, loss: {np.mean(losses)} ({elapsed:.3f}s)')
                losses = []
                
        if scheduler is not None:
            scheduler.step()
            
        if save_freq and ((e+1) % save_freq == 0 or e == epochs-1):
            save_model(save_path, model, optimizer, epochs)
            print(f'Saved to {save_path}')       
         
        f1 = f1_score(torch.cat(all_true, dim=0), 
                      (torch.sigmoid(torch.cat(all_pred, dim=0)) > 0.5).type(torch.float), 
                      average='weighted')
        print(f'F1 score: {f1}')

