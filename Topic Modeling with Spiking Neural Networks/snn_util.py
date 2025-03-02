import time
import torch
import gensim
import numpy as np
import snntorch as snn

import snntorch.functional.loss as snnloss

from snntorch import surrogate, utils as snnutils
from torch.utils.data import DataLoader
from torch import nn, optim, jit
from torch.nn import Parameter

from torchtext.data import utils
from collections import OrderedDict

from sklearn.metrics import f1_score

from dataset import AbstractDataset
from util import load_model, load_model_and_opt, save_model, batch_predict, load_from_cnn


class SpikingBCELoss(snnloss.LossFunctions):
    '''
    Custom loss for SNNs that replicates BCEWithLogitsLoss from PyTorch
    '''
    def __init__(self, reduction='mean', weight=None, pos_weight=None):
        super().__init__(reduction=reduction, weight=weight)
        
        self.pos_weight = pos_weight
        self.loss_fn = nn.BCEWithLogitsLoss( 
                                       weight=self.weight, 
                                       pos_weight=self.pos_weight)
        self.__name__ = 'spiking_bce_loss'
        
    def to(self, device):
        if self.pos_weight is not None:
            self.pos_weight = self.pos_weight.to(device)
        if self.weight is not None:
            self.weight = self.weight.to(device)
        self.loss_fn = self.loss_fn.to(device)
        return self
        
    def _compute_loss(self, spk_out, targets):
        '''
        spk_out: shape (time steps, batch, num_targets)
        targets: shape (batch, num_targets)
        '''
        
        prob_to_logit = lambda p: (p - 0.5) * 250 #-torch.log(( (1+1e-7) / (p + 1e-8)) - 1)
        
        device, num_steps, num_outputs = self._prediction_check(spk_out)
        
        loss_shape = (spk_out.size(1)) if self._intermediate_reduction() == 'none' else (1)
        loss = torch.zeros(loss_shape, dtype=torch.float, device=device)
        
        logits = prob_to_logit(spk_out)
        for step in range(num_steps):
            loss += self.loss_fn(logits[step], targets)

        return loss / num_steps



class AbstractSNN_1(nn.Module):
    
    def __init__(self, T, embed_keys_path, null_word=None, beta=1.0):
        
        super().__init__()
        
        # Number of time steps for spiking
        self.T = T
        
        # Get the word vectors; pad_idx is the index in wv of the null word (whose embedding vector is all 0's)
        wv = gensim.models.KeyedVectors.load(embed_keys_path, mmap='r')
        pad_idx = (wv.key_to_index.get(null_word, None))
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(wv.vectors), padding_idx=pad_idx, freeze=True)
        
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Network architecture begins here:
        self.lstm1 = nn.LSTM(200, 50, batch_first=True)
        
        self.network1 = nn.Sequential(
            nn.Conv2d(1, 16, (5, 50), bias=False),
            nn.Flatten(start_dim=2),
        )
        
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        self.linear1 = nn.Linear(665, 300, bias=False)
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)
        
        self.network2 = nn.Sequential(
            nn.Conv1d(16, 16, 2, 2, bias=False),
            nn.Flatten(),
        )
        
        self.lif3 = snn.Leaky(beta=beta, init_hidden=True)
        self.network3 = nn.Linear(2400, 91, bias=False)
        
        self.lif4 = snn.Leaky(beta=beta, init_hidden=False, output=True)
        self.inhibitor = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(91, 91)))
        self.clamp_inhibitor()
    
    def clamp_inhibitor(self):
        self.inhibitor.data = self.inhibitor.data.fill_diagonal_(0).clamp(min=0)
    
    def forward(self, x, stm_mem):
        '''Complete a full forward pass for a single time step'''
        
        out, _ = self.lstm1(x)
        out = self.network1(out.unsqueeze(1))
        spk1 = self.lif1(out)
        out = self.linear1(spk1)
        spk2 = self.lif2(out)
        out = self.network2(spk2)
        spk3 = self.lif3(out)
        out = self.network3(spk3)
        spk4, stm_mem = self.lif4(out, stm_mem)
        return spk4, stm_mem - spk4.mm(self.inhibitor)

class AbstractSNN_2(nn.Module):
    
    def __init__(self, T, embed_keys_path, null_word=None, beta=1.0):
        
        super().__init__()
        
        # Number of time steps for spiking
        self.T = T
        
        # Get the word vectors; pad_idx is the index in wv of the null word (whose embedding vector is all 0's)
        wv = gensim.models.KeyedVectors.load(embed_keys_path, mmap='r')
        pad_idx = (wv.key_to_index.get(null_word, None))
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(wv.vectors), padding_idx=pad_idx, freeze=True)
        
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Network architecture begins here:
        self.lstm1 = nn.LSTM(200, 50, batch_first=True)
        
        self.conv1 = nn.Conv2d(1, 16, (5, 50), bias=False)
        self.flatten1 = nn.Flatten(start_dim=2)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        
        self.linear1 = nn.Linear(665, 300, bias=False)
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)
        self.conv2 = nn.Conv1d(16, 16, 2, 2, bias=False)
        
        self.lif3 = snn.Leaky(beta=beta, init_hidden=True)
        self.conv3 = nn.Conv1d(16, 32, 5, bias=False)
        self.flatten2 = nn.Flatten()
        
        self.lif4 = snn.Leaky(beta=beta, init_hidden=True)
        self.linear2 = nn.Linear(4672, 91, bias=False)
        self.lif5 = snn.Leaky(beta=beta, init_hidden=False, output=True)
        self.inhibitor = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(91, 91)))
        self.clamp_inhibitor()
    
    def clamp_inhibitor(self):
        self.inhibitor.data = self.inhibitor.data.fill_diagonal_(0).clamp(min=0)
    
    def forward(self, x, stm_mem):
        '''Complete a full forward pass for a single time step'''
        
        out, _ = self.lstm1(x)
        out = self.conv1(out.unsqueeze(1))
        out = self.flatten1(out)
        spk1 = self.lif1(out)
        
        out = self.linear1(spk1)
        spk2 = self.lif2(out)
        out = self.conv2(spk2)
        
        spk3 = self.lif3(out)
        out = self.conv3(spk3)
        out = self.flatten2(out)
        
        spk4 = self.lif4(out)
        out = self.linear2(spk4)
        spk5, stm_mem = self.lif5(out, stm_mem)
        return spk5, stm_mem - spk5.mm(self.inhibitor)

class AbstractHybrid(nn.Module):
    
    def __init__(self, T, embed_keys_path, null_word=None, beta=1.0):
        
        super().__init__()
        
        # Number of time steps for spiking
        self.T = T
        
        # Get the word vectors; pad_idx is the index in wv of the null word (whose embedding vector is all 0's)
        wv = gensim.models.KeyedVectors.load(embed_keys_path, mmap='r')
        pad_idx = (wv.key_to_index.get(null_word, None))
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(wv.vectors), padding_idx=pad_idx, freeze=True)
                
        # Network architecture begins here:
        self.lstm1 = nn.LSTM(200, 25, batch_first=True)
        
        self.network1 = nn.Sequential(
            nn.Conv2d(1, 16, (5, 25), bias=False),
            nn.Flatten(start_dim=2),
        )
        
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)

        self.network2 = nn.Sequential(
            nn.Conv1d(16, 16, 2, 2, bias=False),
            nn.Flatten(),
        )
        
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)
        self.network3 = nn.Linear(5312, 91, bias=False)
    
    def forward(self, x):
        '''Complete a full forward pass for a single time step'''
        
        out, _ = self.lstm1(x)
        out = self.network1(out.unsqueeze(1))
        spk1 = self.lif1(out)
        out = self.network2(spk1)
        spk2 = self.lif2(out)
        out = self.network3(spk2)
        return out


def forward_pass(net, T, x, return_mem=False):
    '''
    Complete a full forward pass for all timesteps, T.
    The spikes returned by the model will be averaged and interpreted as a probability (rate coding).
    E.g., if the output neuron for class #20 spikes 32 times over 45 total time steps, we predict that 
    the abstract belongs to class #20 with probability 32/45 = 71.1%
    
    Returns logits since the BCEWithLogitsLoss expects logits, not probabilities.
    '''
    snnutils.reset(net)
    
    stm_mem = net.lif3.reset_mem()
    spk = []
    mem = []
    x = net.embedding(x)
    for t in range(T):
        out, stm_mem = net(torch.bernoulli(x), stm_mem)
        spk.append(out)
        mem.append(stm_mem)

    probs = torch.stack(spk)
    mem = torch.stack(mem)
    return probs if not return_mem else (probs, mem)


def train_model(model, optimizer, dataset, loss_fn, T, epochs, batch_size, 
                save_freq=None, save_path=None, scheduler=None, device='cpu'):
    '''Train the model for a desired number of epochs'''
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    
    print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
    print(f'Training for {epochs} epochs, with T={T}, batch size={batch_size}')
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
            pred = forward_pass(model, T, txt)
            loss = loss_fn(pred, labels)

            loss.backward()
            optimizer.step()
            model.clamp_inhibitor()
            optimizer.zero_grad()
            
            losses.append(loss.item())
            all_pred.append(pred.mean(dim=0))
            all_true.append(labels)

            if len(losses) == 75 or i == len(loader)-1:
                elapsed = time.time() - t
                t = time.time()
                print(f'Batch {i+1}/{len(loader)}, loss: {np.mean(losses)} ({elapsed:.3f}s)')
                losses = []
                
        if scheduler is not None:
            scheduler.step()
            
        if save_freq and ((e+1) % save_freq == 0 or e == epochs-1):
            save_model(save_path, model, optimizer, epochs)
            print(f'Saved to {save_path}')
                
        f1 = f1_score(torch.cat(all_true, dim=0).cpu(), 
                      ((torch.cat(all_pred, dim=0).cpu()) > 0.5).type(torch.float), 
                      average='weighted')
        print(f'F1 score: {f1}')

    
