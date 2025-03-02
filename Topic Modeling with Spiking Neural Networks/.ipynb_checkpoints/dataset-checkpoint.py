import torch
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer
from nltk.stem import WordNetLemmatizer
from torch.utils.data import Dataset

class AbstractDataset(Dataset):
    def __init__(self, csv_path, txt_col, label_col, tokenizer, word_to_ind, null_word=None, 
                 min_len=None, lemmatize=True, verbose=False):
        '''
        csv_path: path to abstract data
        txt_col: name of column containing abstracts
        label_col: name of column containing labels (CPC or IPCR)
        tokenizer: torchtext.data.utils tokenizer
        word_to_ind: dict mapping word -> index in embedding
        null_word: word in the KeyedVectors of the "null" embedding, used to represent padding
        min_len: minimum length to enforce of each abstract; if in abstract is too short, it will be dropped
        lemmatize: whether or not to lemmatize each word
        verbose: show progress bar
        '''
        df = pd.read_csv(csv_path)
        if min_len:
            df = df[df[txt_col].str.split().str.len() >= min_len]
        
        abst, classes = df[txt_col], df[label_col]
        
        # Get only the section and class of each IPCR category
        classes = classes.str.split('; ').apply(lambda classes: pd.Series(classes).str.extract(r'([A-Z]\d{2})', expand=False).tolist())
        unique_classes = classes.explode().sort_values().unique()
        # Create multi-hot vectors for each abstract
        mlb = MultiLabelBinarizer(classes=unique_classes)
        
        self.tokenizer = tokenizer
        self.word_to_ind = word_to_ind
        self.lemmatizer = (WordNetLemmatizer() if lemmatize else lambda w: w)
        self.labels = torch.tensor(mlb.fit_transform(classes)).to_sparse_csr().type(torch.float)
        self.pad_idx = word_to_ind.get(null_word, None)
        self.len = max(len(self.tokenizer(doc)) for doc in abst) if null_word else None
        
        self.abst_data = [torch.tensor(self.doc_to_ind(a)).unsqueeze(0) for a in (tqdm(abst) if verbose else abst)]
        self.classes = unique_classes
        
        
    def doc_to_ind(self, doc):
        '''
        Convert document to list of indices; each index represents a word's position
        in the KeyedVectors
        '''
        tokens = [self.word_to_ind[self.lemmatizer.lemmatize(w)] for w in self.tokenizer(doc)]
        padding = [self.pad_idx]*(self.len - len(tokens)) if self.pad_idx else []
        return tokens + padding
        
    
    def __len__(self):
        return len(self.abst_data)
    
    def __getitem__(self, idx):
        return (self.abst_data[idx], self.labels[idx].to_dense())
