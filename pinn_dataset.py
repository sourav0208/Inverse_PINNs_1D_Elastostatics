import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,x):
        self.x=x
    def __getitem__(self,index):
        return self.x[index]
    def __len__(self):
        return self.x.shape[0]