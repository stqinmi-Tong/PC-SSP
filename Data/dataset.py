import random

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from Data.preprocessing import load_data, neg_entity_tensor_v2


class Train_Dataset(data.Dataset):
    def  __init__(self, args):
        self.args = args
        self.relations_store = args.train_set[0] 
        self.masks_store = args.train_set[1]  
        self.entity_store = args.train_set[2] 

        self.ent_vocab_size = len(args.entity2id)  
        self.ent_str2id = args.entity2id  
        # self.tuple2tailset = args.tuple2tailset  
        # self.rel2tailset = args.rel2tailset 

    def __getitem__(self, idx):
        relations = [example for example in self.relations_store[idx]]
        entity = [example for example in self.entity_store[idx]]
        masks = [example for example in self.masks_store[idx]]
        
        return np.array(relations), np.array(masks), np.array(entity) #, negs

    def __len__(self):
        return len(self.relations_store)

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        label = torch.stack([_[1] for _ in data], dim=0)
        return triple, label

    def get_label(self, label):
        y = np.zeros([self.ent_vocab_size], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)

class Valid_Dataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.relations_store = args.valid_set[0]  
        self.masks_store = args.valid_set[1]  
        self.entity_store = args.valid_set[2]  
        self.ent_vocab_size = len(args.entity2id)  
      
    def __getitem__(self, i):
        relations = [example for example in self.relations_store[i]]
        entity = [example for example in self.entity_store[i]]
        masks = [example for example in self.masks_store[i]]
        
        return np.array(relations), np.array(masks), np.array(entity)#, label

    def __len__(self):
        return len(self.relations_store)


    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        label = torch.stack([_[1] for _ in data], dim=0)
        return triple, label

    def get_label(self, label):
        y = np.zeros([self.ent_vocab_size], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)

class Test_Dataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.relations_store = args.test_set[0] 
        self.masks_store = args.test_set[1] 
        self.entity_store = args.test_set[2]  
        self.ent_vocab_size = len(args.entity2id) 
        # self.tuple2tailset = args.tuple2tailset  
        # self.rel2tailset = args.rel2tailset 

    def __getitem__(self, i):
        relations = [example for example in self.relations_store[i]]
        entity = [example for example in self.entity_store[i]]
        masks = [example for example in self.masks_store[i]]
        pairs = (entity[1], relations[-1])
        # label = self.tuple2tailset[pairs]
        # label = self.get_label(label)
        # print(label)
        return np.array(relations), np.array(masks), np.array(entity)#, label

    def __len__(self):
        return len(self.relations_store)


    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        label = torch.stack([_[1] for _ in data], dim=0)
        return triple, label

    def get_label(self, label):
        y = np.zeros([self.ent_vocab_size], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)


class Train_Path_Dataset(data.Dataset):
    def  __init__(self, args):
        self.args = args
        self.dataset = torch.load(args.data_dir + "train_paths.pt")

        # self.path_length = self.dataset.shape[1]
        self.ent2id = args.entity2id  
        self.rel2id = args.relation2id 

    def __getitem__(self, idx):
        paths = [path for path in self.dataset[idx]]
        return np.array(paths)

    def __len__(self):
        return len(self.dataset)


class Valid_Path_Dataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.dataset = torch.load(args.data_dir + "dev_paths.pt")
        # self.path_length = self.dataset.shape[1]

    def __getitem__(self, idx):
        paths = [path for path in self.dataset[idx]]
        return np.array(paths)

    def __len__(self):
        return len(self.dataset)



class Test_Path_Dataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.dataset = torch.load(args.data_dir + "test_paths.pt")

    def __getitem__(self, idx):
        paths = [path for path in self.dataset[idx]]
        return np.array(paths)

    def __len__(self):
        return len(self.dataset)


class Path_Dataset(data.Dataset):
    def  __init__(self, data):
        self.dataset = data

    def __getitem__(self, idx):
        paths = [path for path in self.dataset[idx]]
        return np.array(paths)

    def __len__(self):
        return len(self.dataset)


class Train_Joint_Dataset(data.Dataset):
    def  __init__(self, args):
        self.args = args
        
        self.relations_store = args.train_set[0]  
        self.masks_store = args.train_set[1]  
        self.entity_store = args.train_set[2] 

        self.ent_vocab_size = len(args.entity2id)  
        self.ent_str2id = args.entity2id 

        self.dataset = torch.load(args.data_dir + "train_paths.pt")
        self.ent2id = args.entity2id  
        self.rel2id = args.relation2id 


    def __getitem__(self, idx):
        relations = [example for example in self.relations_store[idx]]
        entity = [example for example in self.entity_store[idx]]
        masks = [example for example in self.masks_store[idx]]
        if idx < len(self.dataset):
            paths = [path for path in self.dataset[idx]]
        else:
            i = np.random.randint(0, len(self.dataset), 1)
            i = i[0]
            paths = [path for path in self.dataset[i]]
            # print('the index is too large for the sentence data')


        return np.array(relations), np.array(masks), np.array(entity), np.array(paths)

    def __len__(self):
        return len(self.relations_store)

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        label = torch.stack([_[1] for _ in data], dim=0)
        return triple, label

    def get_label(self, label):
        y = np.zeros([self.ent_vocab_size], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)

