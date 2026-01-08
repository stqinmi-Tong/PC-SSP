import torch
import os.path as osp
import numpy as np
import random

def keylist_2_valuelist(keylist, dic, start_index=0):
    value_list = []
    for key in keylist:
        value = dic.get(key)
        if value is None:
            value = len(dic) + start_index
            dic[key] = value
        value_list.append(value)
    return value_list
def get_rel_dict(rel_list,relation_str2id,relation_id2wordlist):
    one_path = []  
    for potential_relation in rel_list:
        rel_id = relation_str2id.get(potential_relation)
        if rel_id is None:
            rel_id = len(relation_str2id) + 1  
            relation_str2id[potential_relation] = rel_id
        wordlist = potential_relation.split('_')  
        relation_id2wordlist[rel_id] = wordlist 
        one_path.append(rel_id)  
    return one_path,relation_str2id,relation_id2wordlist
def add_tuple2tailset(ent_path, one_path, tuple2tailset):
    size = len(one_path)
    if len(ent_path) != size + 1:
        print('len(ent_path)!=len(one_path)+1:', len(ent_path), size)
        exit(0)
    for i in range(size):
        tuple = (ent_path[i], one_path[i]) 
        tail = ent_path[i + 1]  
        tailset = tuple2tailset.get(tuple)
        if tailset is None:
            tailset = set()
        if tail not in tailset:
            tailset.add(tail)
            tuple2tailset[tuple] = tailset 


def add_rel2tailset(ent_path, one_path, rel2tailset):
    size = len(one_path)
    if len(ent_path) != size + 1:
        print('len(ent_path)!=len(one_path)+1:', len(ent_path), size)
        exit(0)
    for i in range(size):
        #         tuple=(ent_path[i], one_path[i])
        tail = ent_path[i + 1]
        rel = one_path[i]
        tailset = rel2tailset.get(rel)
        if tailset is None:
            tailset = set()
        if tail not in tailset:
            tailset.add(tail)
            rel2tailset[rel] = tailset 


def add_ent2relset(ent_path, one_path, ent2relset, maxSetSize):
    size = len(one_path)
    if len(ent_path) != size + 1:
        print('len(ent_path)!=len(one_path)+1:', len(ent_path), size)
        exit(0)
    for i in range(size):
        ent_id = ent_path[i + 1]
        rel_id = one_path[i]
        relset = ent2relset.get(ent_id)
        if relset is None:
            relset = set()
        if rel_id not in relset:
            relset.add(rel_id)
            if len(relset) > maxSetSize:
                maxSetSize = len(relset)  
            ent2relset[ent_id] = relset 
    return maxSetSize


def load_data(rootPath, files, maxPathLen=2): 
    tuple2tailset = {}  
    rel2tailset = {} 
    ent2relset = {}  
    ent2relset_maxSetSize = 0

    train_paths_store = []  
    train_ents_store = []  
    train_masks_store = [] 

    dev_paths_store=[]
    dev_ents_store = []
    dev_masks_store=[]


    test_paths_store = []
    test_ents_store = []
    test_masks_store = []

    max_path_len = 0
    for file_id, fil in enumerate(files):

        # filename = rootPath + fil  
        filename = osp.join(rootPath, fil)
        print('loading', filename, '...')
        read_pathfile = torch.load(filename)

        for line in read_pathfile:
            parts = line  
            ent_list = []  
            rel_list = [] 
            if len(parts)==1:
                continue
            for i in range(len(parts)):
                if i % 2 == 0:
                    ent_list.append(int(parts[i]))
                else:
                    rel_list.append(int(parts[i]))
            if len(ent_list) != len(rel_list) + 1:
                print('len(ent_list)!=len(rel_list)+1:', len(ent_list), len(rel_list))
                print('line:', line)
                exit(0)
            ent_path = ent_list  
            one_path = rel_list
            
            add_tuple2tailset(ent_path, one_path, tuple2tailset) 
            add_rel2tailset(ent_path, one_path, rel2tailset) 
            ent2relset_maxSetSize = add_ent2relset(ent_path, one_path, ent2relset, ent2relset_maxSetSize) 

            # pad
            valid_size = len(one_path)
            if valid_size > max_path_len:
                max_path_len = valid_size
            pad_size = maxPathLen - valid_size
            if pad_size > 0:
                one_path = [0] * pad_size + one_path 
                ent_path = ent_path[:1] * (pad_size + 1) + ent_path[1:] 
                one_mask = [0.0] * pad_size + [1.0] * valid_size  
            else:
                one_path = one_path[-maxPathLen:]  # select the last max_len relations
                ent_path = ent_path[:1] + ent_path[-maxPathLen:]
                one_mask = [1.0] * maxPathLen

            if file_id == 0: 
                if len(ent_path) != maxPathLen + 1 or len(one_path) != maxPathLen:
                    print('line:', line)
                    exit(0)
                train_paths_store.append(one_path)  
                train_ents_store.append(ent_path) 
                train_masks_store.append(one_mask)  
            if file_id == 1:  
                if len(ent_path) != maxPathLen + 1 or len(one_path) != maxPathLen:
                    print('line:', line)
                    exit(0)
                dev_paths_store.append(one_path)
                dev_ents_store.append(ent_path)
                dev_masks_store.append(one_mask)
            if file_id == 2: 
                if len(ent_path) != maxPathLen + 1 or len(one_path) != maxPathLen:
                    print('line:', line)
                    exit(0)
                test_paths_store.append(one_path)
                test_ents_store.append(ent_path)
                test_masks_store.append(one_mask)

        print('load over, overall ', len(train_paths_store), ' train,',len(dev_paths_store), ' validation,', len(test_paths_store), ' test,',
              'tuple2tailset size:', len(tuple2tailset), ', max path len:', max_path_len, 'max ent2relsetSize:',
              ent2relset_maxSetSize)

    return ((train_paths_store, train_masks_store, train_ents_store),(dev_paths_store, dev_masks_store, dev_ents_store),
            (test_paths_store, test_masks_store,test_ents_store)), \
            tuple2tailset, rel2tailset

def neg_entity_tensor_v2(ent_idmatrix, rel_idmatrix, pair2tailset, rel2tailset, neg_size, ent_vocab_size):

    length = len(rel_idmatrix)
    ent_vocab_set = set(range(ent_vocab_size))

    if len(ent_idmatrix) != length + 1:
        print('error in neg ent generation, len(ent_idlist)!=length+1:', len(ent_idmatrix), length)
        exit(0)
    negs = []
    for i in range(length):
        rel_id = rel_idmatrix[i]
        pair = (ent_idmatrix[i], rel_id)
        tailset = pair2tailset.get(pair)
        if tailset is None:
            tailset = set()

        rel_tailset = rel2tailset.get(rel_id)
        if rel_tailset is None:
            rel_tailset = set()
        key_neg_range_set = rel_tailset - tailset#if [rel, (ent1, ent2....entn)]
        remain_size = neg_size - len(key_neg_range_set)
        if remain_size <= 0:
            neg_list = random.sample(key_neg_range_set, neg_size)
        else:
            neg_cand_set = ent_vocab_set - key_neg_range_set
            neg_list = list(key_neg_range_set) + random.sample(neg_cand_set, remain_size)
        negs.append(neg_list)
    return np.asarray(negs).reshape((length, neg_size))


  

