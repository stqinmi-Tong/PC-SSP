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
    one_path = []  # 关系集合中各关系的编号
    for potential_relation in rel_list:
        rel_id = relation_str2id.get(potential_relation)
        if rel_id is None:
            rel_id = len(relation_str2id) + 1  # 将关系进行编号，并储存到字典中
            relation_str2id[potential_relation] = rel_id
        wordlist = potential_relation.split('_')  # 将关系中的各个单词分割
        relation_id2wordlist[rel_id] = wordlist # 构成关系和关系中包含的词列表的词典：{r：[r_word1,r_word2,...]}
        one_path.append(rel_id)  # 关系集合编号
    return one_path,relation_str2id,relation_id2wordlist
def add_tuple2tailset(ent_path, one_path, tuple2tailset):
    size = len(one_path)
    if len(ent_path) != size + 1:
        print('len(ent_path)!=len(one_path)+1:', len(ent_path), size)
        exit(0)
    for i in range(size):
        tuple = (ent_path[i], one_path[i])  # 实体En与关系Rn
        tail = ent_path[i + 1]  # 实体En+1
        tailset = tuple2tailset.get(tuple)
        if tailset is None:
            tailset = set()
        if tail not in tailset:
            tailset.add(tail)
            tuple2tailset[tuple] = tailset  # 三元组的存储方式:{(En, Rn): set(En+1, ...)}


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
            rel2tailset[rel] = tailset  # 存储的是关系Rn与(实体En+1, ...)


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
                maxSetSize = len(relset)  # 找出一个实体对应着多种关系
            ent2relset[ent_id] = relset  # 存储的是实体En+1与(关系Rn, ...)
    return maxSetSize


def load_data(rootPath, files, maxPathLen=2): ##也就是读取ent_recovered.txt文件，而我们自己生成的FB和WN的路径文件

    tuple2tailset = {}  # 三元组
    rel2tailset = {}  # 存储的是关系Rn与(实体En+1, ...)
    ent2relset = {}  # 存储的是实体En+1与(关系Rn, ...)
    ent2relset_maxSetSize = 0

    train_paths_store = []  # 数据集中的关系列表
    train_ents_store = []  # 数据集中的实体列表
    train_masks_store = []  # 数据集中的标签列表

    dev_paths_store=[]
    dev_ents_store = []
    dev_masks_store=[]


    test_paths_store = []
    test_ents_store = []
    test_masks_store = []

    max_path_len = 0
    for file_id, fil in enumerate(files):

        # filename = rootPath + fil  # 数据集路径
        filename = osp.join(rootPath, fil)
        print('loading', filename, '...')
        read_pathfile = torch.load(filename)

        for line in read_pathfile:
            parts = line  # 数据集形式：E1\tR1\tE2...,通过分割将实体与关系分开
            ent_list = []  # 实体集合
            rel_list = []  # 关系集合
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
            ent_path = ent_list  # 存储路径中实体的编号
            one_path = rel_list
            #print(parts,ent_path,one_path) #[3061, 1, 15881, 0, 3062] [3061, 15881, 3062] [1, 0]
            add_tuple2tailset(ent_path, one_path, tuple2tailset)  # 三元组的存储方式:{(En, Rn), set(En+1, ...)}
            add_rel2tailset(ent_path, one_path, rel2tailset)  # 存储的是关系Rn与(实体En+1, ...)
            ent2relset_maxSetSize = add_ent2relset(ent_path, one_path, ent2relset, ent2relset_maxSetSize)  # 存储的是实体En+1与(关系Rn, ...)

            # pad
            valid_size = len(one_path)
            if valid_size > max_path_len:
                max_path_len = valid_size
            pad_size = maxPathLen - valid_size
            if pad_size > 0:
                one_path = [0] * pad_size + one_path  # 若关系集合长度小于5在前面需要补0
                ent_path = ent_path[:1] * (pad_size + 1) + ent_path[1:] # 将头实体在集合前面补全至集合长度为6
                one_mask = [0.0] * pad_size + [1.0] * valid_size  # 将在标签集合中补0至集合长度为5
            else:
                one_path = one_path[-maxPathLen:]  # select the last max_len relations
                ent_path = ent_path[:1] + ent_path[-maxPathLen:]
                one_mask = [1.0] * maxPathLen

            if file_id == 0:  # 训练集
                if len(ent_path) != maxPathLen + 1 or len(one_path) != maxPathLen:
                    print('line:', line)
                    exit(0)
                train_paths_store.append(one_path)  # 数据集中的关系列表
                train_ents_store.append(ent_path)  # 数据集中的实体列表
                train_masks_store.append(one_mask)  # 数据集中的标签列表
            if file_id == 1:  # 验证集
                if len(ent_path) != maxPathLen + 1 or len(one_path) != maxPathLen:
                    print('line:', line)
                    exit(0)
                dev_paths_store.append(one_path)
                dev_ents_store.append(ent_path)
                dev_masks_store.append(one_mask)
            if file_id == 2:  # 测试集
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


if __name__ == "__main__":
    root = "/home/shent/.conda/envs/GCNConv/SSROP/generate_path/data/FB15k-237"
    files = ['train_paths.pt','path_store_dev_5.pt','path_store_test_5.pt']
    corpus, tuple2tailset, rel2tailset = load_data(root, files)
    filename = osp.join(root, files[0])
    read_pathfile = torch.load(filename)
    train_set = corpus[0]  # 训练数据集
    train_paths_store = train_set[0]  # 关系列表
    train_masks_store = train_set[1]  # 标签列表
    train_ents_store = train_set[2]  # 实体列表
    print(read_pathfile[:10])
    print(train_paths_store[:10])
    print(train_masks_store[:10])
    print(train_ents_store[:10])