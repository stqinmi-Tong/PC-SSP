import torch
import os
import numpy as np
import networkx as nx

##数据预处理，获得实体id词典，关系id词典，初始实体/关系向量列表
from tqdm import tqdm


def read_entity_from_id(file):
    filename = file+'entity2id.txt'
    entity2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                entity, entity_id = line.strip().split()[0].strip(), line.strip().split()[1].strip()
                entity2id[entity] = int(entity_id)
    return entity2id

def read_relation_from_id(file):##读取关系id并添加逆关系列表
    filename = file + 'relation2id.txt'
    rel_num = 0
    rel2id = {}
    r_rel = []
    rel2r_rel = {}
    with open(filename, 'r') as f:
        for line in f:
            rel_num += 1
            if len(line.strip().split()) > 1:
                relation, relation_id = line.strip().split()[0].strip(), line.strip().split()[1].strip()
                rel2id[relation] = int(relation_id)
                r_r = 'r' + relation
                rel2r_rel[relation] = r_r
                r_rel.append(r_r)
    for i, r_r in enumerate(r_rel):
        rel2id[r_r] = int(i+rel_num)
    return rel2id,rel2r_rel
def init_embeddings(entity_file, relation_file):
    entity_emb, relation_emb = [], []

    with open(entity_file) as f:
        for line in f:
            entity_emb.append([float(val) for val in line.strip().split()])

    with open(relation_file) as f:
        for line in f:
            relation_emb.append([float(val) for val in line.strip().split()])

    return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)##转换成数组


def parse_line(line):###定义三元组解析函数，解析出头尾实体和关系
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2


def load_data(file, entity2id, relation2id, is_unweigted=False, directed=True):
    # with open("./data/FB15k-237/train.txt") as f:
    #     lines_train = f.readlines()
    # with open("./data/FB15k-237/valid.txt") as f:
    #     lines_valid = f.readlines()
    # with open("./data/FB15k-237/test.txt") as f:
    #     lines_test = f.readlines()
    # triples_data = []
    # lines = lines_train+lines_valid+lines_test
    triples_data = []
    with open(file) as f:
        lines = f.readlines()
    # 对于稀疏张量，行列表包含对应的稀疏张量行，cols列表包含对应的稀疏张量列，data包含关系类型
    # Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
    # type they are connected with 实体的邻接矩阵是无向的,源实体和尾部实体应该知道它们所连接的关系类型
    rows, cols, data = [], [], []
    unique_entities = set()
    for line in lines:
        e1, relation, e2 = parse_line(line)
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append((entity2id[e1], relation2id[relation], entity2id[e2])) #转换为实体和关系对应的id，triples_data=[(3,6,9),(2,4,8),...]
        if not directed:
                # Connecting source and tail entity 源实体和尾实体相连
            rows.append(entity2id[e1]) ##源实体
            cols.append(entity2id[e2]) ##尾实体
            if is_unweigted:
                data.append(1) ##如果没有关系标签
            else:
                data.append(relation2id[relation])##如果存在具体的关系类型标签，则将关系id添加进data

        # Connecting tail and source entity 尾实体与源实体相连
        rows.append(entity2id[e2])##尾实体,rows=[9,8,...]
        cols.append(entity2id[e1])##头实体,cols=[3,2,...]
        if is_unweigted:
            data.append(1)##如果不存在关系
        else:
            data.append(relation2id[relation])##如果存在关系，则将关系id添加进data,data=[6,4,...,1,..,1,26,...]
    print("number of unique_entities ->", len(unique_entities))
    return triples_data, (rows, cols, data)#, list(unique_entities)

def add_reversetodata(relation_reverse,dataset_path,file_path,flag): ### add reverse tripltes into train/dev/test.txt
    triples = []
    with open(os.path.join(file_path)) as f:
        lines = f.readlines()
    for line in lines:
        e1, relation, e2 = parse_line(line)
        rel_r = relation_reverse[relation]
        l = [e1, relation, e2]
        l_r = [e2, rel_r, e1]
        triples.append(l)
        triples.append(l_r)

    if flag == 'train':
        with open(dataset_path + 'train_r.txt','w') as f:
            for t in triples:
                f.writelines(str(t[0]) + ' ' + str(t[1]) + ' ' + str(t[2])+ '\n')
    elif flag == 'test':
        with open(dataset_path + 'test_r.txt','w') as f:
            for t in triples:
                f.writelines(str(t[0]) + ' ' + str(t[1]) + ' ' + str(t[2]) + '\n')
    else:
        with open(dataset_path + 'valid_r.txt','w') as f:
            for t in triples:
                f.writelines(str(t[0]) + ' ' + str(t[1]) + ' ' + str(t[2]) + '\n')
    return triples

def build_data_1(dataset_path, is_unweigted=False, directed=True):###是为了采样bern

    entity2id = read_entity_from_id(dataset_path)
    relation2id,rel2r_rel = read_relation_from_id(dataset_path)
    # ##生成逆三元组
    # add_reversetodata(rel2r_rel, dataset_path, os.path.join(dataset_path, 'train.txt'),'train')
    # add_reversetodata(rel2r_rel, dataset_path, os.path.join(dataset_path, 'valid.txt'), 'valid')
    # add_reversetodata(rel2r_rel, dataset_path, os.path.join(dataset_path, 'test.txt'), 'test')
    # print('reverse triples loading....')
    #
    # train_triples, train_adjacency_mat, unique_entities_train = load_data(os.path.join(
    #     dataset_path, 'train_r.txt'), entity2id, relation2id, is_unweigted, directed)
    #
    # validation_triples, valid_adjacency_mat, unique_entities_validation = load_data(
    #     os.path.join(dataset_path, 'valid_r.txt'), entity2id, relation2id, is_unweigted, directed)
    #
    # test_triples, test_adjacency_mat, unique_entities_test = load_data(os.path.join(
    #     dataset_path, 'test_r.txt'), entity2id, relation2id, is_unweigted, directed)

    triples, adjacency_mat = load_data(entity2id, relation2id, is_unweigted, directed)

    left_entity, right_entity = {}, {}

    with open(os.path.join(dataset_path, 'train_r.txt')) as f:
        lines = f.readlines()

    for line in lines:
        e1, relation, e2 = parse_line(line)

        # Count number of occurences for each (e1, relation) 统计每个(e1, relation)的共现数
        if relation2id[relation] not in left_entity:
            left_entity[relation2id[relation]] = {}
        if entity2id[e1] not in left_entity[relation2id[relation]]:
            left_entity[relation2id[relation]][entity2id[e1]] = 0
        left_entity[relation2id[relation]][entity2id[e1]] += 1 ####得到的left_entity词典：{{rel_1_id:{ent_1_id:3,ent_2_id:6,...},rel_2_id:{}}}

        # Count number of occurences for each (relation, e2) 统计每个(relation, e2)的共现数
        if relation2id[relation] not in right_entity:
            right_entity[relation2id[relation]] = {}
        if entity2id[e2] not in right_entity[relation2id[relation]]:
            right_entity[relation2id[relation]][entity2id[e2]] = 0
        right_entity[relation2id[relation]][entity2id[e2]] += 1 ####得到的right_entity词典：{{rel_1_id:{ent_1_id:3,ent_2_id:6,...},}}

    left_entity_avg = {} ### left_entity_avg：{0：5,1：7,...,},指的是每个关系平均有几个左实体与它相连，例如，关系0平均有5个头实体与之相连
    for i in range(len(relation2id)):
        left_entity_avg[i] = sum(
            left_entity[i].values()) * 1.0 / len(left_entity[i])

    right_entity_avg = {} ### left_entity_avg：{0：6,1：2,...,},指的是每个关系平均有几个右实体与它相连，例如，关系0平均有6个尾实体与之相连
    for i in range(len(relation2id)):
        right_entity_avg[i] = sum(right_entity[i].values()) * 1.0 / len(right_entity[i])

    headTailSelector = {} ###Bern负采样计算公式：tph/（tph+hpt），根据公式选择是替换头实体还是尾实体 headTailSelector：{0：，1：，}， 每个关系对应的Bern计算得分
    for i in range(len(relation2id)):
        headTailSelector[i] = 1000 * right_entity_avg[i] / \
            (right_entity_avg[i] + left_entity_avg[i])

    return (triples, adjacency_mat), entity2id, relation2id, headTailSelector#, unique_entities_train

def build_data(dataset_path, is_unweigted=False, directed=True):###是为了采样bern

    entity2id = read_entity_from_id(dataset_path)
    relation2id,rel2r_rel = read_relation_from_id(dataset_path)

    triples, adjacency_mat  = load_data(os.path.join(
        dataset_path, 'train.txt'),entity2id, relation2id, is_unweigted, directed)
    validation_triples, valid_adjacency_mat = load_data(os.path.join(
        dataset_path, 'valid.txt'), entity2id, relation2id, is_unweigted, directed)
    test_triples, test_adjacency_mat = load_data(os.path.join(
        dataset_path, 'test.txt'),entity2id, relation2id, is_unweigted, directed)

    return (triples, adjacency_mat),(validation_triples, valid_adjacency_mat),\
           (test_triples, test_adjacency_mat), entity2id, relation2id

def recover_entities_for_guu_paths(ent_nighbors):
    print('recovering entity path file')
    path_store = []
    for i in tqdm(range(len(ent_nighbors.keys()))):
        s_ent = list(ent_nighbors.keys())[i]
        line_co=0

        for path_distance in ent_nighbors[s_ent]:
            path_turples=ent_nighbors[s_ent][path_distance]
            for path in path_turples:
                rel_list = list(path[0])
                ent_list = list(path[1])
                path_single = [s_ent]
                path_str = str(s_ent) + '\t'
                if len(ent_list) != len(rel_list):
                    print('the length of ent_list and rel_list are not equal')
                    exit(0)
                for i in range(len(rel_list)):
                    path_str +=  str(rel_list[i])+'\t' + str(ent_list[i]) +'\t'
                    path_single.append(rel_list[i])
                    path_single.append(ent_list[i])
                path_str += '\n'
                path_store.append(path_single)

            line_co+=1
            if line_co % 1000==0:
                print(line_co, '....') ###记录并显示处理的进度

    print('recovered file produced ... over')  ###路径数据处理完成
    return path_store