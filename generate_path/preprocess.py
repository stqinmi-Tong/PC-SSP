import torch
import os
import numpy as np
import networkx as nx
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

def read_relation_from_id(file):
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

    return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)


def parse_line(line):
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2


def load_data(file, entity2id, relation2id, is_unweigted=False, directed=True):
    
    triples_data = []
    with open(file) as f:
        lines = f.readlines()
   
    rows, cols, data = [], [], []
    unique_entities = set()
    for line in lines:
        e1, relation, e2 = parse_line(line)
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append((entity2id[e1], relation2id[relation], entity2id[e2])) 
        if not directed:
            rows.append(entity2id[e1]) 
            cols.append(entity2id[e2]) 
            if is_unweigted:
                data.append(1)
            else:
                data.append(relation2id[relation])

        # Connecting tail and source entity
        rows.append(entity2id[e2])
        cols.append(entity2id[e1])
        if is_unweigted:
            data.append(1)
        else:
            data.append(relation2id[relation])
    print("number of unique_entities ->", len(unique_entities))
    return triples_data, (rows, cols, data)#, list(unique_entities)

def add_reversetodata(relation_reverse,dataset_path,file_path,flag): ### add reverse tripltes into train/valid/test.txt
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

def build_data_1(dataset_path, is_unweigted=False, directed=True):

    entity2id = read_entity_from_id(dataset_path)
    relation2id,rel2r_rel = read_relation_from_id(dataset_path)
    
    # add_reversetodata(rel2r_rel, dataset_path, os.path.join(dataset_path, 'train.txt'),'train')
    # add_reversetodata(rel2r_rel, dataset_path, os.path.join(dataset_path, 'valid.txt'), 'valid')
    # add_reversetodata(rel2r_rel, dataset_path, os.path.join(dataset_path, 'test.txt'), 'test')
    # print('reverse triples loading....')
   
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

        # Count number of occurences for each (e1, relation) 
        if relation2id[relation] not in left_entity:
            left_entity[relation2id[relation]] = {}
        if entity2id[e1] not in left_entity[relation2id[relation]]:
            left_entity[relation2id[relation]][entity2id[e1]] = 0
        left_entity[relation2id[relation]][entity2id[e1]] += 1 
        # Count number of occurences for each (relation, e2) 
        if relation2id[relation] not in right_entity:
            right_entity[relation2id[relation]] = {}
        if entity2id[e2] not in right_entity[relation2id[relation]]:
            right_entity[relation2id[relation]][entity2id[e2]] = 0
        right_entity[relation2id[relation]][entity2id[e2]] += 1 

    left_entity_avg = {} ### left_entity_avg：{0：5,1：7,...,}
    for i in range(len(relation2id)):
        left_entity_avg[i] = sum(
            left_entity[i].values()) * 1.0 / len(left_entity[i])

    right_entity_avg = {} ### left_entity_avg：{0：6,1：2,...,}
    for i in range(len(relation2id)):
        right_entity_avg[i] = sum(right_entity[i].values()) * 1.0 / len(right_entity[i])

    headTailSelector = {} 
    for i in range(len(relation2id)):
        headTailSelector[i] = 1000 * right_entity_avg[i] / \
            (right_entity_avg[i] + left_entity_avg[i])

    return (triples, adjacency_mat), entity2id, relation2id, headTailSelector#, unique_entities_train

def build_data(dataset_path, is_unweigted=False, directed=True):

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
                print(line_co, '....') 

    print('recovered file produced ... over')  
    return path_store
