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


