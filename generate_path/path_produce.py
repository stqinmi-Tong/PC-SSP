import os
import numpy as np
import torch
from tqdm import tqdm
from preprocess import build_data
from collections import Counter
import random
import seaborn as sns  
import matplotlib.pyplot as plt
sns.set()         
def load_all_triples():
    rootpath= './data/FB15k-237/'
    files=['train.txt', 'valid.txt', 'test.txt']
    tuple2tailset={} ##tuple2tailset={tuple(head,rel):tail_set}
    for file_id, file in enumerate(files):
        readfile=open(rootpath+file, 'r')
        for line in readfile:
            parts=line.strip().split()
            length=len(parts)
            if length ==3 or (length ==4 and parts[3]=='1'):
                tuple=(parts[0], parts[1]) 
                tail=parts[2]
                r_tuple=(parts[2], '**'+parts[1])
                r_tail=parts[0]
                exist_tailset = tuple2tailset.get(tuple) 
                if exist_tailset is  None:
                    exist_tailset = set()
                exist_tailset.add(tail)
                tuple2tailset[tuple] = exist_tailset

                r_exist_tailset = tuple2tailset.get(r_tuple)
                if r_exist_tailset is  None:
                    r_exist_tailset = set()
                r_exist_tailset.add(r_tail)
                tuple2tailset[r_tuple] = r_exist_tailset
        readfile.close()
        print(rootpath + file, '... load over, size', len(tuple2tailset))
    return tuple2tailset
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

def get_graph(train_data,validation_data,test_data):
      # graph = {'A': [('B', 'ab'), ('C', 'ac'), ('D', 'ad')], 'B': [('E','be')],...}
        graph_train = {}
        graph_val = {}
        graph_test = {}
        graph = {}
       
        adj_indices = torch.LongTensor([train_data[1][0], train_data[1][1]])  # rows and columns
        adj_values = torch.LongTensor(train_data[1][2])
        train_adj_matrix = (adj_indices, adj_values)

        adj_indices_valid = torch.LongTensor([validation_data[1][0], validation_data[1][1]])  # rows and columns
        adj_values_valid = torch.LongTensor(validation_data[1][2])
        valid_adj_matrix = (adj_indices_valid, adj_values_valid)

        adj_indices_test = torch.LongTensor([test_data[1][0], test_data[1][1]])  # rows and columns
        adj_values_test = torch.LongTensor(test_data[1][2])
        test_adj_matrix = (adj_indices_test, adj_values_test)

        all_tiples_train = torch.cat([train_adj_matrix[0].transpose(0, 1), train_adj_matrix[1].unsqueeze(1)], dim=1)
        all_tiples_valid = torch.cat([valid_adj_matrix[0].transpose(0, 1), valid_adj_matrix[1].unsqueeze(1)],dim=1)
        all_tiples_test = torch.cat([test_adj_matrix[0].transpose(0, 1), test_adj_matrix[1].unsqueeze(1)],dim=1)

        for data in all_tiples_train:
            source = data[1].data.item() #head
            target = data[0].data.item() #tail
            value = data[2].data.item() #rel

            if(source not in graph_train.keys()):
                graph_train[source] = []
                graph_train[source].append((target,value))
            else:
                graph_train[source].append((target,value))
            if (target not in graph_train.keys()):
                graph_train[target] = []
            if (source not in graph.keys()):
                graph[source] = []
                graph[source].append((target, value))
            else:
                graph[source].append((target, value))
            if (target not in graph.keys()):
                graph[target] = []
        print("Train_Graph created")

        for data in all_tiples_valid:
            source_val = data[1].data.item()
            target_val = data[0].data.item()
            value_val = data[2].data.item()
            if(source_val not in graph_val.keys()):
                graph_val[source_val] = []
                graph_val[source_val].append((target_val,value_val))
            else:
                graph_val[source_val].append((target_val,value_val))
            if (target_val not in graph_val.keys()):
                graph_val[target_val] = []
            if (source_val not in graph.keys()):
                graph[source_val] = []
                graph[source_val].append((target_val, value_val))
            else:
                graph[source_val].append((target_val, value_val))
            if (target_val not in graph.keys()):
                graph[target_val] = []
        print("val_Graph created")
        for data in all_tiples_test:
            source_test = data[1].data.item()
            target_test = data[0].data.item()
            value_test = data[2].data.item()
            #print(source1,target1,value1)

            if(source_test not in graph_test.keys()):
                graph_test[source_test] = []
                graph_test[source_test].append((target_test,value_test))
            else:
                graph_test[source_test].append((target_test,value_test))
            if (target_test not in graph_test.keys()):
                graph_test[target_test] = []

            if (source_test not in graph.keys()):
                graph[source_test] = []
                graph[source_test].append((target_test, value_test))
            else:
                graph[source_test].append((target_test, value_test))
            if (target_test not in graph.keys()):
                graph[target_test] = []
        print("Test_Graph created")
        # print(graph[40790])
        return graph_train, graph_val, graph_test,graph

def findAllPath(graph, start, end, flag, path=[]):
    if flag == 0:

        path = path + [start]
        if start == end:
            return [path]
        paths = []  
        flag = 1
        for node in graph[start]:
            if node[0] not in path:
                newpaths = findAllPath(graph, node, end, flag, path)
                for newpath in newpaths:
                    paths.append(newpath)
    else:
        path = path + [start[1]] +[start[0]]
        if start[0] == end:
            return [path]
        paths = [] 
        # print(start)
        for node in graph[start[0]]:
            if node[0] not in path:
                newpaths = findAllPath(graph, node, end, flag, path)
                for newpath in newpaths:
                    paths.append(newpath)

    return paths

def findAllPaths(graph, start, distance,length,flag,path=[]):
    if flag == 0:
        path = path + [start]
        if distance == length:
            # print('distance', distance, start, [path])
            return [path]
        distance += 1
        flag = 1
        paths = []  

        if len(graph[start]) > 8 :
            x = random.sample(graph[start], 4)
        else: x = graph[start]

        # for node in graph[start]:
        for node in x:
            # print(node)
            if node[0] not in path:
                newpaths = findAllPaths(graph, node, distance, length,flag, path)
                # print('newpaths',newpaths)
                for newpath in newpaths:
                    paths.append(newpath)
    else:
        path = path + [start[1]] +[start[0]]
        if distance >= length:
            # print('distance', distance, start, [path])
            return [path]
        distance += 1
        paths = [] 
        # print('start', start, 'distance', distance)\
        if len(graph[start[0]]) > 4:
            y = random.sample(graph[start[0]], 4)
        else: y = graph[start[0]]
        # for node in graph[start[0]]:
        for node in y:
            if node[0] not in path:
                newpaths = findAllPaths(graph, node, distance, length, flag , path)
                # print('newpaths',newpaths)
                for newpath in newpaths:
                    paths.append(newpath)
    return paths

def get_path(graph):
    allpaths = []
    l = list(graph.keys())
    for t in tqdm(range(len(l))):
        node = l[t]
        path = findAllPaths(graph, node, 0, 5, 0)   #(graph, start, distance,length,flag,path=[])
        if path == []:
            continue
        for p in path:
            allpaths.append(p)
       
    return allpaths

def entity_dgree(graph):
    count_dict = {}
    count = []
    i = 0
    for node in graph.keys():
       i=i+1
       l = len(graph[node])
       count_dict[node] = l
       count.append(str(l))
    print(i,count)

def path_length():
    val_paths = torch.load("./data/FB15k-237/valid_paths.pt")
    max_length = 0
    for i in val_paths:
        l = len(i)
        if l > max_length:
            max_length = l
    print(max_length)

def neighbor_number(graph_test):
    count = 0
    max_neighbors = 0
    count_dict = {}
    for node in graph_test.keys():
        l = len(graph_test[node])
        count_dict[node] = l
        if max_neighbors < l:
            max_neighbors = l
        count+=l
    ave_neighbors = count/len(graph_test.keys())
    print('max',max_neighbors,'ave',ave_neighbors)

def statistics_neighbors(graph):
    
    count_dict = {}
    count = []
    for node in graph.keys():
        l = len(graph[node])
        count_dict[node] = l
        count.append(str(l))

    freq = Counter(count)
    most_freq = []
    for i in list(freq.keys())[:10]:
        most_freq.append(i)
    x = []
    for i in count:
        if i in most_freq:
            x.append(i)
    x = np.array(x)
    return x
def draw_neighbor_freq(graph_train,graph_valid,graph_test):
    x_train = statistics_neighbors(graph_train)
    x_val = statistics_neighbors(graph_valid)
    x_test = statistics_neighbors(graph_test)
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.45)
    fig, axes = plt.subplots(1, 3, figsize=(20, 12))
    ax1 = sns.histplot(x_train, kde=False, ax=axes[0])
    ax2 = sns.histplot(x_val, kde=False, ax=axes[1])
    ax3 = sns.histplot(x_test, kde=False, ax=axes[2])
    axes_list = [ax1, ax2, ax3]
    ax1.set_xlabel("neighbors count of train graph")
    ax2.set_xlabel("neighbors count of val graph")
    ax3.set_xlabel("neighbors count of test graph")
    
    for ax in axes_list:
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(
                    text=f"{p.get_height():1.0f}",
                    xy=(p.get_x() + p.get_width() / 2., p.get_height()),
                    xycoords='data',
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='black',
                    xytext=(0, 7),
                    textcoords='offset points',
                    clip_on=True,  # <---  important
                )
    plt.suptitle('Neighbor nodes count of entities in WN18RR',  
                 x=0.5,  
                 y=0.93,  
                 size=15,  
                 ha='center',  
                 va='top',  
                 weight='bold')
    plt.savefig(os.path.join("./data/WN18RR/pics/", "neighbor count.pdf"))
    plt.show()

if __name__ == '__main__':
   # datasets = "./data/FB15k-237/"
   datasets = "./data/WN18RR/"
   train_data, validation_data, test_data, entity2id, relation2id = build_data(datasets, is_unweigted=False, directed=True)
   graph_train, graph_valid, graph_test, graph = get_graph(train_data, validation_data, test_data)
  
   train_paths = get_path(graph_train)
   torch.save(train_paths, datasets + "train_paths.pt")
   print(train_paths[:10], len(train_paths))
 


