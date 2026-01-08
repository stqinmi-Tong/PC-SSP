import os

import torch
import numpy as np
from collections import defaultdict
import time
import queue
import random
from tqdm import tqdm
###训练学习的过程
class Corpus:
    def __init__(self, train_data, validation_data, test_data, entity2id,relation2id):
        self.train_triples = train_data[0]
        #print("train_data:",train_data)
        # Converting to sparse tensor
        adj_indices = torch.LongTensor([train_data[1][0], train_data[1][1]])  # rows and columns
        adj_values = torch.LongTensor(train_data[1][2])
        self.train_adj_matrix = (adj_indices, adj_values)

        adj_indices_dev = torch.LongTensor([validation_data[1][0], validation_data[1][1]])  # rows and columns
        adj_values_dev = torch.LongTensor(validation_data[1][2])
        self.dev_adj_matrix = (adj_indices_dev, adj_values_dev)

        adj_indices_test = torch.LongTensor([test_data[1][0], test_data[1][1]])  # rows and columns
        adj_values_test = torch.LongTensor(test_data[1][2])
        self.test_adj_matrix = (adj_indices_test, adj_values_test)

        # adjacency matrix is needed for train_data only, as GAT is trained for
        # training data 邻接矩阵adjacency matrix仅在训练时是必须的，因为GAT仅在训练时用到（也即仅用到训练数据）
        self.validation_triples = validation_data[0]
        self.test_triples = test_data[0]


        self.entity2id = entity2id
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.relation2id = relation2id
        self.id2relation = {v: k for k, v in self.relation2id.items()}

        self.graph_train, self.graph_dev, self.graph_test = self.get_graph()  ###得到图
        # self.node_neighbors_train, self.node_neighbors_valid,self.node_neighbors_test = self.get_further_neighbors()  ###得到邻居



    def get_graph(self):###生成图，是一个字典，以每个源实体为keys
        graph_train = {}   ####生成的图为：graph={s_e1:{t_e1:1,t_e2:0,t_e3:1,...},s_e2:{t_e1:0,t_e2:1,t_e3:0,...},...}
        graph_dev = {}
        graph_test = {}

        # all_tiples_train应该是[[1,2,1],[3,4,1][5,6,1],[7,8,1],...],每项中的前两项是源实体（头实体）和目标实体（尾实体），第三项是label标签，如果有关系就为1，没标签就是0
        # train_adj_matrix[0]是adj_indices, train_adj_matrix[1]是adj_values，

        all_tiples_train = torch.cat([self.train_adj_matrix[0].transpose(0, 1), self.train_adj_matrix[1].unsqueeze(1)], dim=1)
        all_tiples_dev = torch.cat([self.dev_adj_matrix[0].transpose(0, 1), self.dev_adj_matrix[1].unsqueeze(1)],dim=1)
        all_tiples_test = torch.cat([self.test_adj_matrix[0].transpose(0, 1), self.test_adj_matrix[1].unsqueeze(1)],dim=1)
        for data in all_tiples_train:
            source = data[1].data.item()
            target = data[0].data.item()
            value = data[2].data.item()

            if(source not in graph_train.keys()):
                graph_train[source] = {}
                graph_train[source][target] = value
            else:
                graph_train[source][target] = value
        print("Train_Graph created")
        for data in all_tiples_dev:
            source1 = data[1].data.item()
            target1 = data[0].data.item()
            value1 = data[2].data.item()
            if(source1 not in graph_dev.keys()):
                graph_dev[source1] = {}
                graph_dev[source1][target1] = value1
            else:
                graph_dev[source1][target1] = value1
        print("Dev_Graph created")
        for data in all_tiples_test:
            source2 = data[1].data.item()
            target2 = data[0].data.item()
            value2 = data[2].data.item()
            #print(source1,target1,value1)

            if(source2 not in graph_test.keys()):
                graph_test[source2] = {}
                graph_test[source2][target2] = value2
            else:
                graph_test[source2][target2] = value2
        print("Test_Graph created")
        return graph_train, graph_dev, graph_test

    def bfs(self, graph, source):####宽度优先搜索 要改进的地方，寻找源实体的多跳邻居
        visit = {}
        distance = {}
        parent = {}
        distance_lengths = {}

        visit[source] = 1
        distance[source] = 0
        parent[source] = (-1, -1)

        q = queue.Queue()
        q.put((source, -1))
        while(not q.empty()):
            top = q.get()
            if top[0] in graph.keys():
                for target in graph[top[0]].keys():###处理每一个source实体与它对应字典里的target实体之间的信息：是否访问过visit[target]，两实体之间的距离distance[target]，
                    # 以及实体的父母节点parent[target]
                    if(target in visit.keys()):###target实体在source实体的访问名单里已经有了
                        continue
                    else: ###target实体还没在source实体的访问名单里
                        q.put((target, graph[top[0]][target]))###将（target，当前source实体对应target实体的value）存进队列

                        distance[target] = distance[top[0]] + 1  ####target实体到源实体的距离

                        visit[target] = 1

                        if distance[target] not in distance_lengths.keys():
                            distance_lengths[distance[target]] = 1

                        if distance[target] > 2:
                            continue

                        parent[target] = (top[0], graph[top[0]][target])
                        ###如果source实体与target实体之间的距离不大于2，那么target实体的父母节点为（该source节点，graph[source][target]值）



        neighbors = {} ###得到关于距离的邻居字典： neighbors ={1：（（rels_turple）,(entities_turple)）,2:((),())}，也就是统计了离source entity距离为1，2，。。。的邻居集合以及™之间的关系
        for target in visit.keys():###只取source可到达的邻居中与它距离为2（nbd_size=2）的
            if(distance[target] != 5):
                continue
            if (distance[target] != 5):
                print('1')
            if target not in parent.keys():
                continue
            edges = [-1, parent[target][1]]
            relations = []
            entities = [target]
            temp = target
            while(parent[temp] != (-1, -1)):
                relations.append(parent[temp][1])
                entities.append(parent[temp][0])
                temp = parent[temp][0]

            if(distance[target] in neighbors.keys()):
                neighbors[distance[target]].append(
                    (tuple(relations), tuple(entities[:-1])))
            else:
                neighbors[distance[target]] = [
                    (tuple(relations), tuple(entities[:-1]))]
        return neighbors

    def get_further_neighbors(self):####寻找源实体的多跳邻居,把nbd_size参数去掉
        neighbors_train = {}
        neighbors_valid = {}
        neighbors_test = {}

        print("length of train_graph keys is ", len(self.graph_train.keys()),
              "length of dev_graph keys is ", len(self.graph_dev.keys()),
              "length of test_graph keys is ",len(self.graph_test.keys()))

        for i in tqdm(range(len(self.graph_train.keys()))):

            source = list(self.graph_train.keys())[i]
            temp_neighbors = self.bfs(self.graph_train, source)####得到source实体的邻居字典，把nbd_size参数去掉
            for distance in temp_neighbors.keys():
                if(source in neighbors_train.keys()):
                    if(distance in neighbors_train[source].keys()):
                        neighbors_train[source][distance].append(
                            temp_neighbors[distance])
                    else:
                        neighbors_train[source][distance] = temp_neighbors[distance]
                else:
                    neighbors_train[source] = {}
                    neighbors_train[source][distance] = temp_neighbors[distance]

        for i in tqdm(range(len(self.graph_dev.keys()))):

            source = list(self.graph_dev.keys())[i]
            temp_neighbors = self.bfs(self.graph_dev, source)####得到source实体的邻居字典，把nbd_size参数去掉
            for distance in temp_neighbors.keys():
                if(source in neighbors_valid.keys()):
                    if(distance in neighbors_valid[source].keys()):
                        neighbors_valid[source][distance].append(
                            temp_neighbors[distance])
                    else:
                        neighbors_valid[source][distance] = temp_neighbors[distance]
                else:
                    neighbors_valid[source] = {}
                    neighbors_valid[source][distance] = temp_neighbors[distance]

        for i in tqdm(range(len(self.graph_test.keys()))):

            source = list(self.graph_test.keys())[i]
            temp_neighbors = self.bfs(self.graph_test, source)  ####得到source实体的邻居字典
            for distance in temp_neighbors.keys():
                if (source in neighbors_test.keys()):
                    if (distance in neighbors_test[source].keys()):
                        neighbors_test[source][distance].append(
                            temp_neighbors[distance])
                    else:
                        neighbors_test[source][distance] = temp_neighbors[distance]
                else:
                    neighbors_test[source] = {}
                    neighbors_test[source][distance] = temp_neighbors[distance]

        torch.save(neighbors_train, './data/FB15k-237/neighbors_train_11.pt')
        torch.save(neighbors_valid,'./data/FB15k-237/neighbors_valid_11.pt')
        torch.save(neighbors_test,'./data/FB15k-237/neighbors_test_11.pt')
        print('neighbors_valid',neighbors_valid)
        return neighbors_train, neighbors_valid, neighbors_test



class Corpus_1:
    def __init__(self, args, train_data, entity2id, relation2id, get_2hop=False):
        self.train_triples = train_data[0]

        adj_indices = torch.LongTensor([train_data[1][0], train_data[1][1]])  # rows and columns
        adj_values = torch.LongTensor(train_data[1][2])
        self.train_adj_matrix = (adj_indices, adj_values)

        self.entity2id = entity2id
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.relation2id = relation2id
        self.id2relation = {v: k for k, v in self.relation2id.items()}


        file_path_train = os.path.join(args.data, 'neighbors.pt')


        self.graph_train = self.get_graph() ###得到图

        if os.path.exists(file_path_train):
                self.node_neighbors = torch.load(file_path_train)
        else:
                self.node_neighbors = self.get_further_neighbors()###得到邻居

    def get_graph(self):###生成图，是一个字典，以每个源实体为keys
        graph_train = {}   ####生成的图为：graph={s_e1:{t_e1:1,t_e2:0,t_e3:1,...},s_e2:{t_e1:0,t_e2:1,t_e3:0,...},...}

        # all_tiples_train应该是[[1,2,1],[3,4,1][5,6,1],[7,8,1],...],每项中的前两项是源实体（头实体）和目标实体（尾实体），第三项是label标签，如果有关系就为1，没标签就是0
        # train_adj_matrix[0]是adj_indices, train_adj_matrix[1]是adj_values，

        all_tiples_train = torch.cat([self.train_adj_matrix[0].transpose(0, 1), self.train_adj_matrix[1].unsqueeze(1)], dim=1)

        for data in all_tiples_train:

            source = data[1].data.item()
            target = data[0].data.item()
            value = data[2].data.item()

            if(source not in graph_train.keys()):
                graph_train[source] = {}
                graph_train[source][target] = value
            else:
                graph_train[source][target] = value
        print("Train_Graph created")
        return graph_train

    def bfs(self, graph, source):####宽度优先搜索 要改进的地方，寻找源实体的多跳邻居
        visit = {}
        distance = {}
        parent = {}
        distance_lengths = {}

        visit[source] = 1
        distance[source] = 0
        parent[source] = (-1, -1)

        q = queue.Queue()
        q.put((source, -1))
        while(not q.empty()):
            top = q.get()
            if top[0] in graph.keys():
                for target in graph[top[0]].keys():###处理每一个source实体与它对应字典里的target实体之间的信息：是否访问过visit[target]，两实体之间的距离distance[target]，
                    # 以及实体的父母节点parent[target]
                    if(target in visit.keys()):###target实体在source实体的访问名单里已经有了
                        continue
                    else: ###target实体还没在source实体的访问名单里
                        q.put((target, graph[top[0]][target]))###将（target，当前source实体对应target实体的value）存进队列

                        distance[target] = distance[top[0]] + 1  ####target实体到源实体的距离

                        visit[target] = 1 ###
                        if distance[target] > 5:
                            continue
                        parent[target] = (top[0], graph[top[0]][target])
                        ###如果source实体与target实体之间的距离不大于2，那么target实体的父母节点为（该source节点，graph[source][target]值）

                        if distance[target] not in distance_lengths.keys():
                            distance_lengths[distance[target]] = 1

        neighbors = {} ###得到关于距离的邻居字典： neighbors ={1：（（rels_turple）,(entities_turple)）,2:((),())}，也就是统计了离source entity距离为1，2，。。。的邻居集合以及™之间的关系
        for target in visit.keys():###只取source可到达的邻居中与它距离为2（nbd_size=2）的
            if(distance[target] != 5):
                continue
            if target not in parent.keys():
                continue
            relations = []
            entities = [target]
            temp = target
            while(parent[temp] != (-1, -1)):
                relations.append(parent[temp][1])
                entities.append(parent[temp][0])
                temp = parent[temp][0]

            if(distance[target] in neighbors.keys()):
                neighbors[distance[target]].append(
                    (tuple(relations), tuple(entities[:-1])))
            else:
                neighbors[distance[target]] = [
                    (tuple(relations), tuple(entities[:-1]))]
        return neighbors

    def get_further_neighbors(self):####寻找源实体的多跳邻居,把nbd_size参数去掉
        neighbors_train = {}

        print("length of train_graph keys is ", len(self.graph_train.keys()))

        for i in tqdm(range(len(self.graph_train.keys()))):

            source = list(self.graph_train.keys())[i]
            temp_neighbors = self.bfs(self.graph_train, source)####得到source实体的邻居字典，把nbd_size参数去掉
            for distance in temp_neighbors.keys():
                if(source in neighbors_train.keys()):
                    if(distance in neighbors_train[source].keys()):
                        neighbors_train[source][distance].append(
                            temp_neighbors[distance])
                    else:
                        neighbors_train[source][distance] = temp_neighbors[distance]
                else:
                    neighbors_train[source] = {}
                    neighbors_train[source][distance] = temp_neighbors[distance]



        torch.save(neighbors_train, './data/FB15k-237/neighbors_5.pt')

        print('ok')
        return neighbors_train



