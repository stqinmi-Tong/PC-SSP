import os

import numpy as np
import torch
from tqdm import tqdm
from preprocess import build_data
from collections import Counter
import random
# %%matplotlib inline  #IPython notebook中的魔法方法，这样每次运行后可以直接得到图像，不再需要使用plt.show()
import seaborn as sns  #习惯上简写成sns
import matplotlib.pyplot as plt
sns.set()           #切换到seaborn的默认运行配置
def load_all_triples():
    rootpath= './data/FB15k-237/'
    files=['train.txt', 'dev.txt', 'test.txt']
#     triple2id={}
    tuple2tailset={} ##tuple2tailset={tuple(head,rel):tail_set},包含正反三元组关系
    for file_id, file in enumerate(files):
        readfile=open(rootpath+file, 'r')
        for line in readfile:
            parts=line.strip().split()
            length=len(parts)
            if length ==3 or (length ==4 and parts[3]=='1'):
                tuple=(parts[0], parts[1]) ###tuple是头实体和关系的对
                tail=parts[2]
                r_tuple=(parts[2], '**'+parts[1])###r_tuple是逆关系
                r_tail=parts[0] ##之前的头实体变成了逆关系中的尾实体
                exist_tailset = tuple2tailset.get(tuple) #Python 字典(Dictionary) get() 函数返回指定键的值。
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
                print(line_co, '....') ###记录并显示处理的进度

    print('recovered file produced ... over')  ###路径数据处理完成
    return path_store




def get_graph(train_data,validation_data,test_data):###生成图，是一个字典，以每个源实体为keys
      # graph = {'A': [('B', 'ab'), ('C', 'ac'), ('D', 'ad')], 'B': [('E','be')],...}
        graph_train = {}
        graph_dev = {}
        graph_test = {}
        graph = {}
        # all_tiples_train应该是[[1,2,1],[3,4,1][5,6,1],[7,8,1],...],每项中是尾实体、头实体和关系），
        # 第三项是label标签，如果有关系就为1，没标签就是0
        # train_adj_matrix[0]是adj_indices, train_adj_matrix[1]是adj_values，
        adj_indices = torch.LongTensor([train_data[1][0], train_data[1][1]])  # rows and columns
        adj_values = torch.LongTensor(train_data[1][2])
        train_adj_matrix = (adj_indices, adj_values)

        adj_indices_dev = torch.LongTensor([validation_data[1][0], validation_data[1][1]])  # rows and columns
        adj_values_dev = torch.LongTensor(validation_data[1][2])
        dev_adj_matrix = (adj_indices_dev, adj_values_dev)

        adj_indices_test = torch.LongTensor([test_data[1][0], test_data[1][1]])  # rows and columns
        adj_values_test = torch.LongTensor(test_data[1][2])
        test_adj_matrix = (adj_indices_test, adj_values_test)

        all_tiples_train = torch.cat([train_adj_matrix[0].transpose(0, 1), train_adj_matrix[1].unsqueeze(1)], dim=1)
        all_tiples_dev = torch.cat([dev_adj_matrix[0].transpose(0, 1), dev_adj_matrix[1].unsqueeze(1)],dim=1)
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

        for data in all_tiples_dev:
            source_dev = data[1].data.item()
            target_dev = data[0].data.item()
            value_dev = data[2].data.item()
            if(source_dev not in graph_dev.keys()):
                graph_dev[source_dev] = []
                graph_dev[source_dev].append((target_dev,value_dev))
            else:
                graph_dev[source_dev].append((target_dev,value_dev))
            if (target_dev not in graph_dev.keys()):
                graph_dev[target_dev] = []
            if (source_dev not in graph.keys()):
                graph[source_dev] = []
                graph[source_dev].append((target_dev, value_dev))
            else:
                graph[source_dev].append((target_dev, value_dev))
            if (target_dev not in graph.keys()):
                graph[target_dev] = []
        print("Dev_Graph created")
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
        return graph_train, graph_dev, graph_test,graph

def findAllPath(graph, start, end, flag, path=[]):
    if flag == 0:

        path = path + [start]
        if start == end:
            return [path]
        paths = []  # 存储全部路径
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
        paths = []  # 存储全部路径
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
        paths = []  # 存储全部路径

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
        paths = []  # 存储全部路径
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
        # for i in [5]:
        #     path = findAllPaths(graph, node, 0, i, 0)   #(graph, start, distance,length,flag,path=[])
        #     if path == []:
        #         continue
        #     for p in path:
        #         allpaths.append(p)
    return allpaths

###查看每个实体的度
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

##统计最长路径的长度
def path_length():
    dev_paths = torch.load("./data/FB15k-237/dev_paths.pt")
    max_length = 0
    for i in dev_paths:
        l = len(i)
        if l > max_length:
            max_length = l
    print(max_length)

###统计图里的最大邻居节点数和平均节点数
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

####绘制每个图中节点的邻接节点个数的频率分布图
def statistics_neighbors(graph):
    ###绘制图中每个实体的邻居个数频率
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
def draw_neighbor_freq(graph_train,graph_dev,graph_test):
    x_train = statistics_neighbors(graph_train)
    x_dev = statistics_neighbors(graph_dev)
    x_test = statistics_neighbors(graph_test)
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.45)
    fig, axes = plt.subplots(1, 3, figsize=(20, 12))
    ax1 = sns.histplot(x_train, kde=False, ax=axes[0])
    ax2 = sns.histplot(x_dev, kde=False, ax=axes[1])
    ax3 = sns.histplot(x_test, kde=False, ax=axes[2])
    axes_list = [ax1, ax2, ax3]
    ax1.set_xlabel("neighbors count of train graph")
    ax2.set_xlabel("neighbors count of dev graph")
    ax3.set_xlabel("neighbors count of test graph")
    # 给直方图添加文字注释（就是在每一个bar的上方加文字）
    for ax in axes_list:
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(
                    # 文字内容
                    text=f"{p.get_height():1.0f}",
                    # 文字的位置
                    xy=(p.get_x() + p.get_width() / 2., p.get_height()),
                    xycoords='data',
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='black',
                    # 文字的偏移量
                    xytext=(0, 7),
                    textcoords='offset points',
                    clip_on=True,  # <---  important
                )
    plt.suptitle('Neighbor nodes count of entities in WN18RR',  # 标题名称
                 x=0.5,  # x轴方向位置
                 y=0.93,  # y轴方向位置
                 size=15,  # 大小
                 ha='center',  # 水平位置，相对于x,y，可选参数：{'center', 'left', right'}, default: 'center'
                 va='top',  # 垂直位置，相对于x,y，可选参数：{'top', 'center', 'bottom', 'baseline'}, default: 'top'
                 weight='bold')
    plt.savefig(os.path.join("./data/WN18RR/pics/", "neighbor count.pdf"))
    plt.show()

if __name__ == '__main__':
   # datasets = "./data/FB15k-237/"
   datasets = "./data/WN18RR/"
   train_data, validation_data, test_data, entity2id, relation2id = build_data(datasets, is_unweigted=False, directed=True)
   graph_train, graph_dev, graph_test, graph = get_graph(train_data, validation_data, test_data)
   #
   # # print(graph_train)
   train_paths = get_path(graph_train)
   torch.save(train_paths, datasets + "train_paths_11.pt")
   print(train_paths[:10], len(train_paths))
   # dev_paths = get_path(graph_dev)
   # torch.save(dev_paths, datasets + "dev_paths.pt")
   #
   # test_paths = get_path(graph_test)
   # torch.save(test_paths, datasets + "test_paths.pt")

   # train_paths = torch.load(datasets + "train_paths_11.pt")
   # print(train_paths[:10],len(train_paths))
   # dev_paths = torch.load(datasets + "dev_paths.pt")
   # print(dev_paths[:10], len(dev_paths))
   # test_paths = torch.load(datasets + "test_paths.pt")
   # print(test_paths[:10], len(test_paths))


   # paths = get_path(graph)
   # torch.save(paths, datasets + "paths.pt")
   #
   # paths = torch.load(datasets + "paths.pt")
   # print(paths[:10], len(paths))
   # np.random.seed(0)
   # shuffled_index = np.random.permutation(len(paths))
   # split_train_index = int(len(paths)*0.8)
   # split_val_index = int(len(paths)*0.9)
   # train_index = shuffled_index[:split_train_index]
   # val_index = shuffled_index[split_train_index+1:split_val_index]
   # test_index = shuffled_index[split_val_index:]
   # train_path = np.array(paths)[np.array(train_index)]
   # val_path = np.array(paths)[np.array(val_index)]
   # test_path = np.array(paths)[np.array(test_index)]
   # torch.save(train_path, datasets + "train_path.pt")
   # print(train_path[:10], len(train_path))
   # torch.save(val_path, datasets + "val_path.pt")
   # print(val_path[:10], len(val_path))
   # torch.save(test_path, datasets + "test_path.pt")
   # print(test_path[:10], len(test_path))