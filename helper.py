# import gensim as gensim
import numpy as np, sys, unicodedata, requests, os, random, pdb, json
import logging, logging.config, itertools, pathlib, socket
# PyTorch related imports
import torch
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_



np.set_printoptions(precision=4)

def makeDirectory(dirpath):
	if not os.path.exists(dirpath):
		os.makedirs(dirpath)

def checkFile(filename):
	return pathlib.Path(filename).is_file()

# def getEmbeddings(embed_loc, wrd_list, embed_dims):
# 	embed_list = []
# 	model = gensim.models.KeyedVectors.load_word2vec_format(embed_loc, binary=False)
#
# 	for wrd in wrd_list:
# 		if wrd in model.vocab: 	embed_list.append(model.word_vec(wrd))
# 		else: 			embed_list.append(np.random.randn(embed_dims))
#
# 	return np.array(embed_list, dtype=np.float32)

def set_gpu(gpus):
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def get_logger(name, log_dir, config_dir):
	config_dict = json.load(open( config_dir + 'log_config.json'))
	print(config_dict)
	config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')

	# print("fffff")
	logging.config.dictConfig(config_dict)
	# print("jkjk")
	logger = logging.getLogger(name)
	# print("ttt")
	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)
	
def partition(lst, n):
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def getChunks(inp_list, chunk_size):
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]

def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))

def sp2torch(sparse_mx):
	"""Convert a scipy sparse matrix to a torch sparse tensor."""
	sparse_mx	= sparse_mx.tocoo().astype(np.float32)
	indices		= torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
	values		= torch.from_numpy(sparse_mx.data)
	shape		= torch.Size(sparse_mx.shape)
	return torch.sparse.FloatTensor(indices, values, shape)
			

def get_combined_results(left_results, right_results):
	results = {}
	count   = float(left_results['count'])

	results['left_mr']	= round(left_results ['mr'] /count, 5)
	results['left_mrr']	= round(left_results ['mrr']/count, 5)
	results['right_mr']	= round(right_results['mr'] /count, 5)
	results['right_mrr'] = round(right_results['mrr']/count, 5)
	results['mr'] = round((left_results['mr']  + right_results['mr']) /(2*count), 5)
	results['mrr'] = round((left_results['mrr'] + right_results['mrr'])/(2*count), 5)

	for k in range(10):
		# results['left_hits@{}'.format(k+1)]	= round(left_results ['hits@{}'.format(k+1)]/count, 5)
		# results['right_hits@{}'.format(k+1)]	= round(right_results['hits@{}'.format(k+1)]/count, 5)
		results['hits@{}'.format(k+1)]		= round((left_results.get('hits@{}'.format(k+1), 0.0) + right_results.get('hits@{}'.format(k+1), 0.0))/(2*count), 5)
	return results

def count_params(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_param(shape):
	param = Parameter(torch.Tensor(*shape)); 	
	xavier_normal_(param.data)
	return param