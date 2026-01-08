import logging.config
from pprint import pprint

from helper import *
from torch.utils import data
import argparse
from Data.dataset import *
from Data.preprocessing import load_data
from generate_path.preprocess import read_entity_from_id, read_relation_from_id
from P2E import *
from P2P import *
import time
from timeit import default_timer as timer
import os
import logging
import torch.optim as optim
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CPC")
    parser.add_argument("--name", default='testrun', help="Set filename for saving or restoring models")
    parser.add_argument("--batch-size", type=int, default=256, help="Number of path sent to the network in one step.")
    parser.add_argument("--data_dir", type=str, default="./generate_path/data/WN18RR/", help="")
    parser.add_argument('-logdir', dest="log_dir", default='./log/', help='Log directory')
    parser.add_argument("--snapshot_dir", type=str, default="./snapshot", help="")
    parser.add_argument("--hidden-size", type=int, default=2000, help="Number of hidden layer")
    parser.add_argument("--embedding-size", type=int, default=2000, help="Number of embedding layer")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="")#0.0001
    parser.add_argument("--momentum", type=float, default=0.9, help="")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="")
    parser.add_argument("--num-workers", type=int, default=4, help="")
    parser.add_argument("--epochs", type=int, default=200, help="")
    parser.add_argument("--l2-weight", type=float, default=0.001, help="")
    parser.add_argument("--max-performance", type=float, default=0.46, help="")
    parser.add_argument("--div-reg", type=float, default=0.1, help="")
    parser.add_argument("--margin", type=float, default=0.3, help="")
    parser.add_argument("--max-length", type=int, default=2, help="")
    parser.add_argument("--neg-size", type=int, default=64, help="")
    parser.add_argument("--power", type=float, default=0.9, help="")
    parser.add_argument("--cuda", type=int, default=1, help="")
    parser.add_argument('-config', dest="config_dir", default='./config/',  help='Config directory')
    parser.add_argument('-restore', dest="restore", action='store_true', help='Restore from the previously saved model')
    parser.add_argument("-gpu", type=str, default='1',  help="Set GPU Ids")
    parser.add_argument("-mode", type=str, default='Element',  help="set Model level : Element or Subpath")
    parser.add_argument('-seed', dest="seed", default=42, type=int, help='Seed for randomization')

    parser.add_argument('-opt', dest="opt", default='adam', help='GPU to use')
    parser.add_argument('-bias', dest="bias", action='store_true', help='Restore from the previously saved model')

    return parser.parse_args()

class Main(object):
    def load_data(self, args):

        args.entity2id = read_entity_from_id(args.data_dir)
        args.relation2id,_ = read_relation_from_id(args.data_dir)
        self.ent_vocab_size = len(args.entity2id)  
        self.rel_vocab_size = len(args.relation2id) + 1 

        if args.mode == 'Element':
            self.corpus, self.tuple2tailset, self.rel2tailset = load_data(args.data_dir, args.train_dataset,
                                                                          args.max_length)

            args.train_set = self.corpus[0]
            args.valid_set = self.corpus[1]
            args.test_set = self.corpus[2]
            # args.tuple2tailset = self.tuple2tailset  ##{(En, Rn): set(En+1, ...)}
            # args.rel2tailset = self.rel2tailset  ##{Rn: set(En+1, ...)}

            # train_dataloader
            self.train_dataset = Train_Dataset(args)
            self.train_dataset_size = len(self.train_dataset) 
            train_indices = list(range(self.train_dataset_size))

            np.random.shuffle(train_indices) 
            # self.train_sampler = data.sampler.SubsetRandomSampler(indices)

            self.train_sampler = data.sampler.SubsetRandomSampler(np.random.choice(range(self.train_dataset_size),
                                                                                   int(1*self.train_dataset_size)))#0.05
            self.data_loader = data.DataLoader(self.train_dataset, batch_size=args.batch_size,
                                                sampler=self.train_sampler,
                                                num_workers=args.num_workers, drop_last=True)
            # dev_dataloader
            self.valid_dataset = Valid_Dataset(args)
            # print(self.valid_dataset.__getitem__(2))
            valid_indices = list(range(len(self.valid_dataset)))

            np.random.shuffle(valid_indices)
            self.valid_sampler = data.sampler.SubsetRandomSampler(valid_indices)
            self.valid_loader = data.DataLoader(self.valid_dataset, batch_size=args.batch_size,
                                                sampler=self.valid_sampler,
                                                num_workers=args.num_workers
                                                )
            self.test_dataset = Test_Dataset(args)
            test_indices = list(range(len(self.test_dataset)))

            np.random.shuffle(test_indices)
            self.test_sampler = data.sampler.SubsetRandomSampler(test_indices)
            self.test_loader = data.DataLoader(self.test_dataset, batch_size=args.batch_size,
                                               sampler=self.test_sampler,
                                               num_workers=args.num_workers)
        if args.mode == 'Subpath':
            self.train_dataset = Train_Path_Dataset(args)
            # self.valid_dataset = Valid_Path_Dataset(args)
            # self.test_dataset = Test_Path_Dataset(args)

            train_indices = list(range(len(self.train_dataset)))
            np.random.shuffle(train_indices) 
            train_sampler = data.sampler.SubsetRandomSampler(
                np.random.choice(range(len(self.train_dataset)), int(0.05 * len(self.train_dataset))))
            self.data_loader = data.DataLoader(self.train_dataset,
                                               batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               drop_last=True
                                               )#0.02

        if args.mode == 'joint':
            self.corpus, _,_ = load_data(args.data_dir, args.train_dataset, args.max_length)
            args.train_set = self.corpus[0]

            self.train_dataset_j = Train_Joint_Dataset(args)
            train_indices_j = list(range(len(self.train_dataset_j)))
            np.random.shuffle(train_indices_j)  
            train_sampler_j = data.sampler.SubsetRandomSampler(np.random.choice(range(len(self.train_dataset_j)),
                                                                                int(0.02*len(self.train_dataset_j))))#0.02
            self.data_loader = data.DataLoader(self.train_dataset_j,
                                           batch_size=args.batch_size,
                                           sampler=train_sampler_j,
                                           drop_last=True
                                           )

    def __init__(self, args):
        self.p = args
        self.logger = self.get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
        self.k_size = 3
        self.logger.info(vars(self.p))
        self.lambda_w =1
        self.lambda_s = 1
        if self.p.gpu != '-1' and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
            torch.cuda.set_device(2)
            self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
            print('device',self.device)
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')


        self.load_data(args)
       
        if  self.p.mode == 'Element':
            self.model = CPC_word(self.ent_vocab_size, self.rel_vocab_size, 200, 200, 5, self.k_size, 5).to(self.device)
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=0.01, betas=(0.9, 0.98), eps=1e-09, weight_decay=0, amsgrad=True)
            total = sum([param.nelement() for param in self.model.parameters()])
            print("Number of parameter: %.2fM" % (total / 1e6))
        elif self.p.mode == 'Subpath':
            self.model = CPC_sentence(self.ent_vocab_size, self.rel_vocab_size, 2000,2000,2000, 5, self.k_size, 11,self.p.bias).to(self.device)
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=0.000005, betas=(0.9, 0.98), eps=1e-09, weight_decay=0, amsgrad=True)
            total = sum([param.nelement() for param in self.model.parameters()])
            print("Number of parameter: %.2fM" % (total / 1e6))
        else:
            self.model_w = CPC_word(self.ent_vocab_size, self.rel_vocab_size,
                                  200, 200, 5,
                                    self.k_size, 5).to(self.device)
            self.model_s = CPC_sentence(self.ent_vocab_size, self.rel_vocab_size,
                                        200, 200, 200,
                                        5, self.k_size, 11,
                                      self.p.bias).to(self.device)
            self.optimizer = optim.Adam([{'params': self.model_w.parameters(), 'lr':0.0001},
                                                 {'params': self.model_s.parameters(), 'lr': 0.0001}],
                                        betas=(0.9, 0.98), eps=1e-09, weight_decay=0, amsgrad=True)
            total = sum([param.nelement() for param in self.model_w.parameters()])+sum([param.nelement() for param in self.model_s.parameters()])
            print("Number of parameter: %.2fM" % (total / 1e6),)


    def lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    def adjust_learning_rate(self, optimizer, i_iter, num_batches):
        lr = self.lr_poly(args.learning_rate, i_iter, num_batches, args.power)
        optimizer.param_groups[0]['lr'] = lr


    def get_logger(self,name, log_dir, config_dir):
        config_dict = json.load(open(config_dir + 'log_config.json'))
        config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
        logging.config.dictConfig(config_dict)
        logger = logging.getLogger(name)
        std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logging.Formatter(std_out_format))
        logger.addHandler(consoleHandler)

        return logger

    def save_model(self, save_path):
        state = {
            'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_test': self.best_test,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        state = torch.load(load_path)
        state_dict = state['state_dict']
        self.best_test = state['best_test']
        self.best_val = state['best_val']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def snapshot(self, dir_path, run_name, state):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        snapshot_file = os.path.join(dir_path,run_name + '-model_best.pth')

        torch.save(state, snapshot_file)
        self.logger.info("Snapshot saved to {}\n".format(snapshot_file))


    def train(self, step, epoch, log_interval):

        if args.mode == 'Element' or args.mode == 'Subpath':
            self.model.train()
            total_loss = {i: 0.0 for i in range(1, self.k_size + 1)}
            total_acc = {i: 0.0 for i in range(1, self.k_size + 1)}
            for batch_idx, data in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                self.adjust_learning_rate(self.optimizer, batch_idx, len(self.data_loader))
                if self.p.mode == 'Element':
                    relation, mask, entity = data
                    relation = relation.long().to(self.device)
                    entity = entity.long().to(self.device)
                    loss, accuracy = self.model.forward(entity, relation, self.device)
                else:
                    data = data.long().to(self.device)
                    loss, accuracy = self.model.forward(data, self.device)

                acc = torch.mean(accuracy, 0)
                loss = torch.mean(loss, 0)  # torch.Size([batch_size，3])
                step += 1
                for i, (a, l) in enumerate(zip(acc, loss)):
                    total_loss[i + 1] += l.detach().item() 
                    total_acc[i + 1] += a.detach().item()

                loss.sum().backward()
                self.optimizer.step()
                if batch_idx % log_interval == 0:
                    self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                        epoch, batch_idx * (self.p.batch_size), len(self.data_loader) * self.p.batch_size,
                               100. * batch_idx / len(self.data_loader), acc.mean().detach().item(),
                        loss.mean().detach().item()))
        else:
            self.model_w.train()
            self.model_s.train()
            total_loss = {i: 0.0 for i in range(1, self.k_size + 1)}
            total_acc = {i: 0.0 for i in range(1, self.k_size + 1)}
            total_loss_w = {i: 0.0 for i in range(1, self.k_size + 1)}
            total_loss_s = {i: 0.0 for i in range(1, self.k_size + 1)}
            total_acc_w = {i: 0.0 for i in range(1, self.k_size + 1)}
            total_acc_s = {i: 0.0 for i in range(1, self.k_size + 1)}
            for batch_idx, data in enumerate(self.data_loader):
                self.optimizer.zero_grad()

                self.adjust_learning_rate(self.optimizer, batch_idx, len(self.data_loader))

                relation, mask, entity, path = data
                relation = relation.long().to(self.device)
                entity = entity.long().to(self.device)
                path = path.long().to(self.device)

                loss_w, accuracy_w = self.model_w.forward(entity, relation, self.device)
                loss_s, accuracy_s = self.model_s.forward(path, self.device)

                for i, (l_w, l_s) in enumerate(zip(torch.mean(loss_w, 0), torch.mean(loss_s, 0))):
                    total_loss_w[i + 1] += l_w.detach().item()
                    total_loss_s[i + 1] += l_s.detach().item()

                loss = self.lambda_w * loss_w + self.lambda_s *loss_s

                mean_w = sum(total_loss_w.values()) / len(total_loss_w.keys())
                mean_s = sum(total_loss_s.values()) / len(total_loss_s.keys())
                self.lambda_w = mean_s / (mean_w + 1e-12)  # ~0.936
                self.lambda_s = 1.0

                acc_w = torch.mean(accuracy_w, 0)
                acc_s = torch.mean(accuracy_s, 0)
                loss = torch.mean(loss, 0)  # torch.Size([batch_size，3])
                step += 1
                for i, (a_w, a_s, l) in enumerate(zip(acc_w, acc_s, loss)):
                    total_loss[i + 1] += l.detach().item()  
                    total_acc_w[i + 1] += a_w.detach().item()
                    total_acc_s[i + 1] += a_s.detach().item()
                    total_acc[i + 1] = (total_acc_w[i + 1] + total_acc_s[i + 1]) * 0.5
                loss.sum().backward()
                self.optimizer.step()


                if batch_idx % log_interval == 0:
                    self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tAccuracy_w: {:.4f}\tAccuracy_s: {:.4f}\tLoss: {:.6f}'.format(
                        epoch, batch_idx * (self.p.batch_size), len(self.data_loader) * self.p.batch_size,
                               100. * batch_idx / len(self.data_loader), acc_w.mean().detach().item(),acc_s.mean().detach().item(),
                        loss.mean().detach().item()))

        # average loss # average acc

        final_acc = (sum(total_acc.values())/len(total_acc.keys())) / len(self.data_loader)
        final_loss = (sum(total_loss.values())/len(total_loss.keys())) / len(self.data_loader)
        # final_loss_w = (sum(total_loss_w.values()) / len(total_loss_w.keys())) / len(self.data_loader)
        # final_loss_s = (sum(total_loss_s.values()) / len(total_loss_s.keys())) / len(self.data_loader)
        self.logger.info('===> Training set: Average loss: {:.4f}\tAccuracy: {:.4f}'.format(
            final_loss, final_acc))
        return final_acc, final_loss, step

    def validation(self, data_loader, step):
        # with experiment.validate():
        self.logger.info("Starting Validation")
        self.model.eval()

        total_loss = {i: 0.0 for i in range(1, self.k_size + 1)}
        total_acc = {i: 0.0 for i in range(1, self.k_size + 1)}
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                if self.p.mode == 'word':
                    relation, mask, entity = data
                    relation = relation.long().to(self.device)
                    entity = entity.long().to(self.device)
                    loss, accuracy = self.model.forward(entity, relation, self.device)
                else:
                    data = data.long().to(self.device)
                    loss, accuracy = self.model.forward(data, self.device)
                acc = torch.mean(accuracy, 0)

                loss = torch.mean(loss, 0)
                for i, (a, l) in enumerate(zip(acc, loss)):
                    total_loss[i + 1] += l.detach().item()
                    total_acc[i + 1] += a.detach().item()

        # average loss # average acc
        final_acc = (sum(total_acc.values())/len(total_acc.keys()))/ len(data_loader)
        final_loss = (sum(total_loss.values())/len(total_loss.keys())) / len(data_loader)
        if step == 'validation':
            self.logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                final_loss, final_acc))
        else:
            self.logger.info('===> Test set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                final_loss, final_acc))


        return final_acc, final_loss


    def fit(self):
        self.best_acc, self.best_val, self.best_test, self.best_test_mrr, self.best_epoch = 0., {}, {}, 0., 0.
        save_path = os.path.join('./models', self.p.name)

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')
            test_results = self.validation('test', -1)
            pprint(test_results)
        run_name = "cpc" + time.strftime("-%Y-%m-%d_%H_%M_%S")

        for epoch in range(self.p.epochs):
            train_acc, train_loss, step = self.train(0, epoch, 100)
            if train_acc > self.best_acc:
                best_acc = max(train_acc, self.best_acc)
                if self.p.mode == 'joint':
                    dict_to_save = {'model_w': self.model_w.state_dict(),
                                    'model_s': self.model_s.state_dict()}

                else:
                    dict_to_save = self.model.state_dict()
                best_epoch = epoch
                self.snapshot('./logs', run_name, {
                    'epoch': epoch,
                    'step_train': step,
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                    'state_dict': dict_to_save,
                    'optimizer': self.optimizer.state_dict(),
                    'best_acc': best_acc,
                    'best_epoch': best_epoch
                })
        if args.mode == 'word':
            ent_embed, rel_embed = self.model_w.get_embed()
        elif args.mode == 'sentence':
            ent_embed, rel_embed = self.model_s.get_embed()
        elif args.mode == 'joint':
            ent_embed, rel_embed = self.model_w.get_embed()

        print(ent_embed.shape)  ##(40943,500) ([14541, 200])
        print(rel_embed.shape)  ##(23,500) ([475, 200])

        torch.save(rel_embed, self.p.data_dir+"rel_embedding.pt")
        torch.save(ent_embed, self.p.data_dir+"ent_embedding.pt")
           


if __name__ == "__main__":
    args = get_arguments()
    if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = Main(args)
    model.fit()




