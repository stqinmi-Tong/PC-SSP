import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# CNN Encoder Model
class Conve_Encoder(nn.Module):###genc
    def __init__(self, ker_sz, k_w, k_h, hidden_dim, encoder_dim, sentence_len,window,bias):
        super(Conve_Encoder, self).__init__()
        self.ker_sz = ker_sz
        self.k_w = k_w
        self.k_h = k_h
        self.hidden_dim = hidden_dim
        self.sentence_len = sentence_len
        self.windows = window
        self.input_drop = torch.nn.Dropout(0.2)
        self.feature_drop = torch.nn.Dropout2d(0.5)
        self.hidden_drop = torch.nn.Dropout(0.5)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_dim)
        self.bias = bias
        self.num_filt = 32


        # sentence encoder genc (a 2D-convolution + 1D-convolution + ReLU + mean-pooling)
        self.conv1 = torch.nn.Conv2d(1, out_channels=self.num_filt, kernel_size=(self.ker_sz, self.ker_sz),
                                     stride=1, padding=0, bias=self.bias)

        flat_sz_w = int(self.sentence_len * self.k_w)-self.ker_sz+1
        flat_sz_h = self.k_h-self.ker_sz+1
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt  ###flat_sz_h * flat_sz_w ：卷积之后的每个通道的卷积特征图大小，
        self.conv2 = nn.Conv1d(in_channels=self.flat_sz, out_channels=encoder_dim, kernel_size=1)
        self.fc = torch.nn.Linear(self.flat_sz, encoder_dim)


    def concat(self, x): # 之前：x:(B,D) sentence=2时：x.shape:(B,4,2,D)；sentence=3时：x.shape:(B,1,3,D)
        B = x.shape[0]
        x = x.view(-1, self.sentence_len, self.hidden_dim) #self.hidden_dim=embed_dim  #torch.Size([256, 3, 200])
        stack_inp = torch.transpose(x, 2, 1).reshape((B * self.windows, 1, self.sentence_len * self.k_w, self.k_h))
        return stack_inp

    def forward(self, x):  # x.shape: (B,S,D), 这里S=2或者3，由于S长度不一，需要设置两种编码器.这里处理S=2的情况
        B = x.shape[0]

        ###
        # sub_emb = self.ent_embed(sub)
        # rel_emb = self.rel_embed(rel)
        # comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
        # # change
        # chequer_perm = comb_emb[:, :]
        # # chequer_perm = comb_emb[self.chequer_perm[0], :][:, self.chequer_perm[1]]
        # ## end
        # stack_inp = chequer_perm.reshape((-1, self.p.perm, 2 * self.p.k_w, self.p.k_h))
        # # change
        # stack_inp = stack_inp[:, :, self.chequer_perm, :]
        # stack_inp = stack_inp.reshape((-1, self.p.perm, 2 * self.p.k_w, self.p.k_h))  # oops torch.Size([64, 4, 20, 20])
        # # end
        # stack_inp = self.bn0(stack_inp)
        # x = self.inp_drop(stack_inp)
        # x = self.circular_padding_chw(x, self.p.ker_sz // 2)  # torch.Size([64, 4, 30, 30])
        # x = F.conv2d(x, self.conv_filt.repeat(self.p.perm, 1, 1, 1), padding=self.padding,
        #              groups=self.p.perm)  # conv2d torch.Size([64, 384, 20, 20]
        ###
        stk_inp = self.concat(x)
        x = self.bn0(stk_inp)
        x = self.input_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        # x = x.view(1,B * self.windows, -1).transpose(1, 2)  # torch.Size([1, 10368, 1024])
        # x = self.conv2(x)
        # x = x.squeeze(0).transpose(0,1) #torch.Size([1024, 200])
        # x = F.relu(x)  #x.shape:(B*4,D)

        return x  # 压缩矩阵的维数

class CPC_sentence(nn.Module):####输入流数据是每一个批次的每个路径的关系，实体，mask和负采样实体
    def __init__(self, ent_vocab, rel_vocab, embedding_size,enc_dim, ar_dim, window,
                 k_size, path_length,bias):
        super(CPC_sentence, self).__init__()
        # load parameters
        self.window = window
        self.path_length = path_length # 11
        self.ent_vocab = ent_vocab
        self.rel_vocab = rel_vocab
        self.embedding_size = embedding_size
        self.enc_dim = enc_dim
        self.ar_dimension = ar_dim
        self.k_size = k_size   ###指的窗口大小,我们选取的路径长度为5，有两跳，想先选k=3试试，从第一个三元组的尾实体开始测试
        self.kernel_sz = 3
        self.k_w = 10
        self.k_h = 20
        gamma = 24.0
        self.epsilon = 2.0
        # define embedding layer
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / (self.embedding_size / 2)]),
            requires_grad=False
        )
        self.ent_embedding = nn.Parameter(torch.zeros(self.ent_vocab, self.embedding_size))
        nn.init.uniform_(
            tensor=self.ent_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.rel_embedding = nn.Parameter(torch.zeros(self.rel_vocab, self.embedding_size))
        nn.init.uniform_(
            tensor=self.rel_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # define type of encoder: genc，输入为emb_dim维度的向量，输出为enc_dim维度的向量
        self.encoder_1 = Conve_Encoder(
            ker_sz=self.kernel_sz,
            k_w=self.k_w,
            k_h=self.k_h,
            hidden_dim=self.embedding_size,
            encoder_dim=self.enc_dim,
            sentence_len=2,  ###最大句子长度
            window=4,
            bias = bias
        )
        self.encoder_2 = Conve_Encoder(
            ker_sz=self.kernel_sz,
            k_w=self.k_w,
            k_h=self.k_h,
            hidden_dim=self.embedding_size,
            encoder_dim=self.enc_dim,
            sentence_len=3,  ###最大句子长度
            window=1,
            bias=bias
        )
        # define autoregressive model: gar,输入维度为enc_dimension的向量得到维度为ar_dimension的结果向量
        self.gru = nn.GRU(
            self.enc_dim,
            self.ar_dimension,
            batch_first=True)
        # define predictive layer: W2
        self.Wk = nn.ModuleList([nn.Linear(self.ar_dimension,self.enc_dim, bias=False)
                                 for i in range(self.k_size)]) # shape：（self.enc_dimension, self.ar_dimension）

        self.register_parameter('bias', Parameter(torch.zeros(self.ent_vocab)))
    def init_hidden(self, batch_size, device=None):
        if device:
            return torch.zeros(1, batch_size, self.ar_dimension).to(device)
        else:
            return torch.zeros(1, batch_size, self.ar_dimension)

    # B: Batch, W: Window, S: Sentence, D: Dimension
    def forward(self, x, device):  # x.shape: (B,P) ####输入流数据是每一个批次的每个路径P,P=11

        batch = x.shape[0]
        hidden = self.init_hidden(batch, device)  # hidden.shape:(B,D) D(default:2400)
        x = self.get_sentence_embedding(batch, x, device) # x——>(B,P,D)  P=11

        # get sentence embeddings
        z = self.get_cpc_embedding(x)  # z.shape: (B,W,D)

        # separate forward and target samples  W1: forward window, W2: target window
        target = z[:, -self.k_size:, :].transpose(0, 1)  # target.shape: (W2,B,D)
        forward_sequence = z[:, :-self.k_size, :]  # forward_sequence.shape: (B,W1,D)，，W1+W2=len，前W1是输入，后W2是要做验证的输出，len=W

        # feed ag model
        self.gru.flatten_parameters()
        output, hidden = self.gru(forward_sequence, hidden)  # output.shape: (B,W1,D) 也即ct-3，ct-2，ct-1，ct
        context = output[:, -1, :].view(batch, self.ar_dimension)  # context.shape: (B,D) (take last hidden state) 也即把ct取出来，这里D=self.ar_dimension，torch.Size([64, 32])
        pred = torch.empty((self.k_size, batch, self.enc_dim), dtype=torch.float)#, device=device)  # pred (empty container).shape: (W2,B,D)

        # loop of prediction
        for i in range(self.k_size):
            linear = self.Wk[i]  #（self.enc_dim, self.ar_dimension） Linear(in_features=32, out_features=64, bias=True)  context: torch.Size([64, 32])
            pred[i] = linear(context)  # Wk*context.shape: (B,D) pred[i].shape:(B,self.enc_dim),pred.shape: (k_size,B,self.enc_dim)也即（W2，B，D）


        # loss, accuracy = self.info_nce(pred.detach().cpu().numpy(), target)  # shape: (W2,B,D)
        #把detach去掉试试
        loss, accuracy = self.info_nce(pred, target)  # shape: (W2,B,D)
        return loss, accuracy

    def get_sentence_embedding(self, batch, x, device):  # batch * ent_num,batch * rel_num
        ents,rels = [],[]
        for path in x:
            for idx, word in enumerate(path):
                if idx % 2 == 0:
                    ents.append(word)
                else:
                    rels.append(word)
        ents, rels = torch.tensor(ents).to(device),torch.tensor(rels).to(device)

        ent_embed = torch.index_select(
            self.ent_embedding,
            dim=0,
            index=ents
        ).reshape(batch, 6, -1)
        rel_embed = torch.index_select(
            self.rel_embedding,
            dim=0,
            index=rels
        ).reshape(batch, 5, -1)  # .view(-1, batch)

        path_embed = []
        paths_embed = []

        for i, x in enumerate(ent_embed):
            for j, y in enumerate(x):
                path_embed.append(ent_embed[i][j])
                if len(path_embed) != self.path_length:
                    path_embed.append(rel_embed[i][j])
                else:
                    paths_embed.append(path_embed)
                    path_embed = []

        inputs_embedding = []
        for i in paths_embed:
            input_embedding = torch.cat([x for x in i], 0).reshape(5, -1).squeeze(0)
            inputs_embedding.append(input_embedding)
        input_embedding = torch.cat([x for x in inputs_embedding], dim=0).reshape((-1, 11, self.embedding_size))  # (B,11,D) batch * path_length,emb_dimension
        # print(input_embedding.shape) #torch.Size([64, 5, 64])  torch.Size([128, 5, 200]) torch.Size([256, 11, 200])
        return input_embedding


    def get_cpc_embedding(self,x): # x.shape: (B,P,D) ####输入流数据是每一个批次的每个路径P,P=11
        ####将每个p处理成[2,2,2,2,3]
        seg_path = []
        encod_out = []

        y = torch.split(x, [2,2,2,2,3], dim=1)
        for i in y:
            seg_path.append(i) # i分别是(B,2,D)(B,2,D)(B,2,D)(B,2,D)(B,3,D)
            ##### i.shape:
            # torch.Size([256, 2, 200])
            # torch.Size([256, 2, 200])
            # torch.Size([256, 2, 200])
            # torch.Size([256, 2, 200])
            # torch.Size([256, 3, 200])

        conv_input1 = torch.cat([x for x in seg_path[:-1]], dim=1).view(-1, 4, 2, self.embedding_size) # z.shape: (B,W-1,2,D) torch.Size([256, 4, 2, 200])
        conv_input2 = torch.cat([x for x in seg_path[-1]], dim=1).reshape(-1, 1, 3, self.embedding_size)  # z.shape: (B,1,3,D)
        out1 = self.encoder_1(conv_input1).view(-1,int(self.window-1),self.enc_dim)
        out2 = self.encoder_2(conv_input2).view(-1,1,self.enc_dim)
        encod_out.append(out1) # shape:(B,4,D)
        encod_out.append(out2)  # shape:(B,1,D)

        z = torch.cat([x for x in encod_out], dim=1).reshape(-1, self.window, self.enc_dim)  # z.shape: (B,W,D) torch.Size([256, 5, 200])
        return z


    def info_nce(self, prediction, target):
        k_size, batch_size, hidden_size = target.shape #(W2,B,D)
        label = torch.arange(0, batch_size * k_size, dtype=torch.long, device=target.device)

        logits = torch.matmul(
            torch.Tensor(prediction).reshape([-1, hidden_size]).to(target.device),
            target.reshape([-1, hidden_size]).transpose(-1, -2)
        )
        loss = nn.functional.cross_entropy(logits, label, reduction='none')
        accuracy = torch.eq(
            torch.argmax(F.softmax(logits, dim=1), dim=1),
            label)
        # process for split loss and accuracy into k pieces (useful for logging)
        nce, acc = [], []
        for i in range(k_size):
            start = i * batch_size
            end = i * batch_size + batch_size
            nce.append(torch.sum(loss[start:end]) / batch_size)
            acc.append(torch.sum(accuracy[start:end], dtype=torch.float) / batch_size)

        return torch.stack(nce).unsqueeze(0), torch.stack(acc).unsqueeze(0)

    def get_embedding(self, x, y):
        ent = torch.index_select(
            self.ent_embedding,
            dim=0,
            index=x
        ).unsqueeze(1)
        rel = torch.index_select(
            self.rel_embedding,
            dim=0,
            index=y
        ).unsqueeze(1)
        return ent, rel

    def get_embed(self):
        return self.ent_embedding,self.rel_embedding


class TxtClassifier(nn.Module):
    ''' linear classifier '''

    def __init__(self, d_input,n_class):
        super(TxtClassifier, self).__init__()
        self.classifier = nn.Linear(d_input, n_class)

    def forward(self, x):
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)

# if __name__ =='__main__':
#     print('=====CPC testing=====')
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'CPU')
#     ents = torch.ones([128, 3]).long().to(device)
#     rels = torch.ones([128, 2]).long().to(device)
#
#     # gru_ouput = torch.zeros([64, 300, 2]).to(device)
#     # ent_negs = torch.zeros([64, 2, 20]).long().to(device)
#     ent_negs = None
#     # (self, ent_vocab, rel_vocab, embedding_size,ar_dim, window, k_size,path_length)
#     model = CPC_sentence(60, 6, 100, 100,100, 5, 3, 5).to(device)
#     loss, accuracy = model(ents, rels,device)  # forward(self, ents, rels)
#     print('loss', loss) #loss tensor([[5.5461, 5.2585, 6.1626]], device='cuda:0',       grad_fn=<UnsqueezeBackward0>)
#     print('accuracy', accuracy) #accuracy tensor([[0.0078, 0.0078, 0.0000]], device='cuda:0')
#
#     # txt_model = TxtClassifier(config).to(device)

