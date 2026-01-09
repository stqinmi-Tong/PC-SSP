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
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt  
        self.conv2 = nn.Conv1d(in_channels=self.flat_sz, out_channels=encoder_dim, kernel_size=1)
        self.fc = torch.nn.Linear(self.flat_sz, encoder_dim)


    def concat(self, x): 
        B = x.shape[0]
        x = x.view(-1, self.sentence_len, self.hidden_dim) #self.hidden_dim=embed_dim  #torch.Size([256, 3, 200])
        stack_inp = torch.transpose(x, 2, 1).reshape((B * self.windows, 1, self.sentence_len * self.k_w, self.k_h))
        return stack_inp

    def forward(self, x):  # x.shape: (B,S,D)
        B = x.shape[0]

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

        return x  

class P2P(nn.Module):
    def __init__(self, ent_vocab, rel_vocab, embedding_size,enc_dim, ar_dim, window,
                 k_size, path_length,bias):
        super(P2P, self).__init__()
        # load parameters
        self.window = window
        self.path_length = path_length # 11
        self.ent_vocab = ent_vocab
        self.rel_vocab = rel_vocab
        self.embedding_size = embedding_size
        self.enc_dim = enc_dim
        self.ar_dimension = ar_dim
        self.k_size = k_size   
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

        self.encoder_1 = Conve_Encoder(
            ker_sz=self.kernel_sz,
            k_w=self.k_w,
            k_h=self.k_h,
            hidden_dim=self.embedding_size,
            encoder_dim=self.enc_dim,
            sentence_len=2,  
            window=4,
            bias = bias
        )
        self.encoder_2 = Conve_Encoder(
            ker_sz=self.kernel_sz,
            k_w=self.k_w,
            k_h=self.k_h,
            hidden_dim=self.embedding_size,
            encoder_dim=self.enc_dim,
            sentence_len=3,  
            window=1,
            bias=bias
        )
        # define autoregressive model: gar
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
    def forward(self, x, device):  # x.shape: (B,P) 

        batch = x.shape[0]
        hidden = self.init_hidden(batch, device)  # hidden.shape:(B,D) D(default:2400)
        x = self.get_sentence_embedding(batch, x, device) # x——>(B,P,D)  P=11

        # get sentence embeddings
        z = self.get_cpc_embedding(x)  # z.shape: (B,W,D)

        # separate forward and target samples  W1: forward window, W2: target window
        target = z[:, -self.k_size:, :].transpose(0, 1)  # target.shape: (W2,B,D)
        forward_sequence = z[:, :-self.k_size, :]  # forward_sequence.shape: (B,W1,D)，W1+W2=len

        # feed ag model
        self.gru.flatten_parameters()
        output, hidden = self.gru(forward_sequence, hidden)  # output.shape: (B,W1,D) 
        context = output[:, -1, :].view(batch, self.ar_dimension)  # context.shape: (B,D) (take last hidden state) 
        pred = torch.empty((self.k_size, batch, self.enc_dim), dtype=torch.float)#, device=device)  # pred (empty container).shape: (W2,B,D)

        # loop of prediction
        for i in range(self.k_size):
            linear = self.Wk[i]  #（self.enc_dim, self.ar_dimension） Linear(in_features=32, out_features=64, bias=True)  context: torch.Size([64, 32])
            pred[i] = linear(context)  # Wk*context.shape: (B,D) pred[i].shape:(B,self.enc_dim),pred.shape: (k_size,B,self.enc_dim) =（W2，B，D）
       
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


    def get_cpc_embedding(self,x): # x.shape: (B,P,D) 
       
        seg_path = []
        encod_out = []

        y = torch.split(x, [2,2,2,2,3], dim=1)
        for i in y:
            seg_path.append(i) 
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





