import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class P2E(nn.Module):  
    def __init__(self, ent_vocab, rel_vocab, embedding_size, ar_dim, window, k_size, path_length):
        super(P2E, self).__init__()
        # load parameters
        self.window = window
        self.path_length = path_length
        self.ent_vocab = ent_vocab
        self.rel_vocab = rel_vocab
        self.embedding_size = embedding_size

        self.ar_dimension = ar_dim
        self.k_size = k_size  
        self.epsilon = 2.0
        gamma = 24.0
       
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
        if self.window != self.path_length:
            self.window = self.path_length

        self.gru = nn.GRU(
            # self.enc_dimension,
            self.embedding_size,
            self.ar_dimension,
            batch_first=True)
      
        self.Wk = nn.ModuleList([nn.Linear(self.ar_dimension, self.embedding_size, bias=False)
                                 for i in range(self.k_size)])  # shape：（self.enc_dimension, self.ar_dimension）

    def init_hidden(self, batch_size, device=None):
        if device:
            return torch.zeros(1, batch_size, self.ar_dimension).to(device)
        else:
            return torch.zeros(1, batch_size, self.ar_dimension)

    # B: Batch, W: Window, S: Sentence, D: Dimension
    def forward(self, ents, rels, device):  # x.shape: (B,W,S) 
        batch = ents.shape[0]
        ent_num = self.path_length // 2 + 1
        rel_num = self.path_length // 2
        hidden = self.init_hidden(batch, device)  # hidden.shape:(B,D) D(default:2400)

        # get sentence embeddings
        z = self.get_input_embedding(batch, ents.view(batch * ent_num),
                                     rels.view(batch * rel_num))  # print(z.shape) #torch.Size([64, 5, 64])

        z = z.view(batch, self.window, self.embedding_size)  # z.shape:(B,W,D)

        # separate forward and target samples  W1: forward window, W2: target window
        target = z[:, -self.k_size:, :].transpose(0, 1)  # target.shape: (W2,B,D)
        forward_sequence = z[:, :-self.k_size,:]  # forward_sequence.shape: (B,W1,D)
        # feed ag model
        self.gru.flatten_parameters()
        output, hidden = self.gru(forward_sequence, hidden)  # output.shape: (B,W1,D) 
        context = output[:, -1, :].view(batch,self.ar_dimension)  # context.shape: (B,D) (take last hidden state) 
        pred = torch.empty((self.k_size, batch, self.embedding_size), dtype=torch.float, device=device)  # pred (empty container).shape: (W2,B,D)

        # loop of prediction
        for i in range(self.k_size):
            linear = self.Wk[i]  # （self.embedding_size, self.ar_dimension） Linear(in_features=32, out_features=64, bias=True)  context: torch.Size([64, 32])
            pred[i] = linear(context)  # Wk*context.shape: (B,D) pred[i].shape:(B,self.embedding_size),pred.shape: (k_size,B,self.embedding_size):（W2，B，D）

        loss, accuracy = self.info_nce(pred, target)  # shape: (W2,B,D)

        return loss, accuracy

    def get_input_embedding(self, batch, ents, rels):  # batch * ent_num,batch * rel_num
        # print(ents)
        ent_embed = torch.index_select(
            self.ent_embedding,
            dim=0,
            index=ents.view(-1)
        ).reshape(batch, 3, -1)
        rel_embed = torch.index_select(
            self.rel_embedding,
            dim=0,
            index=rels.view(-1)
        ).reshape(batch, 2, -1)  # .view(-1, batch)
        # ent_embed = self.ent_embedding(ents).reshape(batch,3,-1)
        # rel_embed = self.rel_embedding(rels).reshape(batch,2,-1)
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
        input_embedding = torch.cat([x for x in inputs_embedding], dim=0).reshape(-1, 5, self.embedding_size)  # .unsqueeze(2).cuda(self.gpu) # batch * path_length,emb_dimension
        # print(input_embedding.shape) #torch.Size([64, 5, 64])  torch.Size([128, 5, 200])

        return input_embedding

    def info_nce(self, prediction, target):
        k_size, batch_size, hidden_size = target.shape  # (W2,B,D)
        label = torch.arange(0, batch_size * k_size, dtype=torch.long, device=target.device)

        logits = torch.matmul(
            prediction.reshape([-1, hidden_size]),
            target.reshape([-1, hidden_size]).transpose(-1, -2)
        )

        loss = nn.functional.cross_entropy(logits, label, reduction='none')   
        accuracy = torch.eq(torch.argmax(F.softmax(logits, dim=1), dim=1), label)

        # process for split loss and accuracy into k pieces (useful for logging)
        nce, acc = [], []
        for i in range(k_size):
            start = i * batch_size
            end = i * batch_size + batch_size
            nce.append(torch.sum(loss[start:end]) / batch_size)
            acc.append(torch.sum(accuracy[start:end], dtype=torch.float) / batch_size)

        return torch.stack(nce).unsqueeze(0), torch.stack(acc).unsqueeze(0)

    def get_embedding(self, x, y):

        # x = self.ent_embedding(x)  # out.shape: (B*W,S,D)
        # y = self.rel_embedding(y)  # z.shape: (B*W,D)
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

        return self.ent_embedding, self.rel_embedding  # self.ent_embedding.weight.data.cpu().numpy(),self.rel_embedding.weight.data.cpu().numpy()


class TxtClassifier(nn.Module):
    ''' linear classifier '''

    def __init__(self, d_input, n_class):
        super(TxtClassifier, self).__init__()
        self.classifier = nn.Linear(d_input, n_class)

    def forward(self, x):
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)





