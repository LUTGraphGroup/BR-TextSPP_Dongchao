from torch import nn as nn
from torch.nn import functional as F
import torch,time,os,random
import numpy as np
from collections import OrderedDict
from model import *







class TextSPP(nn.Module):
    def __init__(self, embedding, dropout=0.3, freeze=False, name='textEmbedding'):
        super(TextSPP, self).__init__()
        self.name = name
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding,dtype=torch.float32), freeze=freeze)
        self.dropout1 = nn.Dropout2d(p=dropout/2)
        self.dropout2 = nn.Dropout(p=dropout/2)
        self.p = dropout
        self.spp=TextSPP(size=128)

    def forward(self, x):
        if self.p>0:
            x = self.dropout2(self.dropout1(self.embedding(x.long())))
            x=self.spp(x)
        else:
            x = self.embedding(x.long())
            x =self.spp(x)
        return x

class BatchNorm1d(nn.Module):
    def __init__(self, inSize, name='batchNorm1d'):
        super(BatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(inSize)
        self.name = name
    def forward(self, x):
        return self.bn(x)
class BatchNorm1d(nn.Module):
    def __init__(self, inSize, name='batchNorm1d'):
        super(BatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(inSize)
        self.name = name
    def forward(self, x):
        return self.bn(x)
class TextSPP(nn.Module):
    def __init__(self, size=128, name='textSpp2'):
        super(TextSPP, self).__init__()
        self.name = name
        self.spp1 = nn.AdaptiveMaxPool1d(size)
        self.spp2 = nn.AdaptiveAvgPool1d(size)

    def forward(self, x):
        x1 = self.spp1(x).unsqueeze(dim=3)
        x2 = self.spp2(x).unsqueeze(dim=3)
        x3 = -self.spp1(-x).unsqueeze(dim=3)
        y=(torch.cat([x1, x2, x3], dim=3))
        y=y[:, :, :, 0]
        return y

class BiTextLSTM(nn.Module):
    def __init__(self, feaSize, hiddenSize, num_layers=1, dropout=0.0, bidirectional=True, name='textBiLSTM'):
        super(BiTextLSTM, self).__init__()
        self.name = name
        self.biLSTM = nn.LSTM(feaSize, hiddenSize, bidirectional=bidirectional, batch_first=True,
                              num_layers=num_layers, dropout=dropout)
    def forward(self, x, xlen=None):
        if xlen is not None:
            xlen, indices = torch.sort(xlen, descending=True)
            _, desortedIndices = torch.sort(indices, descending=False)
            x = x[indices]
            x = nn.utils.rnn.pack_padded_sequence(x, xlen, batch_first=True)
        output, hn = self.biLSTM(x)
        if xlen is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            output = output[desortedIndices]
        return output
    def orthogonalize_gate(self):
        nn.init.orthogonal_(self.biLSTM.weight_ih_l0)
        nn.init.orthogonal_(self.biLSTM.weight_hh_l0)
        nn.init.ones_(self.biLSTM.bias_ih_l0)
        nn.init.ones_(self.biLSTM.bias_hh_l0)

class FastText(nn.Module):
    def __init__(self, feaSize, name='fastText'):
        super(FastText, self).__init__()
        self.name = name
    def forward(self, x, xLen):
        x = torch.sum(x, dim=1) / xLen.float().view(-1,1)
        return x


class MLP(nn.Module):
    def __init__(self, inSize, outSize, hiddenList=[], dropout=0.0, bnEveryLayer=False, dpEveryLayer=False, outBn=False, outAct=False, outDp=False, name='MLP', actFunc=nn.LeakyReLU):
        super(MLP, self).__init__()
        self.name = name
        hiddens,bns = [],[]
        for i,os in enumerate(hiddenList):
            hiddens.append( nn.Sequential(
                nn.Linear(inSize, os),
            ) )
            bns.append(nn.BatchNorm1d(os))
            inSize = os
        bns.append(nn.BatchNorm1d(outSize))
        self.actFunc = actFunc()
        self.dropout = nn.Dropout(p=dropout)
        self.hiddens = nn.ModuleList(hiddens)
        self.bns = nn.ModuleList(bns)
        self.out = nn.Linear(inSize, outSize)
        self.bnEveryLayer = bnEveryLayer
        self.dpEveryLayer = dpEveryLayer
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp
    def forward(self, x):

        for h,bn in zip(self.hiddens,self.bns):
            x = h(x)
            if self.bnEveryLayer:
                if len(x.shape)==3:
                    x = bn(x.transpose(1,2)).transpose(1,2)
                else:
                    x = bn(x)
            x = self.actFunc(x)
            if self.dpEveryLayer:
                x = self.dropout(x)
        x = self.out(x)
        if self.outBn: x = self.bns[-1](x)
        if self.outAct: x = self.actFunc(x)
        if self.outDp: x = self.dropout(x)


class pseudolabelAttention(nn.Module):
    def __init__(self, inSize, classNum, labSize=1024, hdnDropout=0.1, attnList=[], labDescVec=None, name='DeepICDAttn'):
        super(pseudolabelAttention, self).__init__()
        hdns,attns,bns = [],[],[]
        for i,os in enumerate(attnList):
            attns.append(nn.Linear(inSize,os))
            if i==len(attnList)-1:
                hdns.append(nn.Linear(inSize, labSize))
                inSize = labSize
            else:
                hdns.append(nn.Linear(inSize,inSize))
            bns.append(nn.BatchNorm1d(inSize))
        self.hdns = nn.ModuleList(hdns)
        self.attns = nn.ModuleList(attns)
        self.bns = nn.ModuleList(bns)
        self.dropout = nn.Dropout(p=hdnDropout)
        self.labDescVec = nn.Parameter(torch.tensor(labDescVec, dtype=torch.float32)) if labDescVec is not None else None
        self.name = name
    def forward(self, X, labDescVec=None):
        if labDescVec is None:
            labDescVec = self.labDescVec
        for h,a,b in zip(self.hdns,self.attns,self.bns):
            alpha = F.softmax(a(X), dim=1)
            X = torch.matmul(alpha.transpose(1,2), X)
            X = h(X)
            X = b(X.transpose(1,2)).transpose(1,2)
            X = F.relu(X)
            X = self.dropout(X)
        alpha = F.softmax(torch.matmul(X, labDescVec.transpose(0,1)), dim=1)
        X = torch.matmul(alpha.transpose(1,2), X)

        return X

class LuongAttention(nn.Module):
    def __init__(self, method):
        super(LuongAttention, self).__init__()
        self.method = method
    def dot_score(self, hidden, encoderOutput):
        return torch.matmul(encoderOutput, hidden.transpose(-1,-2))
    def forward(self, hidden, encoderOutput):
        attentionScore = self.dot_score(hidden, encoderOutput).transpose(-1,-2)
        return F.softmax(attentionScore, dim=-1)


class LayerNormAndDropout_l2_regularization(nn.Module):
    def __init__(self, feaSize, dropout=0.1, name='LayerNormAndDropout_l2_regularization', l2_reg=1e-5):
        super(LayerNormAndDropout_l2_regularization, self).__init__()
        self.layerNorm = nn.LayerNorm(feaSize)
        self.dropout = nn.Dropout(p=dropout)
        self.l2_reg = l2_reg
        self.name = name

    def forward(self, x):
        x = self.layerNorm(x)
        x = self.dropout(x)
        return x

    def l2_regularization(self):
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        return self.l2_reg * l2_norm


        










