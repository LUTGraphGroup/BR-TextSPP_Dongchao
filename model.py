import torch,time,os,pickle
from torch import nn
import numpy as np
from layer import *
from metrics import *
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BaseModel:
    def train(self, dataClass, trainSize, batchSize, epoch,
              lr=0.001, weightDecay=0.0, stopRounds=10, threshold=0.2, earlyStop=10,
              savePath='model/BR-TextSPP', saveRounds=1, isHigherBetter=True, metrics="MiF", report=["ACC", "MiF"]):
        assert batchSize%trainSize==0
        metrictor = Metrictor(dataClass.classNum)
        self.stepCounter = 0
        self.stepUpdate = batchSize//trainSize
        optimizer = torch.optim.Adam(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        trainStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='train', device=self.device)
        itersPerEpoch = (dataClass.trainSampleNum+trainSize-1)//trainSize
        mtc,bestMtc,stopSteps = 0.0,0.0,0
        if dataClass.validSampleNum>0: validStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='valid', device=self.device)
        st = time.time()
        results_df = pd.DataFrame(columns=['Epoch', 'Train_Metric', 'Valid_Metric', 'Test_Metric'])
        for e in range(epoch):
            for i in range(itersPerEpoch):
                self.to_train_mode()
                X, Y = next(trainStream)
                loss = self._train_step(X, Y, optimizer)
                if stopRounds>0 and (e*itersPerEpoch+i+1)%stopRounds==0:
                    self.to_eval_mode()
                    print("After iters %d: [train] loss= %.3f;"%(e*itersPerEpoch+i+1,loss), end='')
                    if dataClass.validSampleNum>0:
                        X, Y = next(validStream)
                        loss = self.calculate_loss(X,Y)
                        print(' [valid] loss= %.3f;'%loss, end='')
                    restNum = ((itersPerEpoch-i-1)+(epoch-e-1)*itersPerEpoch)*trainSize
                    speed = (e*itersPerEpoch+i+1)*trainSize/(time.time()-st)
                    print(" speed: %.3lf items/s; remaining time: %.3lfs;"%(speed, restNum/speed))
            if dataClass.validSampleNum>0 and (e+1)%saveRounds==0:
                self.to_eval_mode()
                print('========== Epoch:%5d =========='%(e+1))
                print('[Total Train]', end='')
                Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
                metrictor.set_data(Y_pre, Y, threshold)
                metrictor(report)
                print('[Total Valid]', end='')
                Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
                metrictor.set_data(Y_pre, Y, threshold)
                res = metrictor(report)
                res[metrics]
                print('=================================')
                if (mtc>bestMtc and isHigherBetter) or (mtc<bestMtc and not isHigherBetter):
                    print('Bingo!!! Get a better Model with test %s: %.3f!!!'%(metrics,mtc))
                    bestMtc = mtc
                    self.save("%s.pkl"%savePath, e+1, bestMtc, dataClass)
                    stopSteps = 0
                else:
                    stopSteps += 1
                    if stopSteps>=earlyStop:
                        print('The test %s has not improved for more than %d steps in epoch %d, stop training.'%(metrics,earlyStop,e+1))
                        break
        self.load("%s.pkl"%savePath, dataClass=dataClass)
        os.rename("%s.pkl"%savePath, "%s_%s.pkl"%(savePath, ("%.3lf"%bestMtc)[2:]))

        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
        metrictor.set_data(Y_pre, Y, threshold)
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
        metrictor.set_data(Y_pre, Y, threshold)
        return res

    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()
    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'],stateDict['testIdList'] = dataClass.trainIdList,dataClass.validIdList,dataClass.testIdList
            stateDict['nword2id'],stateDict['tword2id'] = dataClass.nword2id,dataClass.tword2id
            stateDict['id2nword'],stateDict['id2tword'] = dataClass.id2nword,dataClass.id2tword
            stateDict['icd2id'],stateDict['id2icd'] = dataClass.id2icd,dataClass.icd2id
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            dataClass.trainIdList = parameters['trainIdList']
            dataClass.validIdList = parameters['validIdList']
            dataClass.testIdList = parameters['testIdList']

            dataClass.nword2id,dataClass.tword2id = parameters['nword2id'],parameters['tword2id']
            dataClass.id2nword,dataClass.id2tword = parameters['id2nword'],parameters['id2tword']
            dataClass.id2icd,dataClass.icd2id = parameters['icd2id'],parameters['id2icd']
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)['y_logit']
        return torch.sigmoid(Y_pre)
    def calculate_y(self, X, threshold=0.2):
        Y_pre = self.calculate_y_prob(X)
        isONE = Y_pre>threshold
        Y_pre[isONE],Y_pre[~isONE] = 1,0
        return Y_pre
    def calculate_loss(self, X, Y):
        out = self.calculate_y_logit(X)
        Y_logit = out['y_logit']
        addLoss = 0.0
        if 'loss' in out: addLoss += out['loss']
        return self.crition(Y_logit, Y) + addLoss
    def calculate_indicator_by_iterator(self, dataStream, classNum, report, threshold):
        metrictor = Metrictor(classNum)
        Y_prob_pre,Y = self.calculate_y_prob_by_iterator(dataStream)
        Metrictor.set_data(Y_prob_pre, Y, threshold)
        return metrictor(report)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy(),Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr
    def calculate_y_by_iterator(self, dataStream, threshold=0.2):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        isONE = Y_preArr>threshold
        Y_preArr[isONE],Y_preArr[~isONE] = 1,0
        return Y_preArr, YArr
    def to_train_mode(self):
        for module in self.moduleList:
            module.train()
    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()
    def _train_step(self, X, Y, optimizer):
        self.stepCounter += 1
        if self.stepCounter<self.stepUpdate:
            p = False
        else:
            self.stepCounter = 0
            p = True
        loss = self.calculate_loss(X, Y)/self.stepUpdate
        loss.backward()
        if p:
            nn.utils.clip_grad_norm_(self.moduleList.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
        return loss*self.stepUpdate

class MultiLabelCircleLoss(nn.Module):
    def __init__(self):
        super(MultiLabelCircleLoss, self).__init__()
    def forward(self, Y_logit, Y):
        loss,cnt = 0,0
        for yp,yt in zip(Y_logit,Y):
            neg = yp[yt==0]
            pos = yp[yt==1]
            loss += torch.log(1+torch.exp(neg).sum()) + torch.log(1+torch.exp(-pos).sum())
            cnt += 1
        return loss/cnt

class BR_TextSPP(BaseModel):
    def __init__(self, classNum, embedding, labDescVec,
                 rnnHiddenSize=64, attnList=[], embDropout=0.4, hdnDropout=0.4, fcDropout=0.5, device=torch.device('cuda:0'),
                 useCircleLoss=False):
        self.embedding = torch.from_numpy(embedding).to(device)
        self.embedding = TextSPP(embedding=embedding, dropout=embDropout,freeze=False).to(device)
        self.biLSTM = BiTextLSTM(feaSize=128, hiddenSize=128, num_layers=1, dropout=0.2, bidirectional=True,name='TextLSTM').to(device)
        self.LNandl2=LayerNormAndDropout_l2_regularization(feaSize=rnnHiddenSize*2,dropout=0.1,l2_reg=1e-6).to(device)
        self.icdAttn = pseudolabelAttention(inSize=rnnHiddenSize*2, classNum=classNum,labSize=labDescVec.shape[1], hdnDropout=hdnDropout,attnList=attnList,labDescVec=labDescVec).to(device)
        self.fcLinear = MLP(inSize=labDescVec.shape[1], outSize=1, hiddenList=[], dropout=fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.embedding,self.biLSTM, self.LNandl2,self.icdAttn, self.fcLinear])
        self.crition = nn.MultiLabelSoftMarginLoss() if not useCircleLoss else MultiLabelCircleLoss()
        self.labDescVec = torch.tensor(labDescVec, dtype=torch.float32).to(device)
        self.device = device
        self.hdnDropout = hdnDropout
        self.fcDropout = fcDropout
    def calculate_y_logit(self, input):
        x = input['noteArr']
        x = self.embedding(x)
        x = self.biLSTM(x)
        x = self.LNandDP(x)
        x = self.icdAttn(x)
        x = self.fcLinear(x).squeeze(dim=2)

        return {'y_logit':x}


