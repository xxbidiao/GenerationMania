### model_ff_cols.py: Model trained for columns
# Nearly the same, except that the final output layer is column selection.
# This model assumes everything is a playable.
# Requires both this model and the basic feedforward model to work.

from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import sys, select
import matplotlib
import sklearn.metrics
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from .Config import config
from contextlib import suppress
#Constants. will make them parameters later
BATCH_SIZE = 128

GRAD_CLIP = 0.25 # prevents the loss to be like 1e+30

use_history_in_training = True
use_diff_in_training = True
use_audio_features_in_training = True

isCuda = True
if isCuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


history_length = 2

data_dim = 302
EPOCHS = config.training_epochs
model_state_name = config.training_state_feedforward_columns
data_location = config.training_file

do_columns = True

if do_columns:
    out_layer_size = 8+1

else:
    out_layer_size = 1+1

def tr_orig(arr):
    return arr[:,2:(2+2)]

def tr(arr):
    if not do_columns:
        return tr_orig(arr)
    else:
        result = []
        for item in arr:
            col = int(np.rint((item[31]-np.rint(item[31]))*100))
            #print("ORIG:%f,col:%d"%(item[31],col))
            if col == 30: #scratch
                result.append([1,0,0,0,0,0,0,0,0])
            elif 0 < col < 8:
                #print(col)
                one_hot_arr = [0]*9
                one_hot_arr[col] = 1

                result.append(one_hot_arr)
                # if col not in range(1, 8):
                #     print("OOB column:%d"%col)
            else:
                result.append([0,0,0,0,0,0,0,0,1])
        return np.array(result)
# remove labels
def anti_tr(arr):
    # removing history + audio classes
    # result = arr[:,0:2]
    # return result

    if use_history_in_training:
        # full, only removing labels themselves

        arr2 = arr[:]
        arr2[:, 31] = 0

        return np.delete(arr2, [2, 3], axis=1)
    else:

        # removing history
        result = arr[:]

        if not use_diff_in_training:
            result[:,0] = 0

        result[:,31] = 0
        for idx in range(32,data_dim):
            result[:,idx] = 0
        #print(result)
        return np.delete(result,[2,3],axis=1)

def real_len(arr):
    try:
        return arr.index(0)
    except ValueError:
        return len(arr)

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        #Original data have data size 304, but 2 are removed in the input making process.
        self.fc1 = nn.Linear(data_dim, 64)
        #self.fc1.weight.data.normal_(0,0.02)
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(64,32)
        self.rl2 = nn.ReLU()
        self.fc3 = nn.Linear(32,16)
        self.rl3 = nn.ReLU()
        self.fc4 = nn.Linear(16, out_layer_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


    def forward(self, x):
        out = self.fc1(x)
        out = self.rl1(out)
        out = self.fc2(out)
        out = self.rl2(out)
        out = self.fc3(out)
        out = self.rl3(out)
        out = self.fc4(out)
        #out = self.softmax(out)
        out = self.relu(out)
        # out = self.sigmoid(out)
        return out


class weighted_mse_loss(nn.MSELoss):
    def setWeight(self,weight):
        self.weight = weight.to(device)

    def forward(self,input, target):
        # print(input.size())
        # print(target.size())
        # print(self.weight.size())
        return torch.sum(self.weight * (input - target) ** 2)


def train_model():
    type_fv = torch.DoubleTensor
    type_lv = torch.LongTensor

    print("Looking for latest checkpoint...")
    state = None
    try:
        state = torch.load(model_state_name)
        print("Resuming training from saved model...")
    except:
        print("Let's start fresh.")



    # load data and make training set
    print("Loading data from disk...")
    data_orig = torch.load(data_location)
    # print(data.shape)
    # print(data[100])
    # print(data[200])

    data = []
    # remove all non-playables from the dataset
    for item in data_orig:
        if item[2] == 1:
            data.append(item)
    data = np.array(data)
    print("After: data len = %d"%len(data))


    # create unrolling dataset
    #data  = data[:8,:128]
    #data = data[:10000]

    # print("Size = "+str(data.shape))
    length = data.shape[0]
    verif_size = int(length * 1.0 / 10)
    eval_size = int(length * 1.0 / 10)
    train_size = length - eval_size - verif_size

    #print(data[55])

    #print('target:'+str(torch.from_numpy(tr(data[train_size:, 1:, :]))))
    print("Creating batches for training sets...")

    inputs = []
    targets = []

    # The last few data are cut. Should be fine.
    for i in tqdm(range(int(train_size / BATCH_SIZE))):
        data_thisbatch = data[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE]
        target = Variable(torch.from_numpy(tr(data_thisbatch)).type(type_fv), requires_grad=False)
        input = Variable(torch.from_numpy(anti_tr(data_thisbatch)).type(type_fv), requires_grad=False)
        inputs.append(input)
        targets.append(target)
    print("Done for training set, now working on verification set...")
    verif_size_end = length - eval_size
    verif_size_start = length - eval_size - verif_size
    test_input = Variable(torch.from_numpy(anti_tr(data[verif_size_start:verif_size_end])).type(type_fv), requires_grad=False).to(device)
    # print(test_input)
    # exit()
    test_target = Variable(torch.from_numpy(tr(data[verif_size_start:verif_size_end])).type(type_fv), requires_grad=False).to(device)
    #print("Test input size is:%s"%str(test_input.size()))
    # print(verif_size_start)
    # print(verif_size_end)
    print("Done for verification set.")
    # build the model
    seq = Sequence().to(device)
    seq.double()
    if state is not None:
        seq.load_state_dict(state)

    # def customized_loss(x,y):
    #     for xx,yy in zip(x,y):
    #         if xx =


    #weighted MSE for two classes
    criterion = weighted_mse_loss()
    if not do_columns:
        criterion.setWeight(Variable(torch.DoubleTensor([1,0.2])))
    else:
        criterion.setWeight(Variable(torch.DoubleTensor([1,1,1,1,1,1,1,1,0.2])))
    #criterion = nn.BCELoss()
    #criterion = nn.MSELoss()
    #criterion = nn.BCELoss(weight=torch.DoubleTensor([[1,0.2]]*BATCH_SIZE).to(device))
    #optimizer = optim.SGD(seq.parameters(),lr=0.0001)
    optimizer = optim.Adam(seq.parameters(),weight_decay=0.0001)
    #begin to train
    stable_count = 0
    old_loss = 100000000.0
    for epoch_count in range(EPOCHS):
        print('EPOCH: ', epoch_count)
        for idx in tqdm(range(len(inputs))):
            input = inputs[idx].to(device)
            target = targets[idx].to(device)

            def closure():
                optimizer.zero_grad()
                #print(input)
                out = seq(input)
                #print(out.size())
                loss = criterion(out, target)
                loss.backward()
                #torch.nn.utils.clip_grad_norm(seq.parameters(), GRAD_CLIP)
                #print(loss)
                return loss
            optimizer.step(closure)
        pred = seq(test_input)
        #print(pred.size())
        #criterion2= nn.BCELoss(weight=torch.DoubleTensor([[1, 0.25]] * pred.size()[0]).to(device))
        loss = criterion(pred, test_target)
        print('Verification Loss:', loss.cpu().data.numpy())

        # _,predicted = torch.max(pred,1)
        # total = len(predicted)
        # _,true_label = torch.max(test_target,1)
        # tp = 0
        # fp = 0
        # tn = 0
        # fn = 0
        # for i in range(total):
        #     if int(predicted[i]) == 0:
        #         if int(true_label[i]) == 0:
        #             tp += 1
        #         else:
        #             fp += 1
        #     if int(predicted[i]) == 1:
        #         if int(true_label[i]) == 1:
        #             tn += 1
        #         else:
        #             fn += 1
        # print("tp=%d tn=%d fp=%d fn=%d"%(tp,tn,fp,fn))
        # try:
        #     precision = 1.0*tp/(tp+fp)
        #     recall = 1.0*tp/(tp+fn)
        #     f1_score = 2/(1/precision+1/recall)
        #     print("F1 score = %f"%f1_score)
        #     print("AUC-ROC score = %f"%sklearn.metrics.roc_auc_score(list(true_label.data.numpy()),list(predicted.data.numpy())))
        # except:
        #     print("Exception on calculating F-1 score.")
        print("Saving state to disk...")
        state_dict = seq.state_dict()
        torch.save(state_dict,open(model_state_name, 'wb'), pickle_protocol=4)

eval_seq = None

def eval_model(i,override_model_state = None):
    global eval_seq
    i = Variable(torch.from_numpy(i.reshape(-1,data_dim))).to(device)
    # build the model
    if eval_seq is None:
        print("Loading model from disk...", end=".")
        #try:
        eval_seq = Sequence().to(device)
        eval_seq.double()
        if override_model_state is None:
            state = torch.load(model_state_name)
        else:
            state = torch.load(override_model_state)
        eval_seq.load_state_dict(state)
        print("Success.")
        # except:
        #     print("Failed.")
        #     exit()
    output = eval_seq(i)
    #print("input:%s"%i)
    #print("output:%s"%output)
    return output

# if __name__ == '__main__':
#     train_model()