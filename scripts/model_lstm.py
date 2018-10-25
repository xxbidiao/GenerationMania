# model_lstm.py
# This is the LSTM model, and its evaluation code.
# Part of the model is from the LSTM auto encoder/decoder.

from .LSTMAE import EncoderRNN
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from .data_utils import *
from copy import copy
from tqdm import tqdm
import statistics
from random import random
from .Config import config

# Switch this on/off to use GPU training, however LSTM model doesn't support GPU yet.
isCuda = False
if isCuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


use_history_in_training = False
model_state_name = config.training_state_LSTM
data_file_name = config.training_file_per_chart
epochs = config.lstm_epochs

# If True, very long sequences (charts) will be skipped.
max_sequence_length = 5120+256
skip_long_seqs = False

# Not implemented.
spacial_random_sampling = True
random_sample_size = 256

# Weighted Mean Square Error.
class weighted_mse_loss(nn.MSELoss):
    def setWeight(self,weight):
        self.weight = weight

    def forward(self,input, target):
        return torch.sum(self.weight * (input - target) ** 2)

def avg(arr):
    try:
        return sum(arr) / len(arr)
    except:
        print("Something wrong happened in avg()")
        return -1


def std(arr):
    try:
        return statistics.stdev(arr)
    except:
        print("Something wrong happened in std()")
        return -1

# Get labels from `arr`.
def tr(arr):
    arr2 = np.array(arr)
    return arr2[:,2:(2+2)]

# Get part of the sequence randomly.
def get_random_subsequence(arr,rows=256,row_dim=1,method=0):
    if method == 0:
        lower_bound_max = arr.shape[row_dim] - rows
        if lower_bound_max <= 0:
            return arr
        else:
            upper_bound = random.randint(0,lower_bound_max) + rows
        result = arr[(slice(None),) * row_dim + (range(upper_bound-rows,upper_bound-1),)]
        return result
    else:
        print("Unknown random sequence creation method specified.")
        exit()

# remove labels, only keep features.
def anti_tr(arr):
    # removing history + audio classes
    # result = arr[:,0:2]
    # return result

    if use_history_in_training:
        # full, only removing labels themselves

        arr2 = np.array(copy(arr))
        arr2[:, 31] = 0
        return np.delete(arr2, [2, 3], axis=1)
    else:

        # removing history
        result = np.array(copy(arr))
        #print(result.shape)
        mask = np.ones(result.shape[1], dtype=bool)
        result[:,31] = 0
        for idx in range(32,len(mask)):
            #result[:,idx] = 0
            mask[idx] = False
        #print(result)
        result2 =  np.delete(result[:,mask],[2,3],axis=1)
        #print(result2.shape)
        #exit()
        return result2

# Train the LSTM model.
def train():
    print("Looking for latest checkpoint...")
    state = None
    try:
        state = torch.load(model_state_name)
        print("Read saved model.")
    except:
        print("Let's start fresh.")
    dtype = torch.float #if torch.cuda.is_available() else torch.FloatStorage
    if use_history_in_training:
        feature_vector_size = 302
    else:
        feature_vector_size = 30
    label_vector_size = 2
    autoencoder_vector_size = 4
    encoder_vector_size = feature_vector_size + label_vector_size
    decoder_vector_size = feature_vector_size + autoencoder_vector_size
    LSTM_layers = 1
    print("Loading data from disk...")
    data = torch.load(data_file_name)
    data = np.array(data)
    print("Loaded!")
    length = data.shape[0]
    train_size_start = 0
    train_size_end = length - 2 * int(length * 1.0 / 10)
    verif_size_start = length - 1 * int(length * 1.0 / 10)
    verif_size_end = length# - 1 * int(length * 1.0 / 10)

    # LSTM AE can be used in this place too, but it performs poorly.
    #    def __init__(self, feature_size,label_size, hidden_size, num_layers, isCuda):
    #ae = LSTMAE(feature_vector_size, label_vector_size, autoencoder_vector_size, LSTM_layers, False)
    ae = EncoderRNN(feature_vector_size, label_vector_size, LSTM_layers, isCuda).to(device)

    if state is not None:
        ae.load_state_dict(state)
    for epoch in range(epochs):
        print("Epoch:%d"%epoch)
        # print("Evaluating...")
        # f1_1 = []
        # precision_all = []
        # recall_all = []
        # #print("Train acc")
        # skip_counter = 0
        # for chart_index,chart_data in enumerate(data[verif_size_start:verif_size_end]):
        #     #print(np.array(chart_data).shape)
        #     if skip_long_seqs:
        #         if np.array(chart_data).shape[0] > max_sequence_length:
        #             skip_counter += 1
        #             continue
        #     # if np.array(chart_data).shape[0] > max_sequence_length * 3 / 4:
        #     #     print("Edge case:%d"%np.array(chart_data).shape[0])
        #     precision = 0
        #     recall = 0
        #     tp = fp = tn = fn = 0
        #     counter = 0
        #     # print(np.array(chart_data).shape)
        #     # exit()
        #     features = Variable(torch.tensor([anti_tr(chart_data)]),requires_grad=False).float().to(device)
        #     labels = Variable(torch.tensor([tr(chart_data)]),requires_grad=False).float().to(device)
        #     #print(labels.size())
        #     #print("Before eval ae")
        #     #print("Features size:%s"%str(features.size()))
        #     result = ae(features).detach()
        #     #print("After eval ae")
        #     result_max_index = np.argmax(result,axis=2)
        #     correct_max_index = np.argmax(labels,axis=2)
        #     for sample_index, sample_value in enumerate(result_max_index):
        #         for index, value in enumerate(sample_value):
        #             if labels[sample_index][index][0] == labels[sample_index][index][1]:
        #                 print("Something wrong is in the data.")
        #             if correct_max_index[sample_index][index] == 0:
        #                 counter += 1
        #                 if result_max_index[sample_index][index] == 0:
        #                     tp += 1
        #                 else:
        #                     fn += 1
        #             else:
        #                 if result_max_index[sample_index][index] == 0:
        #                     fp += 1
        #                 else:
        #                     tn += 1
        #
        #         try:
        #             precision = 1.0 * tp / (tp + fp)
        #             recall = 1.0 * tp / (tp + fn)
        #             f1_score = 2 / (1 / precision + 1 / recall)
        #             if chart_index % 10 == 1:
        #                 print("Chart #%d\tF1 = %f\t(tp=%d tn=%d fp=%d fn=%d)" % (chart_index, f1_score, tp, tn, fp, fn))
        #             #print("Precision:%f Recall:%f"%(precision,recall))
        #             # print("F1 score = %f" % f1_score)
        #             f1_1.append(f1_score)
        #             precision_all.append(precision)
        #             recall_all.append(recall)
        #             #print(precision_all)
        #         except:
        #             print("Chart #%d\tF1 = %f\t(tp=%d tn=%d fp=%d fn=%d)" % (chart_index, 0, tp, tn, fp, fn))
        #             f1_1.append(0)
        #             precision_all.append(0)
        #             recall_all.append(0)
        #
        # print("Skipped %d charts."%skip_counter)
        # print("F1-score:%f stdev:%f"%(avg(f1_1),std(f1_1)))
        # print("precision:%f stdev:%f"%(avg(precision_all),std(precision_all)))
        # print("recall:%f stdev:%f"%(avg(recall_all),std(recall_all)))
        # # print("Average F1 Score: %f / changed threshold:%f"%(avg(f1_1),avg(f1_2)))
        # print("Evaluate ends.")
        last_loss = -1
        for chart_index,chart_data in enumerate(tqdm(data[train_size_start:train_size_end])):
            #print(np.array(chart_data).shape)
            if skip_long_seqs:
                if np.array(chart_data).shape[0] > max_sequence_length:
                    skip_counter += 1
                    continue
            # exit()

            features = Variable(torch.tensor([anti_tr(chart_data)]),requires_grad=False).float().to(device)
            labels = Variable(torch.tensor([tr(chart_data)]),requires_grad=False).float().to(device)
            criterion = weighted_mse_loss()
            criterion.setWeight(Variable(torch.FloatTensor([1, 0.2])))
            optimizer = optim.Adam(ae.parameters(), weight_decay=0.0001)
            def closure():
                optimizer.zero_grad()
                out = ae(features)
                loss = criterion(out, labels)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            last_loss = loss
            #pred = ae(features, labels)
        print("Last loss is %f."%last_loss)
        print("Saving state to %s..."%model_state_name)
        state_dict = ae.state_dict()
        torch.save(state_dict, open(model_state_name, 'wb'), pickle_protocol=4)

# Evaluate the LSTM model.
def eval():
    print("Looking for latest checkpoint...")
    state = None
    try:
        state = torch.load(model_state_name)
        print("Read saved model.")
    except:
        print("No saved model found. Stop.")
        return
    dtype = torch.float #if torch.cuda.is_available() else torch.FloatStorage
    if use_history_in_training:
        feature_vector_size = 302
    else:
        feature_vector_size = 30
    label_vector_size = 2
    autoencoder_vector_size = 4
    encoder_vector_size = feature_vector_size + label_vector_size
    decoder_vector_size = feature_vector_size + autoencoder_vector_size
    LSTM_layers = 1
    print("Loading data from disk...")
    data = torch.load(data_file_name)
    data = np.array(data)
    print("Loaded!")
    length = data.shape[0]

    train_size_start = 0
    train_size_end = length - 2 * int(length * 1.0 / 10)
    verif_size_start = length - 1 * int(length * 1.0 / 10)
    verif_size_end = length# - 1 * int(length * 1.0 / 10)

    #ae = LSTMAE(feature_vector_size, label_vector_size, autoencoder_vector_size, LSTM_layers, False)
    ae = EncoderRNN(feature_vector_size, label_vector_size, LSTM_layers, isCuda).to(device)
    if state is not None:
        ae.load_state_dict(state)
    for epoch in range(1):
        print("Evaluating LSTM model...")
        f1_1 = []
        precision_all = []
        recall_all = []
        skip_counter = 0
        for chart_index,chart_data in enumerate(data[verif_size_start:verif_size_end]):
            if skip_long_seqs:
                if np.array(chart_data).shape[0] > max_sequence_length:
                    skip_counter += 1
                    continue
            precision = 0
            recall = 0
            tp = fp = tn = fn = 0
            counter = 0
            features = Variable(torch.tensor([anti_tr(chart_data)]),requires_grad=False).float().to(device)
            labels = Variable(torch.tensor([tr(chart_data)]),requires_grad=False).float().to(device)
            result = ae(features).detach()
            result_max_index = np.argmax(result,axis=2)
            correct_max_index = np.argmax(labels,axis=2)
            for sample_index, sample_value in enumerate(result_max_index):
                for index, value in enumerate(sample_value):
                    if labels[sample_index][index][0] == labels[sample_index][index][1]:
                        print("Something wrong is in the data.")
                    if correct_max_index[sample_index][index] == 0:
                        counter += 1
                        if result_max_index[sample_index][index] == 0:
                            tp += 1
                        else:
                            fn += 1
                    else:
                        if result_max_index[sample_index][index] == 0:
                            fp += 1
                        else:
                            tn += 1
                try:
                    precision = 1.0 * tp / (tp + fp)
                    recall = 1.0 * tp / (tp + fn)
                    f1_score = 2 / (1 / precision + 1 / recall)
                    if chart_index % 10 == 1:
                        print("Chart #%d\tF1 = %f\t(tp=%d tn=%d fp=%d fn=%d)" % (chart_index, f1_score, tp, tn, fp, fn))
                    f1_1.append(f1_score)
                    precision_all.append(precision)
                    recall_all.append(recall)
                except:
                    print("Chart #%d\tF1 = %f\t(tp=%d tn=%d fp=%d fn=%d)" % (chart_index, 0, tp, tn, fp, fn))
                    f1_1.append(0)
                    precision_all.append(0)
                    recall_all.append(0)
        print("Skipped %d charts."%skip_counter)
        print("F1-score:%f stdev:%f"%(avg(f1_1),std(f1_1)))
        print("precision:%f stdev:%f"%(avg(precision_all),std(precision_all)))
        print("recall:%f stdev:%f"%(avg(recall_all),std(recall_all)))
        print("Evaluate ends.")

# if __name__ == '__main__':
#     train()