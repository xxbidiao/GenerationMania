# LSTMAE.py
# This is an implementation of LSTM auto encoder-decoder.
# Some part of the code is reused as the LSTM baseline.
#Code based on https://discuss.pytorch.org/t/lstm-autoencoder-architecture/8524/2

from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from .data_utils import *

# Set this to something other than (1,1) to enable extra step to auto-shrink the LSTM sequence length.
maxpool_kernel = (1,1)

add_additional_fc_layer = False

add_labels_to_encoder_input = False

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if add_additional_fc_layer:
            hidden_size_lstm = hidden_size * 8
        else:
            hidden_size_lstm = hidden_size
        self.isCuda = isCuda
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_lstm,
            num_layers=num_layers,
            #dropout=0.5, #enable if num_layers > 1
            batch_first=True

        )
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size_lstm, hidden_size)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(maxpool_kernel)

        # initialize weights
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(3))
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, features,labels=None):
        #labels are ignored.
        if add_labels_to_encoder_input:
            input = pad_sequences_torch(features,labels)
        else:
            input = features
        if add_additional_fc_layer:
            hidden_size_lstm = self.hidden_size * 8
        else:
            hidden_size_lstm = self.hidden_size
        # Sadly, this model is untrainable using GPU on our setup since the sequence is too long (10k+).
        tt = torch#.cuda if self.isCuda else torch
        h0 = Variable(tt.FloatTensor(self.num_layers, input.size(0), hidden_size_lstm))
        c0 = Variable(tt.FloatTensor(self.num_layers, input.size(0), hidden_size_lstm))
        encoded_input, hidden = self.lstm(input, (h0, c0))
        if add_additional_fc_layer:
            encoded_input = self.relu1(encoded_input)
            encoded_input = self.fc1(encoded_input)
        encoded_input = self.relu2(encoded_input)
        #encoded_input = self.maxpool(encoded_input.unsqueeze(0)).squeeze(0)
        return encoded_input


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, isCuda):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.isCuda = isCuda
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=output_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=maxpool_kernel,mode='bilinear')

        # initialize weights
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(3))

    def forward(self, features,encoded):
        # Upsample version of the same code.

        # upsampled_encoded = self.upsample(encoded.unsqueeze(0)).squeeze(0)
        # #upsampled_encoded = upsampled_encoded.expand(-1,features.size()[1],-1)
        # delta_size = -upsampled_encoded.size()[1] + features.size()[1]
        # #print(delta_size)
        # if delta_size > 0:
        #     padding_arr = torch.zeros(
        #         (upsampled_encoded.size()[0],delta_size,upsampled_encoded.size()[2]),
        #     )
        #     #print(upsampled_encoded.size())
        #     #print(padding_arr.size())
        #
        #     upsampled_encoded = torch.cat((upsampled_encoded,padding_arr),dim=1)
        # #print(upsampled_encoded.size())
        # encoded_input = pad_sequences_torch(features,upsampled_encoded)
        encoded_input = pad_sequences_torch(features,encoded)
        tt = torch#.cuda if self.isCuda else torch
        h0 = Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size))
        c0 = Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size))
        decoded_output, hidden = self.lstm(encoded_input, (h0, c0))
        decoded_output = self.relu(decoded_output)
        return decoded_output


class LSTMAE(nn.Module):
    def __init__(self, feature_size,label_size, hidden_size, num_layers, isCuda):
        super(LSTMAE, self).__init__()
        if add_labels_to_encoder_input:
            encoder_input_size = feature_size+label_size
        else:
            encoder_input_size = feature_size
        decoder_input_size = feature_size+hidden_size
        self.encoder = EncoderRNN(encoder_input_size, hidden_size, num_layers, isCuda)
        self.decoder = DecoderRNN(decoder_input_size, label_size, num_layers, isCuda)

    def forward(self, features,labels):
        encoded_input = self.encoder(features,labels)
        decoded_output = self.decoder(features,encoded_input)
        return decoded_output

    def encode(self,features,labels):
        encoded_input = self.encoder(features,labels)
        return encoded_input