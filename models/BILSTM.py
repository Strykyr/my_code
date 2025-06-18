
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp
# initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_input = 2; n_hidden = 64

class Model(nn.Module):

    def __init__(self,configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        # intialise weights of the attention mechanism
        self.weight = nn.Parameter(torch.zeros(1)).to(device)
        self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # intialise cnn structure
        self.cnn = nn.Sequential(
            #nn.Conv1d(in_channels=1, out_channels=n_hidden, kernel_size=3, stride=1, padding=1), # ((5 + 1*2 - 3)/1 + 1) = 5
            # kernel_size 为序列长度
            nn.Conv1d(in_channels=configs.d_model, out_channels= self.d_ff, kernel_size=1), # d_ff = 2048
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(n_hidden, eps=1e-5),
            nn.BatchNorm1d(self.d_ff, eps=1e-5),
            nn.Dropout(0.1),

            #nn.Conv1d(in_channels=n_hidden, out_channels=n_hidden, kernel_size=3, stride=1, padding=1), # ((5 + 1*2 - 3)/1 + 1) = 5
            nn.Conv1d(in_channels=self.d_ff, out_channels=configs.d_model, kernel_size=1),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(n_hidden, eps=1e-5),
            nn.BatchNorm1d(configs.d_model, eps=1e-5),

            nn.Flatten(),
           # nn.Linear(n_input * n_hidden, n_input)
           nn.Linear(configs.d_model*self.seq_len, self.seq_len*n_input) # 100序列长度
        )

        # intialise lstm structure
        #self.lstm = nn.LSTM(n_input, n_hidden, batch_first=True, bidirectional=False)
        #self.linear = nn.Linear(n_hidden, 2)

        self.lstm = nn.LSTM(n_input, n_hidden, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(n_hidden,self.pred_len*2)


    def forward(self, x):

        #x2 = self.enc_embedding(x,None)
        #cnn_output = self.cnn(x)
        #x2 = x2.permute(0, 2, 1) 
        #cnn_output = self.cnn(x2)
        #cnn_output = cnn_output.view(-1, 1, n_input)
        #cnn_output = cnn_output.view(-1,self.seq_len,n_input)
        #residuals = x + self.weight * cnn_output

        #_, (h_n, _)  = self.lstm(x)
        _, (h_n, _)  = self.lstm(x)
        y_hat = self.linear(h_n[-1,:,:])

        return y_hat.view(-1,self.pred_len,2)


