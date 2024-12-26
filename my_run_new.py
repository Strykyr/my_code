import torch
import torch.nn as nn
from typing import Tuple, Optional, List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# 输入的历史look_back步，和预测未来的T步
look_back = 100
num_features = 2  # 输入特证数
c_out = 2 #输出维度
#pre_len = 1   # 预测长度
pred_len = 50
T = 50


epochs = 15  # 迭代次数(15)



embed_dim = 32  # 嵌入维度
dense_dim = 32  # 隐藏层神经元个数
num_heads = 4  # 头数
dropout_rate = 0.01  # 失活率
num_blocks = 3  # 编码器解码器数
learn_rate = 0.001  # 学习率
batch_size = 500  # 批大小


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    #return 0.01*(u / d).mean(-1)
    return 0.01*(u / d)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr




class sLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()

        # Store the input and hidden size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Combine the Weights and Recurrent weights into a single matrix
        self.W = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.randn(self.input_size + self.hidden_size, 4 * self.hidden_size)
            ),
            requires_grad=True,
        )
        # Combine the Bias into a single matrix
        if self.bias:
            self.B = nn.Parameter(
                (torch.zeros(4 * self.hidden_size)), requires_grad=True
            )

    def forward(
        self,
        x: torch.Tensor,
        internal_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        # Unpack the internal state
        h, c, n, m = internal_state  # (batch_size, hidden_size)

        # Combine the weights and the input
        combined = torch.cat((x, h), dim=1)  # (batch_size, input_size + hidden_size)
        # Calculate the linear transformation
        gates = torch.matmul(combined, self.W)  # (batch_size, 4 * hidden_size)

        # Add the bias if included
        if self.bias:
            gates += self.B

        # Split the gates into the input, forget, output and stabilization gates
        z_tilda, i_tilda, f_tilda, o_tilda = torch.split(gates, self.hidden_size, dim=1)

        # Calculate the activation of the states
        z_t = torch.tanh(z_tilda)  # (batch_size, hidden_size)
        # Exponential activation of the input gate
        i_t = torch.exp(i_tilda)  # (batch_size, hidden_size)
        # Exponential activation of the forget gate
        f_t = torch.sigmoid(f_tilda)  # (batch_size, hidden_size)

        # Sigmoid activation of the output gate
        o_t = torch.sigmoid(o_tilda)  # (batch_size, input_size)
        # Calculate the stabilization state
        m_t = torch.max(torch.log(f_t) + m, torch.log(i_t))  # (batch_size, hidden_size)
        # Calculate the input stabilization state
        i_prime = torch.exp(i_tilda - m_t)  # (batch_size, hidden_size)

        # Calculate the new internal states
        c_t = f_t * c + i_prime * z_t  # (batch_size, hidden_size)
        n_t = f_t * n + i_prime  # (batch_size, hidden_size)

        # Calculate the stabilized hidden state
        h_tilda = c_t / n_t  # (batch_size, hidden_size)

        # Calculate the new hidden state
        h_t = o_t * h_tilda  # (batch_size, hidden_size)
        return h_t, (
            h_t,
            c_t,
            n_t,
            m_t,
        )  # (batch_size, hidden_size), (batch_size, hidden_size), (batch_size, hidden_size), (batch_size, hidden_size)

    def init_hidden(
        self, batch_size: int, **kwargs
    #) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # ==========================  改
    ):
        return (
            torch.zeros(batch_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
        )


class sLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        self.cells = nn.ModuleList(
            [
                sLSTMCell(input_size if layer == 0 else hidden_size, hidden_size, bias)
                for layer in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        # Permute the input tensor if batch_first is True
        if self.batch_first:
            x = x.permute(1, 0, 2)

        # Initialize the hidden states if not provided
        if hidden_states is None:
            hidden_states = self.init_hidden(x.size(1), device=x.device, dtype=x.dtype)
        else:
            # Check if the hidden states are of the correct length
            if len(hidden_states) != self.num_layers:
                raise ValueError(
                    f"Expected hidden states of length {self.num_layers}, but got {len(hidden_states)}"
                )
            if any(state[0].size(0) != x.size(1) for state in hidden_states):
                raise ValueError(
                    f"Expected hidden states of batch size {x.size(1)}, but got {hidden_states[0][0].size(0)}"
                )

        H, C, N, M = [], [], [], []

        for layer, cell in enumerate(self.cells):
            lh, lc, ln, lm = [], [], [], []
            for t in range(x.size(0)):
                h_t, hidden_states[layer] = (
                    cell(x[t], hidden_states[layer])
                    if layer == 0
                    else cell(H[layer - 1][t], hidden_states[layer])
                )
                lh.append(h_t)
                lc.append(hidden_states[layer][0])
                ln.append(hidden_states[layer][1])
                lm.append(hidden_states[layer][2])

            H.append(torch.stack(lh, dim=0))
            C.append(torch.stack(lc, dim=0))
            N.append(torch.stack(ln, dim=0))
            M.append(torch.stack(lm, dim=0))

        H = torch.stack(H, dim=0)
        C = torch.stack(C, dim=0)
        N = torch.stack(N, dim=0)
        M = torch.stack(M, dim=0)

        return H[-1], (H, C, N, M)

    def init_hidden(
        self, batch_size: int, **kwargs
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:

        return [cell.init_hidden(batch_size, **kwargs) for cell in self.cells]
class mLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Initialize weights and biases
        self.W_i = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(input_size, hidden_size)),
            requires_grad=True,
        )
        self.W_f = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(input_size, hidden_size)),
            requires_grad=True,
        )
        self.W_o = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(input_size, hidden_size)),
            requires_grad=True,
        )
        self.W_q = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(input_size, hidden_size)),
            requires_grad=True,
        )
        self.W_k = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(input_size, hidden_size)),
            requires_grad=True,
        )
        self.W_v = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(input_size, hidden_size)),
            requires_grad=True,
        )

        if self.bias:
            self.B_i = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            self.B_f = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            self.B_o = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            self.B_q = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            self.B_k = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            self.B_v = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

    def forward(
        self,
        x: torch.Tensor,
        internal_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Get the internal state
        C, n, m = internal_state

        #  Calculate the input, forget, output, query, key and value gates
        i_tilda = (
            torch.matmul(x, self.W_i) + self.B_i
            if self.bias
            else torch.matmul(x, self.W_i)
        )
        f_tilda = (
            torch.matmul(x, self.W_f) + self.B_f
            if self.bias
            else torch.matmul(x, self.W_f)
        )
        o_tilda = (
            torch.matmul(x, self.W_o) + self.B_o
            if self.bias
            else torch.matmul(x, self.W_o)
        )
        q_t = (
            torch.matmul(x, self.W_q) + self.B_q
            if self.bias
            else torch.matmul(x, self.W_q)
        )
        k_t = (
            torch.matmul(x, self.W_k) / torch.sqrt(torch.tensor(self.hidden_size))
            + self.B_k
            if self.bias
            else torch.matmul(x, self.W_k) / torch.sqrt(torch.tensor(self.hidden_size))
        )
        v_t = (
            torch.matmul(x, self.W_v) + self.B_v
            if self.bias
            else torch.matmul(x, self.W_v)
        )

        # Exponential activation of the input gate
        i_t = torch.exp(i_tilda)
        f_t = torch.sigmoid(f_tilda)
        o_t = torch.sigmoid(o_tilda)

        # Stabilization state
        m_t = torch.max(torch.log(f_t) + m, torch.log(i_t))
        i_prime = torch.exp(i_tilda - m_t)

        # Covarieance matrix and normalization state
        C_t = f_t.unsqueeze(-1) * C + i_prime.unsqueeze(-1) * torch.einsum(
            "bi, bk -> bik", v_t, k_t
        )
        n_t = f_t * n + i_prime * k_t

        normalize_inner = torch.diagonal(torch.matmul(n_t, q_t.T))
        divisor = torch.max(
            torch.abs(normalize_inner), torch.ones_like(normalize_inner)
        )
        h_tilda = torch.einsum("bkj,bj -> bk", C_t, q_t) / divisor.view(-1, 1)
        h_t = o_t * h_tilda

        return h_t, (C_t, n_t, m_t)

    def init_hidden(
        self, batch_size: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, self.hidden_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
        )


class mLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        self.cells = nn.ModuleList(
            [
                mLSTMCell(input_size if layer == 0 else hidden_size, hidden_size, bias)
                for layer in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Permute the input tensor if batch_first is True
        if self.batch_first:
            x = x.permute(1, 0, 2)

        if hidden_states is None:
            hidden_states = self.init_hidden(x.size(1), device=x.device, dtype=x.dtype)
        else:
            # Check if the hidden states are of the correct length
            if len(hidden_states) != self.num_layers:
                raise ValueError(
                    f"Expected hidden states of length {self.num_layers}, but got {len(hidden_states)}"
                )
            if any(state[0].size(0) != x.size(1) for state in hidden_states):
                raise ValueError(
                    f"Expected hidden states of batch size {x.size(1)}, but got {hidden_states[0][0].size(0)}"
                )

        H, C, N, M = [], [], [], []

        for layer, cell in enumerate(self.cells):
            lh, lc, ln, lm = [], [], [], []
            for t in range(x.size(0)):
                h_t, hidden_states[layer] = (
                    cell(x[t], hidden_states[layer])
                    if layer == 0
                    else cell(H[layer - 1][t], hidden_states[layer])
                )
                lh.append(h_t)
                lc.append(hidden_states[layer][0])
                ln.append(hidden_states[layer][1])
                lm.append(hidden_states[layer][2])

            H.append(torch.stack(lh, dim=0))
            C.append(torch.stack(lc, dim=0))
            N.append(torch.stack(ln, dim=0))
            M.append(torch.stack(lm, dim=0))

        H = torch.stack(H, dim=0)
        C = torch.stack(C, dim=0)
        N = torch.stack(N, dim=0)
        M = torch.stack(M, dim=0)

        return H[-1], (H, C, N, M)

    def init_hidden(
        self, batch_size: int, **kwargs
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return [cell.init_hidden(batch_size, **kwargs) for cell in self.cells]


# 构建Transformer模型
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate):
        super(TransformerEncoder, self).__init__()

        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense1 = nn.Linear(embed_dim, dense_dim)
        self.dense2 = nn.Linear(dense_dim, embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        attn_output, _ = self.mha(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)

        dense_output = self.dense1(out1)
        dense_output = self.dense2(dense_output)
        dense_output = self.dropout2(dense_output)
        out2 = self.layernorm2(out1 + dense_output)

        return out2


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate):
        super(TransformerDecoder, self).__init__()

        self.mha1 = nn.MultiheadAttention(embed_dim, num_heads)
        self.mha2 = nn.MultiheadAttention(embed_dim, num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense1 = nn.Linear(embed_dim, dense_dim)
        self.dense2 = nn.Linear(dense_dim, embed_dim)
        self.layernorm4 = nn.LayerNorm(embed_dim)
        self.dropout4 = nn.Dropout(dropout_rate)

    def forward(self, inputs, encoder_outputs):
        attn1, _ = self.mha1(inputs, inputs, inputs)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(inputs + attn1)

        attn2, _ = self.mha2(out1, encoder_outputs, encoder_outputs)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)

        dense_output = self.dense1(out2)
        dense_output = self.dense2(dense_output)
        dense_output = self.dropout3(dense_output)
        out3 = self.layernorm3(out2 + dense_output)

        decoder_output = self.dense1(out3)
        decoder_output = self.dense2(decoder_output)
        decoder_output = self.dropout4(decoder_output)
        out4 = self.layernorm4(out3 + decoder_output)

        return out4


class Transformer(nn.Module):
    # 定义SLTM的参数
    input_size = 64
    hidden_size = 128
    num_layers = 1
    seq_length = 10
    batch_size = 32
    dropout = 0.1
    def __init__(self, num_features, embed_dim, dense_dim, num_heads, dropout_rate, num_blocks, output_sequence_length):
        super(Transformer, self).__init__()

        self.embedding = nn.Linear(num_features, embed_dim)
        self.transformer_encoder = nn.ModuleList(
            [TransformerEncoder(embed_dim, dense_dim, num_heads, dropout_rate) for _ in range(num_blocks)])
        #self.transformer_decoder = nn.ModuleList(
        #    [TransformerDecoder(embed_dim, dense_dim, num_heads, dropout_rate) for _ in range(num_blocks)])
        #self.final_layer = nn.Linear(embed_dim * look_back, output_sequence_length)
        
        
        self.final_layer = nn.Linear(embed_dim, c_out, bias=True)
        
        self.xLT=sLSTM(input_size=num_features,hidden_size=num_features,num_layers=2)
        #self.xLT = mLSTM(input_size=num_features, hidden_size=num_features, num_layers=3)
    def forward(self, inputs):
        inputs, hidden_state = self.xLT(inputs)
        encoder_inputs = inputs
        decoder_inputs = inputs
        encoder_outputs = self.embedding(encoder_inputs)
        for i in range(len(self.transformer_encoder)):
            encoder_outputs = self.transformer_encoder[i](encoder_outputs)

        outputs = self.final_layer(encoder_outputs)
        #outputs = outputs.view(-1, T)
        return outputs[:, -pred_len:, :] 
   #  ===========================================   改 
   #     decoder_outputs = self.embedding(decoder_inputs)
   #     for i in range(len(self.transformer_decoder)):
   #        decoder_outputs = self.transformer_decoder[i](decoder_outputs, encoder_outputs)

   #     decoder_outputs = decoder_outputs.view(-1, decoder_outputs.shape[1] * decoder_outputs.shape[2])
   #    decoder_outputs = self.final_layer(decoder_outputs)
   #    decoder_outputs = decoder_outputs.view(-1, T)
   #    return decoder_outputs


# 创建模型实例
model = Transformer(num_features=num_features, embed_dim=embed_dim, dense_dim=dense_dim, num_heads=num_heads,
                    dropout_rate=dropout_rate, num_blocks=num_blocks, output_sequence_length=T)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)



"""
# 定义训练集和测试集的数据加载器
class MyDataset(Dataset):
    def __init__(self, data_X, data_Y):
        self.data_X = data_X
        self.data_Y = data_Y

    def __getitem__(self, index):
        x = self.data_X[index]
        y = self.data_Y[index]
        return x, y

    def __len__(self):
        return len(self.data_X)


train_dataset = MyDataset(trainX, trainY)
val_dataset = MyDataset(valX, valY)
test_dataset = MyDataset(testX, testY)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
"""

# 定义训练集和测试集的数据加载器



folder_path = "./data/water/"
model_path = "my_model/"

# 遍历文件夹中的所有文件
def get_data(folder_path):
    train = []
    test = []
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 按照文件名进行排序（字母顺序）
    csv_files.sort()
    for filename in csv_files:
            # 训练集                 
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path,dtype=float)
            df = df.set_index('Time').sort_index()
            if('test' not in filename):
                train.append(df)
            else:
                test.append(df)
    return train, test, train[0].index;
    # 获取训练数据和测试数据


# 数据集处理，获取对应的数据以及标签
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
       
def my_data(split,data):

    scaler = MinMaxScaler()
    seq = []
    # 训练和验证
    if split != 'test':    
        for i in range(len(data)):
            x = data[i]
            # 归一化
            normalized_data = scaler.fit_transform(x)
            for j in range(0,18):
                for i in range(len(normalized_data) - 150):# 预测30s，但是label大点(100)
                    train_seq,train_label = [],[]
                    for k in range(i,i+100):
                        #train_seq.append([normalized_data[k,j],normalized_data[k,j+18]])
                        # 温度加顶棚温度
                        train_seq.append([normalized_data[k,j],normalized_data[k,-1]])
                    # 未来的10个时间点3s
                    for k in range(i+100,i+150):
                        train_label.append([normalized_data[k,j], normalized_data[k,-1]])
                    train_seq = torch.FloatTensor(train_seq).reshape(-1,2)
                    train_label = torch.FloatTensor(train_label).reshape(-1,2)
                    seq.append((train_seq, train_label))
        seq = MyDataset(seq)
        # 多线程取数据集
        seq = DataLoader(dataset=seq, batch_size=600, shuffle=True, num_workers=4, drop_last=True)
        return seq
    # 测试集
    else:
        # split
        scaler = MinMaxScaler()

        x = data
        # 归一化
        normalized_data = scaler.fit_transform(x)
        for i in range(len(normalized_data) - 150):# 21秒
            test_seq = []
            test_label = []
            for k in range(i,i+100):
            
                test_seq.append([normalized_data[k,0],normalized_data[k,-1]])
                
                #===== 顶棚
                #test_seq.append([normalized_data[k,0],normalized_data[k,-1]])
            # 10个时间点3s
            for k in range(i+100,i+150):
                test_label.append([normalized_data[k,0], normalized_data[k,-1]])
                                
                #==== 顶棚
                #test_label.append([normalized_data[k,0], normalized_data[k,-1]])
            test_seq = torch.FloatTensor(test_seq).reshape(-1,2)
            test_label = torch.FloatTensor(test_label).reshape(-1,2)
            seq.append((test_seq, test_label))
        
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

        return seq


train_data,test_data,time = get_data(folder_path)
print('train len >>>>>>>> ',len(train_data))
print('test len >>>>>>>> ',len(test_data))

train_loader = my_data("train",train_data)
test_loader = my_data("eval",test_data)
val_loader = test_loader


# ==============================================================
train_losses = []
val_losses = []

# 训练模型
for epoch in range(epochs):
    model.train()
    train_loss = []
    val_l = []
    for inputs, labels in tqdm(train_loader, position=0):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model.forward(inputs)
        
        #loss = criterion(outputs, labels)
        
        #=========  改
        outputs = outputs[:, -pred_len:, 0:]
        labels = labels[:, -pred_len:, 0:].to(device)
        loss = criterion(outputs, labels)
        
        train_loss.append(loss.item())
        
        loss.backward()
        optimizer.step()
    
    train_loss = np.average(train_loss)
    train_losses.append(train_loss)
    # 在验证集上计算损失
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, position=0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.forward(inputs)
            #val_loss = criterion(outputs, labels)
            
            
            # ===========改
            outputs = outputs[:, -pred_len:, 0:]
            labels = labels[:, -pred_len:, 0:].to(device)
            val_loss = criterion(outputs, labels)
            val_l.append(val_loss.item())
    
    
    val_l = np.average(val_l)
    val_losses.append(val_l)
    
    
    # ====================== 改
    # 每个epoch打印一次训练和验证损失
    #print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')



# 保存模型 =============================
dir = os.path.join(folder_path, model_path)
check_path = os.path.join(folder_path, model_path,'best_model.pth')
# 检查路径的父目录是否存在，如果不存在则创建它
dir_path = os.path.dirname(check_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path, exist_ok=True)
torch.save(model.state_dict(), check_path)
#================加载模型
model.load_state_dict(torch.load(check_path))


# ========================= 改
# 保存损失
file_name = os.path.join(dir,'loss.csv')
epoch = range(1, len(train_losses)+1)
# 保存到 CSV 文件
d = {
    "train": train_losses,
    "test": val_losses, 
}
pd.DataFrame(d).to_csv(file_name, index=False)



# 测试模型
model.eval()
predictions = []

dict = ['60','100','220']

# ==== 改
k = -1
for j in range(len(test_data)):
    k = k+1
    data = test_data[j]
    test_loader = my_data("test",data)
    # ========================= 改
    preds,preds_t,preds_all = [],[],[]
    trues,trues_t,trues_all = [],[],[]


    # ===================== 改



    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, position=0):
            inputs = inputs.to(device)
            outputs = model(inputs)
            #predictions.extend(outputs.cpu().numpy())

            #================================= 改
            # 温度
            pred = outputs[:, -1, 0]  # outputs.detach().cpu().numpy()  # .squeeze()
            true = labels[:, -1, 0].to(device)  # batch_y.detach().cpu().numpy()  # .squeeze()
            
            # 顶棚温度
            pre_t = outputs[:, -1, -1]
            true_t = labels[:, -1, -1].to(device)

            preds.append(pred)
            trues.append(true)
            preds_t.append(pre_t)
            trues_t.append(true_t)        

    # 测试集数据反归一化
    #predictions = scaler2.inverse_transform(predictions)
    #labels = scaler2.inverse_transform(labels)

    #======================改
    min_val = test_data[j].iloc[:,0].min()
    max_val = test_data[j].iloc[:,0].max()
    
    # ==== 顶棚
#    min_val = test_data[j].iloc[:,0].min()
#    max_val = test_data[j].iloc[:,0].max()
    
    #顶棚
    min_val_t = test_data[j].iloc[:,-1].min()
    max_val_t = test_data[j].iloc[:,-1].max()

    preds = torch.FloatTensor(preds).detach().cpu().numpy()
    trues = torch.FloatTensor(trues).detach().cpu().numpy()
    preds_t = torch.FloatTensor(preds_t).detach().cpu().numpy()
    trues_t = torch.FloatTensor(trues_t).detach().cpu().numpy()

    preds = np.array([x * (max_val-min_val) + min_val for x in preds])
    trues =np.array([x * (max_val-min_val) + min_val for x in trues])
    preds_t = np.array([x * (max_val_t-min_val_t) + min_val_t for x in preds_t])
    trues_t = np.array([x * (max_val_t-min_val_t) + min_val_t for x in trues_t])


    #======== 改
    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    mae_t, mse_t, rmse_t, mape_t, mspe_t, rse_t, corr_t = metric(preds_t, trues_t)

    #====== 预测保存,测点（1or5）
    time = time[-len(trues):]
    
    d = {
        "time": time,  
        "true": trues,
        "pre": preds, 
    }
    pd.DataFrame(d).to_csv(dir + "/" +  dict[k] + '_ssensor1.csv', index=False)


    # 顶棚=============
    
    d = {
        "time": time,  
        "true": trues_t,
        "pre": preds_t, 
    }
    pd.DataFrame(d).to_csv(dir + "/" +  dict[k] + '_ceiling.csv', index=False)


    # 保存到指标
    f = open(dir + dict[k]  + "result.txt", 'a')
    # door
    f.write("测点====>>>>>>>>>>>>>>>>>>>>>>." + "  \n")
    # water+exhaust
    #f.write(str(dict_dir[k]) + '##' + str(dict[k]) + ">>>>>>>>>>>>>>>>>>>>>>." + "  \n")
    f.write('mse:{}, mae:{}, rmse:{},mape:{},mspe:{},rse:{}, corr:{}'.format(mse, mae,rmse, mape, mspe, rse, corr))
    f.write('\n')
    
    f.write("顶棚======>>>>>>>>>>>>>>>>>>>>>>." + "  \n")
    # water+exhaust
    #f.write(str(dict_dir[k]) + '##' + str(dict[k]) + ">>>>>>>>>>>>>>>>>>>>>>." + "  \n")
    f.write('mse_t:{}, mae_t:{}, rmse_t:{},mape_t:{},mspe_t:{},rse_t:{}, corr_t:{}'.format(mse_t, mae_t,rmse_t, mape_t, mspe_t, rse_t, corr_t))
    f.write('\n')
    
    f.write('\n')
    f.close()

