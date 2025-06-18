from exp.exp_basic import Exp_Basic
from models import Attention, Informer, Autoformer, Transformer, DLinear, Linear, NLinear,Resnet_LSTM,Resnet,LSTM
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import argparse
import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings('ignore')



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
    return train, test, test[0].index;
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
                for i in range(len(normalized_data) - 100):# 预测30s，但是label大点(100)
                #for i in range(len(normalized_data) - 190):# 预测30s，但是label大点(100)
                    # 21s => 15s   (70  => 50)
                    # 12s => 15s     (40 => 50)
                    train_seq,train_label = [],[]
                    #for k in range(i,i+100):
                    for k in range(i,i+75):
                        #train_seq.append([normalized_data[k,j],normalized_data[k,j+18]])
                        # 温度加顶棚温度
                        train_seq.append([normalized_data[k,j],normalized_data[k,-1]])
                    # 未来的10个时间点3s
                    for k in range(i+75,i+100):
                    #for k in range(i+100,i+190):
                        train_label.append([normalized_data[k,j], normalized_data[k,-1]])
                    train_seq = torch.FloatTensor(train_seq).reshape(-1,2)
                    train_label = torch.FloatTensor(train_label).reshape(-1,2)
                    seq.append((train_seq, train_label))
        seq = MyDataset(seq)
        # 多线程取数据集
        seq = DataLoader(dataset=seq, batch_size=500, shuffle=True, num_workers=4, drop_last=True)
        return seq
    # 测试集
    else:
        # split
        scaler = MinMaxScaler()

        x = data
        # 归一化
        normalized_data = scaler.fit_transform(x)
        #for i in range(len(normalized_data) - 160):# 21秒
        for i in range(len(normalized_data) - 100):
            test_seq = []
            test_label = []
            #for k in range(i,i+100):
            for k in range(i,i+75):
                # 第一个测点
                test_seq.append([normalized_data[k,3],normalized_data[k,-1]])
                #第二个测定点
                #test_seq.append([normalized_data[k,10],normalized_data[k,-1]])
            # 10个时间点3s
            #for k in range(i+100,i+160):
            for k in range(i+75,i+100):
                # 第一个测点
                test_label.append([normalized_data[k,3], normalized_data[k,-1]])
                # 第二个
                #test_label.append([normalized_data[k,10], normalized_data[k,-1]])
            test_seq = torch.FloatTensor(test_seq).reshape(-1,2)
            test_label = torch.FloatTensor(test_label).reshape(-1,2)
            seq.append((test_seq, test_label))
        
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

        return seq






class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.folder_path = args.folder_path
        self.train_data,self.test_data,self.time = get_data(args.folder_path)
        print('train len >>>>>>>> ',len(self.train_data))
        print('test len >>>>>>>> ',len(self.test_data))
    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Attention_LSTM': Attention,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'Resnet_LSTM': Resnet_LSTM,
            'Resnet': Resnet,
            'LSTM': LSTM,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        print('vali>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if 'Linear' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, None, dec_inp, None)[0]
                    else:
                        outputs = self.model(batch_x, None, dec_inp, None)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        #train_data, train_loader = self._get_data(flag='train')
        # if not self.args.train_only:
        #     vali_data, vali_loader = self._get_data(flag='val')
        #     test_data, test_loader = self._get_data(flag='test')
        train_loader = my_data("train",self.train_data)
        test_loder = my_data("eval",self.test_data)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        loss_all = []
        test_loss_all = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
            #for i, (batch_x, batch_y,x_mask,y_mask) in enumerate(train_loader):mask就时间编码
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                # if self.args.use_amp:
                #     with torch.cuda.amp.autocast():
                #         if 'Linear' in self.args.model:
                #             outputs = self.model(batch_x)
                #         else:
                #             if self.args.output_attention:
                #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                #             else:
                #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                #         f_dim = -1 if self.args.features == 'MS' else 0
                #         outputs = outputs[:, -self.args.pred_len:, f_dim:]
                #         batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                #         loss = criterion(outputs, batch_y)
                #         train_loss.append(loss.item())
                # else:
                if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                elif 'Res' in self.args.model:
                            outputs = self.model(batch_x)
                elif 'LSTM' in self.args.model:
                            outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        outputs = self.model(batch_x, None, dec_inp, None)[0]
                        
                    else:
                        # 时间不编码
                        # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                        outputs = self.model(batch_x, None, dec_inp, None, batch_y)
                # print(outputs.shape,batch_y.shape)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            loss_all.append(train_loss)
            # 测试集当验证集用，5个epoch不下降保存模型
            if not self.args.train_only:
                test_loss = self.vali(None, test_loder, criterion)
                test_loss_all.append(test_loss)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, test_loss))
                early_stopping(test_loss, self.model, path)
            # else:
            # 根据训练集的损失值提早停止训练
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
            #     epoch + 1, train_steps, train_loss))
            # early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        #best_model_path = path + '/' + 'checkpoint.pth'
        # 保存到数据集对应的目录下
        best_model_path = self.folder_path + self.args.model  + '/checkpoints/' + 'checkpoint.pth'
        check_path = os.path.join(self.folder_path, self.args.model, 'checkpoints', 'checkpoint.pth')
        # 检查路径的父目录是否存在，如果不存在则创建它
        dir_path = os.path.dirname(check_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)


        torch.save(self.model.state_dict(), best_model_path)
        self.model.load_state_dict(torch.load(best_model_path))


        #======保存loss
        data_rows = [{'loss': r, 'test loss': p} for r, p in zip(loss_all, test_loss_all)]
            # 将字典列表转换为DataFrame
        df = pd.DataFrame(data_rows)
            # 将DataFrame保存到CSV文件
        df.to_csv(self.folder_path + self.args.model + '/'+ 'loss.csv', index=False)

        return self.model

    def test(self, setting, test=0):
        # test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.folder_path + self.args.model + '/checkpoints/', 'checkpoint.pth')))
        
        
        
        #door
        dict = ['11','12','13','ts1','ts2','ts3','21','22','23']


        #water
        # dict = ['100','200','400','120','220','340','120','220','340']
        # dict_dir = [20,20,20,50,50,50,80,80,80]

        # # exhaust
        #dict = ['100','200','400','120','180','300','120','180','300']
        #dict_dir = [2,2,2,6,6,6,10,10,10]
        k = -1
        for j in range(len(self.test_data)):
            k = k+1
            data = self.test_data[j]
            test_loader = my_data("test",data)

            preds,preds_t,preds_all = [],[],[]
            trues,trues_t,trues_all = [],[],[]
            inputx = []
            # folder_path = self.folder_path + '/test_results/'
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)

            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    # batch_x_mark = batch_x_mark.float().to(self.device)
                    # batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    # if self.args.use_amp:
                    #     with torch.cuda.amp.autocast():
                    #         if 'Linear' in self.args.model:
                    #             outputs = self.model(batch_x)
                    #         else:
                    #             if self.args.output_attention:
                    #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    #             else:
                    #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # else:
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                    elif 'Res' in self.args.model:
                            outputs = self.model(batch_x)
                    elif 'LSTM' in self.args.model and 'Attention_LSTM' != self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, None, dec_inp, None)[0]

                        else:
                            outputs = self.model(batch_x, None, dec_inp, None)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    # print(outputs.shape,batch_y.shape)
                    # outputs = outputs[:, -self.args.pred_len:, f_dim:]    
                    # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # outputs = outputs.detach().cpu().numpy()
                    # batch_y = batch_y.detach().cpu().numpy()
                    
                    # 预测3s的全部
                    outputs_all = outputs[:, -1, f_dim:]
                    batch_y_all = batch_y[:, -1, f_dim:].to(self.device)
                    # 取温度值 [batch_seize, seq_len, 2]  sqe_len = 10
                    # outputs = outputs[:, -1, 0]
                    # batch_y = batch_y[:, -1, 0].to(self.device)

                    

                    #温度
                    #pred = outputs[:, -1, 0]  # outputs.detach().cpu().numpy()  # .squeeze()
                    #true = batch_y[:, -1, 0].to(self.device)  # batch_y.detach().cpu().numpy()  # .squeeze()
                    #=========改
                    pred = outputs[:, -1, 0]  # outputs.detach().cpu().numpy()  # .squeeze()
                    true = batch_y[:, -1, 0].to(self.device)  # batch_y.detach().cpu().numpy()  # .squeeze()
                    # 顶棚温度
                
                    #pre_t = outputs[:, -1, -1]
                    #true_t = batch_y[:, -1, -1].to(self.device)
                    #========
                    pre_t = outputs[:, -1, -1]
                    true_t = batch_y[:, -1, -1].to(self.device)
                                        # gpu转numpy
                    preds_all.append(outputs_all.detach().cpu().numpy())
                    trues_all.append(batch_y_all.detach().cpu().numpy())

                    # preds_all.append(outputs_all)
                    # trues_all.append(batch_y_all)
                    preds.append(pred)
                    trues.append(true)
                    preds_t.append(pre_t)
                    trues_t.append(true_t)
                    inputx.append(batch_x.detach().cpu().numpy())
                    # if i % 20 == 0:
                    #     input = batch_x.detach().cpu().numpy()
                    #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    #      visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

            if self.args.test_flop:
                test_params_flop((batch_x.shape[1],batch_x.shape[2]))
                exit()
            # 画图
            # 第一个测点
            min_val = self.test_data[j].iloc[:,3].min()
            max_val = self.test_data[j].iloc[:,3].max()
            #print(min_val,"=============",max_val)
            # 第二个测点
            #min_val = self.test_data[j].iloc[:,10].min()
            #max_val = self.test_data[j].iloc[:,10].max()

            min_val_t = self.test_data[j].iloc[:,-1].min()
            max_val_t = self.test_data[j].iloc[:,-1].max()
            
            print(min_val_t,"######",max_val_t)

            preds = torch.FloatTensor(preds).detach().cpu().numpy()
            trues = torch.FloatTensor(trues).detach().cpu().numpy()
            preds_t = torch.FloatTensor(preds_t).detach().cpu().numpy()
            trues_t = torch.FloatTensor(trues_t).detach().cpu().numpy()


            # trues = np.array(trues)
            # preds = np.array(preds)
            # preds = preds*(max_val-min_val) + min_val
            # trues = trues*(max_val-min_val) + min_val
            preds = np.array([x * (max_val-min_val) + min_val for x in preds])
            trues =np.array([x * (max_val-min_val) + min_val for x in trues])

            preds_t = np.array([x * (max_val_t-min_val_t) + min_val_t for x in preds_t])
            trues_t = np.array([x * (max_val_t-min_val_t) + min_val_t for x in trues_t])


            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
            mae_t, mse_t, rmse_t, mape_t, mspe_t, rse_t, corr_t = metric(preds_t, trues_t)



                    # 保存csv文件
            # 将数据合并成一个字典列表，每个字典代表一行数据
            data_rows = [{'time': t, 'Real': r, 'Predicted Value': p} for t, r, p in zip(self.time[-len(preds):], trues, preds)]
            # 将字典列表转换为DataFrame
            df = pd.DataFrame(data_rows)
            # 将DataFrame保存到CSV文件
            df.to_csv(self.folder_path + self.args.model + '/' + dict[k] + 'senior1.csv', index=False)

            data_rows = [{'time': t, 'Real': r, 'Predicted Value': p} for t, r, p in zip(self.time[-len(preds):], trues_t, preds_t)]
            # 将字典列表转换为DataFrame
            df = pd.DataFrame(data_rows)
            #将DataFrame保存到CSV文件
            df.to_csv(self.folder_path + self.args.model +  '/' + dict[k] + 'ceiling.csv', index=False)

            # # # print('==========  mse:{}, mae:{}'.format(mse, mae))

            f = open(self.folder_path + self.args.model + "/result.txt", 'a')
            # # door
            #f.write("50个时间步长--------------------" + "  \n")
            f.write(dict[j] + "s1_temperature>>>>>>>>>>>>>>>>>>>>>>." + "  \n")
            f.write('mse:{}, mae:{}, rmse:{},mape:{},mspe:{},rse:{}, corr:{}'.format(mse, mae,rmse, mape, mspe, rse, corr))
            f.write('\n')
            f.write('\n')
            f.write(dict[j] + "ceiling temperature>>>>>>>>>>>>>>>>>>>>>>." + "  \n")
            f.write('mse:{}, mae:{}, rmse:{},mape:{},mspe:{},rse:{}, corr:{}'.format(mse_t, mae_t,rmse_t, mape_t, mspe_t, rse_t, corr_t))
            f.write('\n')
            f.write('\n')
            f.close()

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds)
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)

        return
