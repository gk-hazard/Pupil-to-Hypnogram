import itertools
import os 
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import random


import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optimizers
import torchvision

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from tqdm.notebook import tqdm
from typing import List


def cal_datanum(data_info: pd.DataFrame, length_num: int, delay_num: int, **kwargs: dict) ->pd.DataFrame:
    '''
    calculate number of data and cumulative number of data 
    
    Parameters
    ----------
    data_info: pandas.DataFrame which includes data of animals, date 
    length: Length of data to be input in training (in /0.01s)
    delay: How far into the future does the model predict sleep stages
    
    Returns
    -------
    data_info: pandas.DataFrame
        includes data of animals, date, length,　number of data, cumulative number of data, Cumulative number of data up to n-1
    '''
    
    data_info['datanum'] = data_info['length'].map(lambda x: ((x-delay_num)//length_num)-1)#Just to be safe,　leave space
    data_info['cum_datanum'] = data_info['datanum'].cumsum()
    data_info['start_num'] = data_info['cum_datanum'] - data_info['datanum']
    
    return data_info


def DataProc(data_info: pd.DataFrame, length_num: int, delay_num: int, filename_list: List, data_path: str, input_dir: str, hypnogram_path:str, coordinate_feature:List) -> List:
    '''
    process data with time series length

    parameters
    --------------
    data_info: pandas.DataFrame which includes data of animals, date, length,　number of data, cumulative number of data 
    length: Length of data to be input in training (in /0.01s)
    delay: How far into the future does the model predict sleep stages

    Returns
    ---------
    data_seq: List containing 2 numpy arrays
    '''
    data_seqp = []
    data_seqh = []
    for i, row in tqdm(data_info.iterrows(), total=data_info.shape[0]):
        temp_p = []
        for v in filename_list:
          temp = pd.read_csv(f'{data_path}/{row[0]}/{row[1]}/{input_dir}/{v}', header=None)
          if "coordinate" in v:
            temp = temp.iloc[:, [idx for idx, j in enumerate(coordinate_feature) if j==1]]
          temp_p.append(temp)
        temp_p = pd.concat(temp_p, axis=1)
        temp_h = pd.read_csv(f'{data_path}/{row[0]}/{row[1]}/{hypnogram_path}')
        temp_pp = [temp_p.iloc[j * length_num : (j + 1) * length_num, :] for j in range(row['datanum'])]
        temp_pp = np.stack(temp_pp)
        temp_hh = [temp_h.iloc[(k + 1) * length_num + delay_num - 1] for k in range(row['datanum'])]
        temp_hh2 = np.stack(temp_hh)
        temp_hh2 = np.reshape(temp_hh2, -1)
        data_seqp.append(temp_pp)
        data_seqh.append(temp_hh2)
    data_seqp = np.concatenate(data_seqp, 0)
    data_seqh = np.concatenate(data_seqh, 0)
    data_seq = [data_seqp, data_seqh]
    
    return data_seq


class PupilDatasetK(data.Dataset):
    '''
    Dataset class for pupil dynamics 
    
    Attributes
    ----------
    data_info: pandas.DataFrame which includes data of animals, date, length,　number of data, cumulative number of data 
    length: Length of data to be input in training (in /0.01s)
    delay: How far into the future does the model predict sleep stages
    
    Method
    ------
    __len__(): -> int
    
    '''
    
    def __init__(self, data_info: pd.DataFrame , length_num: int, delay_num: int, random_seed: int, under_sam: bool, 
                 n_class: int, phase: str, filename_list: List, data_path: str, input_dir:str, hypnogram_path:str, coordinate_feature, **kwargs: dict) -> None:
        self.data_info = data_info
        self.length = length_num
        self.delay = delay_num
        self.data_seq = DataProc(data_info, length_num, delay_num, filename_list, data_path, input_dir, hypnogram_path, coordinate_feature)
        self.under_sam = under_sam
        self.random_seed = random_seed
        self.n_class = n_class
        self.phase = phase
        self.data = self.__undersum__()
        
    def __len__(self):
        '''
        Return the length of the data set.
        
        Returns
        -------
        total_length: int
        '''
        if self.phase == 'train' and self.under_sam:
            idx = [np.where(self.data_seq[1][:] == 0)[0], np.where(self.data_seq[1][:] == 1)[0], np.where(self.data_seq[1][:] == 2)[0]]
            size = np.array([len(np.array(idx[0])), len(np.array(idx[1])), len(np.array(idx[2]))])
            total_length = size.min() * self.n_class
        else:
            total_length = int(self.data_info['cum_datanum'][-1:])
        return total_length

    def __undersum__(self) -> tuple:
        if self.phase == 'train' and self.under_sam:
            idx = [np.where(self.data_seq[1][:] == 0)[0], np.where(self.data_seq[1][:] == 1)[0], np.where(self.data_seq[1][:] == 2)[0]]
            size = np.array([len(np.array(idx[0])), len(np.array(idx[1])), len(np.array(idx[2]))])
            idx_proc = []
            for i in idx:
                random.seed(self.random_seed)
                idx_proc.append(random.sample(i.tolist(), size.min()))
            idx_proc = sum(idx_proc, [])
            self.data_seq[0] = self.data_seq[0][idx_proc]
            self.data_seq[1] = self.data_seq[1][idx_proc]
        return self.data_seq
        
    def __getitem__(self, index: int) -> tuple:
        '''
        Return pupil dynamics and corresponding sleep state
        
        Parameters
        ----------
        index: int
            Index corresponding to the pupil dynamics to be acquired
        
        Returns
        -------
        pupil, label: torch.tensor, torch.tensor
            Pupil Dynamics(length*5) and a label(scholar) corresponding to the given index
        
        '''
        

        pupil = torch.tensor(self.data[0][index])
        label = torch.tensor(self.data[1][index])
        return pupil, label


class PupilDataset(data.Dataset):
    '''
    Dataset class for pupil dynamics 
    作動するかわからん
    Attributes
    ----------
    data_info: pandas.DataFrame which includes data of animals, date, length,　number of data, cumulative number of data 
    length: Length of data to be input in training (in /0.01s)
    delay: How far into the future does the model predict sleep stages
    
    Method
    ------
    __len__(): -> int
    
    '''
    
    def __init__(self, data_info: pd.DataFrame , length_num: int, delay_num: int, data_path, input_dir, hypnogram_path, filename_list, **kwargs: dict) -> int:
        self.data_info = data_info
        self.length = length_num
        self.delay = delay_num
        self.data_path = data_path
        self.input_dir = input_dir
        self.hypnogram_path = hypnogram_path
        self.filename_list = filename_list
        
    def __len__(self):
        '''
        Return the length of the data set.
        
        Returns
        -------
        total_length: int
        '''
        total_length = int(self.data_info['cum_datanum'][-1:])
        return total_length
    
    def __getitem__(self, index: int) -> tuple:
        '''
        Return pupil dynamics and corresponding sleep state
        
        Parameters
        ----------
        index: int
            Index corresponding to the pupil dynamics to be acquired
        
        Returns
        -------
        pupil, label: torch.tensor, torch.tensor
            Pupil Dynamics(length*5) and a label(scholar) corresponding to the given index
        
        '''       

        for i, row in self.data_info.iterrows():
            if index < row['cum_datanum']:
                temp_p = [pd.read_csv(f'{self.data_path}/{row[0]}/{row[1]}/{self.input_dir}/{v}',
                                    header=None) for v in self.filename_list]
                temp_p = pd.concat(temp_p, axis=1)
                temp_h = pd.read_csv(f'{self.data_path}/{row[0]}/{row[1]}/{self.hypnogram_path}')
                id_start = (index - row['start_num']) * self.length  
                pupil = torch.tensor(temp_p.iloc[id_start: (id_start + self.length), : ].values)
                label = torch.tensor(temp_h.iloc[id_start + self.length + self.delay, 0])
                break
        return pupil, label


def cal_batch(loss, preds, t, rec, epoch, cat):
    '''
    Calculate and record the results of the mini-batch training
    '''
    rec[f"{cat}_loss"][epoch] += loss.item()
    rec[f"{cat}_acc"][epoch] += accuracy_score(t.tolist(),preds[:,0:3].argmax(dim=-1).tolist())
    rec[f"{cat}_F1"][epoch] += f1_score(t.tolist(),preds[:,0:3].argmax(dim=-1).tolist(),average='macro')
    ck = cohen_kappa_score(t.tolist(),preds[:,0:3].argmax(dim=-1).tolist())
    rec[f"{cat}_cohenK"][epoch] += 1 if math.isnan(ck)  else ck
    rec[f"{cat}_confusion_matrix"][epoch] += confusion_matrix(t.tolist(),preds[:,0:3].argmax(dim=-1).tolist(), labels=[0, 1, 2])
    _, predicted_t = torch.max(preds[:,0:3], 1)#predsの形確認、各行の一番を持ってくる
    arr_predicted_t = predicted_t.cpu().numpy()
    t = t.cpu().numpy()
    c_t = (arr_predicted_t == t)
    list_c_t = c_t.tolist()
    for v in range(len(t)):
            t = t.astype(int)
            label_t = t[v]
            rec[f"{cat}_class_correct"][epoch][label_t] += list_c_t[v]
            rec[f"{cat}_class_total"][epoch][label_t] += 1
    return rec


def cal_epoch(length, rec, n_class, epoch, cat):
    '''
    Calculate and record the results of the each epoch
    lengthはdataloaderのlenなので注意
    '''
    for g in range(n_class):
        rec[f"{cat}_class_acc"][epoch][int(g)] = 100 * (rec[f"{cat}_class_correct"][epoch][int(g)] / rec[f"{cat}_class_total"][epoch][int(g)]) if rec[f"{cat}_class_total"][epoch][int(g)] != 0 else 0
        rec[f"{cat}_avg_acc"][epoch] = sum(rec[f"{cat}_class_acc"][epoch])/len(rec[f"{cat}_class_acc"][epoch])
    
    rec[f"{cat}_loss"][epoch]  /= length#lossはバッチ毎の平均とっている
    rec[f"{cat}_acc"][epoch] /= length
    rec[f"{cat}_F1"][epoch] /= length
    rec[f"{cat}_cohenK"][epoch] /= length

    return rec

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth', save=False, min_epoch=0):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path
        self.save = save
        self.min_epoch = min_epoch

    def __call__(self, val_loss, model, epoch, optimizer):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model, epoch, optimizer)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if (self.counter >= self.patience) & (epoch+1 >= self.min_epoch):  #設定カウントを上回って、minimum epochも上回ったらストップフラグをTrueに変更
                self.early_stop = True
            if (self.counter >= self.patience) & (epoch+1 < self.min_epoch):
                print(f"epoch{epoch+1} not reached min_epoch{self.min_epoch}")
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model, epoch, optimizer)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model, epoch, optimizer):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            if self.save:
              print('Saving model...')
        self.val_loss_min = val_loss
        if self.save:
          torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': val_loss,
              }, f'{self.path}')

class kfold_iterator():
  def __init__(self, train_info, K_cv, random_seed):
    self.info = train_info
    self.kcv = K_cv
    self.seed = random_seed
    self.data_idx = self._kfold_cal()
    self.current = 0
  def __iter__(self):
    return self
  def __next__(self):
    if self.current == self.kcv:
      raise StopIteration()
    train_idx_list = []
    for id, idx in enumerate(self.data_idx):
      if id == self.current:
        val_idx = idx
      else:
        train_idx_list.append(idx)
    train_idx = np.concatenate(train_idx_list)
    self.current += 1
    return train_idx, val_idx

  def _kfold_cal(self):
    #k個のグループに分けて、それぞれのidxを取得
    n_experience = len(self.info)
    n_idx = np.arange(n_experience)
    np.random.seed(self.seed)
    np.random.shuffle(n_idx)
    k_idx = np.arange(n_experience) % self.kcv
    group_idx = [n_idx[k_idx == i] for i in range(self.kcv)]
    data_idx = [self._concat_idx(self.info.iloc[id, :]) for id in group_idx]
    return data_idx
  
  def _concat_idx(self, df):
    idx_list = [np.arange(row['start_num'], row['cum_datanum']) for  _, row in df.iterrows()]
    idx_arr = np.concatenate(idx_list)
    return idx_arr
