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

def cal_datanum(data_info: pd.DataFrame, length_num: int, **kwargs: dict) ->pd.DataFrame:
    '''
    calculate number of data and cumulative number of data 
    
    Parameters
    ----------
    data_info: pandas.DataFrame which includes data of animals, date 
    length: Length of data to be input in training (in /0.01s)
    
    Returns
    -------
    data_info: pandas.DataFrame
        includes data of animals, date, length,　number of data, cumulative number of data, Cumulative number of data up to n-1
    '''
    
    data_info['datanum'] = data_info['length'].map(lambda x: (x//length_num)-1)#Just to be safe,　leave space
    data_info['cum_datanum'] = data_info['datanum'].cumsum()
    data_info['start_num'] = data_info['cum_datanum'] - data_info['datanum']
    
    return data_info

def DataProc(data_info: pd.DataFrame, length_num: int, filename_list: List, Path_default: str, input_dir: str, coordinate_feature:List) -> List:
    '''
    process data with time series length

    parameters
    --------------
    data_info: pandas.DataFrame which includes data of animals, date, length,　number of data, cumulative number of data 
    
    Returns
    ---------
    data_seq: List containing 1 numpy arrays
    '''
    data_seqp = []
    for i, row in tqdm(data_info.iterrows(), total=data_info.shape[0]):
        temp_p = []
        for v in filename_list:
          temp = pd.read_csv(f'{Path_default}/{input_dir}/{v}', header=None)
          if "coordinate" in v:
            temp = temp.iloc[:, [idx for idx, j in enumerate(coordinate_feature) if j==1]]
          temp_p.append(temp)
        temp_p = pd.concat(temp_p, axis=1)
        temp_pp = [temp_p.iloc[j * length_num : (j + 1) * length_num, :] for j in range(row['datanum'])]
        temp_pp = np.stack(temp_pp)
        data_seqp.append(temp_pp)
    data_seqp = np.concatenate(data_seqp, 0)
    data_seq = [data_seqp]
    
    return data_seq

class P2HDataset(data.Dataset):
  '''
  Dataset class for pupil dynamics 
  
  Attributes
  ----------
  data_info: pandas.DataFrame which includes data of animals, date, length,　number of data, cumulative number of data 
  
  Method
  ------
  __len__(): -> int
  
  '''
  def __init__(self, data_info: pd.DataFrame , length_num: int, random_seed: int, Path_default:str,
                n_class: int, phase: str, filename_list: List, input_dir:str, coordinate_feature, **kwargs: dict) -> None:
      self.data_info = data_info
      self.data = DataProc(data_info, length_num, filename_list, Path_default, input_dir, coordinate_feature)
      self.random_seed = random_seed
      self.n_class = n_class
      self.phase = phase
      
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
      pupil: torch.tensor
          Pupil Dynamics(length*5) corresponding to the given index
      
      '''
      

      pupil = torch.tensor(self.data[0][index])
      return pupil

