import os
import re
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset

import h5py
import numpy as np
import pandas as pd

class MultiLeadECGDataset(Dataset):
    def __init__(self, data_path, train):
        super(MultiLeadECGDataset, self).__init__()

        self.train = train
        self.data_path = data_path
        self.hdf5_file_list = []

        print('LOAD DATA...')
        if train:
            self.df = pd.read_csv(os.path.join(self.data_path, 'train', 'exams.csv'), index_col=False)

            for part in range(18):
                f = h5py.File(os.path.join(self.data_path, 'train', 'exams_part{}.hdf5'.format(part)), 'r')
                self.hdf5_file_list.append(f)
        else:
            self.df = pd.read_csv(os.path.join(self.data_path, 'test', 'annotations', 'gold_standard.csv'), index_col=False)
            self.hdf5_file = h5py.File(os.path.join(self.data_path, 'test', 'ecg_tracings.hdf5'), 'r')
        print('COMPLETE!!!')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.train:
            exam_id, target, trace_file = self.df.iloc[idx]['exam_id'], self.df.iloc[idx][['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']], self.df.iloc[idx]['trace_file']
            trace_file_idx = int(re.sub(r'[^0-9]', '', trace_file.split('.')[0]))

            hdf5_file = self.hdf5_file_list[trace_file_idx]
            data = np.array(hdf5_file['tracings'][np.where(np.array(hdf5_file['exam_id']) == exam_id)[0][0]])
        else:
            target = self.df.iloc[idx][['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']]
            data = np.array(self.hdf5_file['tracings'])[idx]

        data = torch.from_numpy(np.transpose(data, (1, 0))).float()
        target = torch.from_numpy(np.array(target, dtype=np.int32)).float()

        return data, target