from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from torchvision.transforms import Compose
import transforms as T
import json
import sys

class S_dataset(Dataset):
    def __init__(self, seis_folder, val_folder, transforms_seis, transforms_vel, sample_ratio = 1, file_size = 500, pre_load=None, fault_fam = False):
        self.seis_folder = seis_folder
        self.vel_folder = val_folder
        self.sample_ratio = sample_ratio
        self.fault_fam = fault_fam
        self.files = self.load_file_name()
        self.transform_seis = transforms_seis
        self.transform_vel = transforms_vel
        self.preload = pre_load
        self.file_size = file_size
        if self.preload:
            self.data_list, self.label_list = [], []
            for file in os.listdir(self.seis_folder):
                file_path = os.path.join(self.seis_folder, file)
                if os.path.isfile(file_path):
                    data, label = self.load_every(self.seis_folder, self.vel_folder, file)
                    self.data_list.append(data)
                    self.label_list.append(label)



    def load_every(self, seis_path, val_path, file_name):
        if self.fault_fam:
            val_data_name = file_name.replace('seis', 'vel')
        else:
            val_data_name = file_name.replace('data', 'model')

        data_path = os.path.join(seis_path, file_name)
        vel_path = os.path.join(val_path, val_data_name)
        print(data_path, vel_path)

        seis = np.load(data_path)[:, :, ::self.sample_ratio, :]
        seis = seis.astype('float32')

        if os.path.isfile(vel_path):
            vel = np.load(vel_path)
            vel = vel.astype('float32')
        else:
            vel = None

        return seis, vel


    def load_file_name(self):
        file_name = []
        for file in os.listdir(self.seis_folder):
            file_path = os.path.join(self.seis_folder, file)
            if os.path.isfile(file_path):
                file_name.append(file)
            else:
                continue

        return file_name


    def __len__(self):
        return len(self.files) * self.file_size

    def __getitem__(self, idx):
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
        file_name = self.files[batch_idx]

        if self.preload:
            data = self.data_list[batch_idx][sample_idx]
            label = self.label_list[batch_idx][sample_idx] if len(self.label_list) != 0 else None
        else:
            data, label = self.load_every(self.seis_folder, self.vel_folder, file_name)
            data = data[sample_idx]
            label = label[sample_idx] if label is not None else None
        if self.transform_seis:
            data = self.transform_seis(data)
        if self.transform_vel and label is not None:
            label = self.transform_vel(label)
        return data, label if label is not None else np.array([])







