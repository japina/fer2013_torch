import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
import pandas as pd
import numpy as np

# n  = 2
# print (f.iloc[n,0]) #emotion
# print (f.iloc[n,1]) #image data
# print (f.iloc[n,2]) #type

class fer2013LoadData():
    def __init__(self, csv_file):
        self.f = pd.read_csv(csv_file)

    def read_data(self):
        train_data = []
        private_data = []
        test_data = []     

        data = self.f.to_numpy()
        for i in data:
            if i[2]=='Training':
                train_data.append([i[1], i[0]])
            if i[2]=='PrivateTest':
                private_data.append([i[1], i[0]])

        #dirty hack - not sure what test_data is
        test_data = private_data
        return train_data, private_data, test_data



class fer2013Dataset(Dataset):
    """fer2013 dataset."""

    def __init__(self, in_data, transform=None):
        """
        Arguments:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.in_data = in_data
        self.transform = transform

    def __len__(self):
        return len(self.f)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        emotion = self.in_data[idx][1]
        img_data = self.in_data[idx][0]
        sample = {'emotion': emotion, 'image': img_data}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, emotion = sample['image'], sample['emotion']

        #7 possible emotions
        return {'image': torch.from_numpy(np.fromstring(image, sep=' ', dtype=np.float32)),
               'emotion': torch.from_numpy(np.array(emotion))}