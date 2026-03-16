import torch
from torch.utils.data import Dataset
import os
import numpy as np
import lmdb
import pickle
import json

from collections import Counter

import random

import pickle

import h5py

from scipy import signal


def zscore(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-6)




class HMC_PickleDataset(Dataset):
    def __init__(self, directory, segment_Len_secs, sampling_Freq, num_classes, seed, mode):

        self.files = json.load(open(directory, "r"))
        self.old_sr = self.files['dataset_info']['sampling_rate']  
        self.channel_name = self.files['dataset_info']['ch_names']  
        self.data = self.files['subject_data'] 
        self.new_sr = sampling_Freq  
        self.ts = segment_Len_secs  

        start_i = 0.1
        stop_i  = 1.1
        self.times_array = np.linspace(start=start_i,stop=stop_i,num=self.ts*self.new_sr)

        rng = random.Random(seed)
        rng.shuffle(self.data)
        #self.data = self.data[:350]

    def __len__(self):
        return len(self.data)

    def get_ch_names(self):
        return self.channel_name

    def normalize(self, X):
        X = X / 100
        return X

    def resample_data(self, data):
        if self.old_sr == self.new_sr:
            return data
        else:
            number_of_samples = int(data.shape[-1] * self.new_sr / self.old_sr)
            return signal.resample(data, number_of_samples, axis=-1)
        

    def __getitem__(self, idx):
        trial = self.data[idx]
        file_path = trial['file']
        sample = pickle.load(open(file_path, "rb"))
        X = sample["X"]
        X = self.resample_data(X)
        X = self.normalize(X)
        Y = int(sample["Y"])

        return torch.tensor(X, dtype=torch.float32), torch.from_numpy(self.times_array), torch.from_numpy(np.asarray(Y))






class TUAB_PickleDataset(Dataset):
    def __init__(self, directory, segment_Len_secs, sampling_Freq, num_classes, seed, mode):

        self.files = sorted([
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".pkl")
        ])

        rng = random.Random(seed)
        rng.shuffle(self.files)

        if mode == "train":
            self.files = self.files[:50000]
        elif mode == "val":
            self.files = self.files[:50000]
       


        start_i = 0.1
        stop_i  = 1.1
        self.times_array = np.linspace(start=start_i,stop=stop_i,num=segment_Len_secs*sampling_Freq)

        #self.files = self.files[:350]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as f:
            sample = pickle.load(f)

        x = sample["X"]/100
        y = sample["y"]  


        return torch.tensor(x, dtype=torch.float32), torch.from_numpy(self.times_array), torch.from_numpy(np.asarray(y))




class SIENA_PickleDataset(Dataset):
    def __init__(self, directory, segment_Len_secs, sampling_Freq, num_classes, seed, mode):

        self.files = json.load(open(directory, "r"))
        self.old_sr = self.files['dataset_info']['sampling_rate']  
        self.channel_name = self.files['dataset_info']['ch_names']  
        self.data = self.files['subject_data'] 
        self.new_sr = sampling_Freq  
        self.ts = segment_Len_secs  

        start_i = 0.1
        stop_i  = 1.1
        self.times_array = np.linspace(start=start_i,stop=stop_i,num=self.ts*self.new_sr)

        rng = random.Random(seed)
        rng.shuffle(self.data)
        #self.data = self.data[:350]

        #"""
        if num_classes > 1 and mode == "train":
            dr = directory.split("train.json")[0]
            with open(os.path.join(dr, "training_labels.pkl"), "rb") as f:
                list_all_labels = pickle.load(f)
            counts = Counter(list_all_labels)
            print("\n**********************")
            for value, count in counts.items():
                print(value, count)
            print("**********************\n")
        #"""

    def __len__(self):
        return len(self.data)

    def get_ch_names(self):
        return self.channel_name

    def normalize(self, X):
        X = X * 10000
        return X

    def resample_data(self, data):
        if self.old_sr == self.new_sr:
            return data
        else:
            number_of_samples = int(data.shape[-1] * self.new_sr / self.old_sr)
            return signal.resample(data, number_of_samples, axis=-1)
        

    def __getitem__(self, idx):
        trial = self.data[idx]
        file_path = trial['file']
        sample = pickle.load(open(file_path, "rb"))
        X = sample["X"]
        X = self.resample_data(X)
        X = self.normalize(X)
        Y = int(sample["Y"])

        return torch.tensor(X, dtype=torch.float32), torch.from_numpy(self.times_array), torch.from_numpy(np.asarray(Y))





class PharmacoEEG_Dataset(Dataset):
    def __init__(self, directory, segment_Len_secs, sampling_Freq, num_classes, seed, mode):

        self.files = sorted([
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".pkl")
        ])


        if mode == "train":
            self.files = [f for f in self.files if f.split("_")[-3] in ['303','304','340','518','587','591']]
        elif mode == "val":
            self.files = [f for f in self.files if f.split("_")[-3] in ['336','574']]
        elif mode == "test":
            self.files = [f for f in self.files if f.split("_")[-3] in ['305','530']]

        
        rng = random.Random(seed)
        rng.shuffle(self.files)


        self.label_corresp = {"Sal":0,"SalT":0,"Amph":1,"CLZ1":2,"CLZ3":2,"SCH":3,"Raclo":4}

        start_i = 0.1
        stop_i  = 1.1
        self.times_array = np.linspace(start=start_i,stop=stop_i,num=segment_Len_secs*sampling_Freq)

        #self.files = self.files[:200]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as f:
            sample = pickle.load(f)

        x = sample["X"]/100 
        y = sample["y"]  
        y = self.label_corresp[y]


        return torch.tensor(x, dtype=torch.float32), torch.from_numpy(self.times_array), torch.from_numpy(np.asarray(y))




class CustomDataset(Dataset):
    def __init__(self, directory, segment_Len_secs, sampling_Freq, num_classes, seed, mode):
        super(CustomDataset, self).__init__()
        
        self.db = lmdb.open(directory, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]
        
        rng = random.Random(seed)
        rng.shuffle(self.keys)

        #"""
        if num_classes > 1 and mode == "train":
            with open(os.path.join(directory, "training_labels.pkl"), "rb") as f:
                list_all_labels = pickle.load(f)
            counts = Counter(list_all_labels)
            print("\n**********************")
            for value, count in counts.items():
                print(value, count)
            print("**********************\n")
        #"""


        start_i = 0.1
        stop_i  = 1.1
        self.times_array = np.linspace(start=start_i,stop=stop_i,num=segment_Len_secs*sampling_Freq)

        #self.keys = self.keys[:200]

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))

        data = pair['sample']/100
        C,L,patch = data.shape
        data = data.reshape(C,L*patch)
        label = pair['label']

        #return data, label
        return torch.tensor(data, dtype=torch.float32), torch.from_numpy(self.times_array), torch.from_numpy(np.asarray(label))









def data_generator_np(training_files_path, validation_files_path, test_files_path, segment_Len_secs, sampling_Freq, seed, num_classes, dataset_name):

    if dataset_name == "TUAB": 
        test_dataset = TUAB_PickleDataset(test_files_path, segment_Len_secs, sampling_Freq, num_classes, seed, mode='test')
        val_dataset = TUAB_PickleDataset(validation_files_path, segment_Len_secs, sampling_Freq, num_classes, seed, mode='val')
        train_dataset = TUAB_PickleDataset(training_files_path, segment_Len_secs, sampling_Freq, num_classes, seed, mode='train')
    elif dataset_name == "DA_Pharmaco": 
        test_dataset = PharmacoEEG_Dataset(test_files_path, segment_Len_secs, sampling_Freq, num_classes, seed, mode='test')
        val_dataset = PharmacoEEG_Dataset(validation_files_path, segment_Len_secs, sampling_Freq, num_classes, seed, mode='val')
        train_dataset = PharmacoEEG_Dataset(training_files_path, segment_Len_secs, sampling_Freq, num_classes, seed, mode='train')
    elif dataset_name == "HMC": 
        test_dataset = HMC_PickleDataset(test_files_path + '/test.json', segment_Len_secs, sampling_Freq, num_classes, seed, mode='test')
        val_dataset = HMC_PickleDataset(validation_files_path + '/val.json', segment_Len_secs, sampling_Freq, num_classes, seed, mode='val')
        train_dataset = HMC_PickleDataset(training_files_path + '/train.json', segment_Len_secs, sampling_Freq, num_classes, seed, mode='train')
    elif dataset_name == "SIENA": 
        test_dataset = SIENA_PickleDataset(test_files_path + '/test.json', segment_Len_secs, sampling_Freq, num_classes, seed, mode='test')
        val_dataset = SIENA_PickleDataset(validation_files_path + '/val.json', segment_Len_secs, sampling_Freq, num_classes, seed, mode='val')
        train_dataset = SIENA_PickleDataset(training_files_path + '/train.json', segment_Len_secs, sampling_Freq, num_classes, seed, mode='train')
    else:
        test_dataset = CustomDataset(test_files_path, segment_Len_secs, sampling_Freq, num_classes, seed, mode='test')
        val_dataset = CustomDataset(validation_files_path, segment_Len_secs, sampling_Freq, num_classes, seed, mode='val')
        train_dataset = CustomDataset(training_files_path, segment_Len_secs, sampling_Freq, num_classes, seed, mode='train')
    
    print("\n\n")
    print("len(train_dataset), len(val_dataset), len(test_dataset)")
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    print("\n\n")

    return train_dataset, val_dataset, test_dataset
















class MACO_Dataset(Dataset):
    
    def __init__(self, signals_path, segment_Len_Secs, original_sampling_Freq, desired_sampling_Freq, nb_structures, df_files):
        super().__init__()
        self.signals_path = signals_path
        self.df_files = df_files
        self.original_sampling_Freq = original_sampling_Freq
        self.desired_sampling_Freq = desired_sampling_Freq
        self.segment_Len_Dpts = int(segment_Len_Secs*self.desired_sampling_Freq)
        self.nb_structures = nb_structures
        self.list_of_files_corrupted = []


        
    def __len__(self):
        return self.df_files.shape[0] 
    
    
    def __getitem__(self, index):
        current_row = self.df_files.iloc[index,:]
        bloc_index = int(current_row.iloc[-1]) 

        molecule_i = current_row.iloc[-4]
        dose_i = current_row.iloc[-3]
        file_name_i = current_row.iloc[0]


        nb_Baseline_blocs = int(current_row.iloc[0].split("_")[-1].split(".")[0])
        list_amplit_arrays, list_psd_arrays, list_phase_arrays = [],[],[]


        for struct_i in range(self.nb_structures):
            try:
                with h5py.File(self.signals_path+current_row.iloc[struct_i], 'r') as hdf5_file:
                    #================= Raw EEG =====================================
                    amplits_array = hdf5_file['dataset'][nb_Baseline_blocs+bloc_index:nb_Baseline_blocs+bloc_index+1]
                    amplits_array = signal.resample(amplits_array.reshape(amplits_array.size), int((amplits_array.size*self.desired_sampling_Freq)/self.original_sampling_Freq))
                    nb_rows = int(amplits_array.size/self.segment_Len_Dpts)
                    amplits_array = amplits_array.reshape(nb_rows,self.segment_Len_Dpts)
                    list_amplit_arrays.append(amplits_array)
                    #================= Raw EEG =====================================

            except:
                print(f"\n!!!!! {current_row.iloc[struct_i]} File Corrupted !!!!!!!!\n")
        
        # Normalization ------------------------------------------------------
        list_amplit_arrays = [(sig - np.mean(sig, axis=1, keepdims=True)) /
                                  (np.std(sig, axis=1, keepdims=True) + 1e-6)
                                  for sig in list_amplit_arrays]
        
        # Absolute POSITIONAL ENCODINGS --------------------------------
        start_i = 1 + bloc_index * 100
        stop_i  = start_i + 99  
        times_array = np.linspace(start=start_i,stop=stop_i,num=int(amplits_array.size))
        times_array = times_array.reshape(nb_rows,self.segment_Len_Dpts)

        # LABELS ------------------------------------------------------
        file_label = int(current_row.iloc[-2])
        labels     = [file_label]*nb_rows


        list_amplit_arrays = [torch.tensor(x, dtype=torch.float32).unsqueeze(1) for x in list_amplit_arrays]
        #print(list_amplit_arrays[0].shape)
        data = torch.cat(list_amplit_arrays, dim=1)
        #print(data.shape)
        #print("--------")


        return data, torch.from_numpy(times_array), torch.tensor(labels)



def data_generator_maco(signals_path, segment_Len_Dpts, original_sampling_Freq, desired_sampling_Freq, nb_structures, df_training, df_validation, df_test):

    test_dataset = MACO_Dataset(signals_path, segment_Len_Dpts, original_sampling_Freq, desired_sampling_Freq, nb_structures, df_test)

    val_dataset = MACO_Dataset(signals_path, segment_Len_Dpts, original_sampling_Freq, desired_sampling_Freq, nb_structures, df_validation)

    train_dataset = MACO_Dataset(signals_path, segment_Len_Dpts, original_sampling_Freq, desired_sampling_Freq, nb_structures, df_training)

    print("\n\n")
    print("len(train_dataset), len(val_dataset), len(test_dataset)")
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    print("\n\n")

    return train_dataset, val_dataset, test_dataset



















