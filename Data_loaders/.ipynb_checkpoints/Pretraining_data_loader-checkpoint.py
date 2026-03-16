from scipy import signal
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import random

import matplotlib.pyplot as plt
from scipy.signal import stft, istft

import pickle
import lmdb

from Utils.masking_strategy import stft_masking





class Custom_SSL_Dataset_TUEG(Dataset):
    def __init__(self,
                 signals_path,
                 segment_Len_Secs,
                 original_sampling_Freq,
                 desired_sampling_Freq,
                 nb_structures,
                 masking=True,
                 mask_params_list=None,
                 sample_start=0,
                 sample_end=0,
                 seed=1):

        super().__init__()
        self.signals_path = signals_path
        self.original_sampling_Freq = original_sampling_Freq
        self.desired_sampling_Freq = desired_sampling_Freq
        self.segment_Len_Dpts = int(segment_Len_Secs * desired_sampling_Freq)
        self.nb_structures = nb_structures
        self.masking = masking
        self.mask_params_list = mask_params_list or []
        self.seed = seed

        self._env = None

        env = lmdb.open(signals_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get(b'__keys__'))
        env.close()

        
        self.keys = self.keys[sample_start:sample_end]

        print(f"\n\nTotal Nb samples {len(self.keys)}")
        print(f"sample_start:{sample_start} --- sample_end:{sample_end}\n")

    
    def _get_env(self):
        # One LMDB env per worker process
        if self._env is None:
            self._env = lmdb.open(
                self.signals_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=2048
            )
        return self._env

    
    def __len__(self):
        return len(self.keys)

        
    def __getitem__(self, index):
        file_name_i = self.keys[index]
        bloc_index = int(file_name_i.split("_")[-1])

        # ====== Load EEG signals ======
        env = self._get_env()
        with env.begin(write=False) as txn:
            patch = pickle.loads(txn.get(file_name_i.encode()))

        patch = patch.reshape(patch.shape[0], patch.shape[1] * patch.shape[2])
        #print(f"patch : {patch.shape}")
        
        
        # ====== Positional Encoding ======
        start_i = 1 + bloc_index * 10
        stop_i = start_i + 9
        times_array = np.linspace(start=start_i, stop=stop_i, num=self.segment_Len_Dpts)

        
        # ====== Masking via utility function ======
        if self.masking == "True":
            patch_masked, info = stft_masking(
                patch,
                sr=self.desired_sampling_Freq,
                n_fft=self.mask_params_list[0],
                noverlap=self.mask_params_list[1],
                mask_ratio=self.mask_params_list[2],
                mask_level=self.mask_params_list[3],
                mask_type_ratio=self.mask_params_list[4],
                global_seed=self.seed,
                batch_idx=index,
                band_bias=self.mask_params_list[5],
                band_probs=self.mask_params_list[6],
                visualize=self.mask_params_list[7],
            )
        else:
            patch_masked = patch

            
        # ====== Convert all to tensors ======
        patch_masked = torch.tensor(patch_masked, dtype=torch.float32)
        times_array = torch.tensor(times_array, dtype=torch.float32)
        
        
        return patch_masked, times_array




# ====== Utility wrapper to instantiate datasets ======
def data_generator_np(signals_path, segment_Len_Secs,
                      original_sampling_Freq, desired_sampling_Freq, nb_structures,
                      masking=True, mask_params_list=[],
                      sample_start=0, sample_end=0,
                      seed=1):

    train_dataset = Custom_SSL_Dataset_TUEG(
        signals_path=signals_path,
        segment_Len_Secs=segment_Len_Secs,
        original_sampling_Freq=original_sampling_Freq,
        desired_sampling_Freq=desired_sampling_Freq,
        nb_structures=nb_structures,
        masking=masking,
        sample_start=sample_start,
        sample_end=sample_end,
        mask_params_list=mask_params_list, seed=seed
    )

    return train_dataset




