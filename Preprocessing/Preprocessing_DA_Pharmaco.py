import h5py
import os
import random
from scipy import signal
from scipy.signal import iirnotch, filtfilt
import pickle
import numpy as np

#from scipy.io import loadmat
#import lmdb



def read_matlab_hdf5_string(ds):
    """Decode MATLAB v7.3 string dataset into a clean Python str."""
    raw = ds[()]

    # Turn raw into a bytes buffer
    if isinstance(raw, bytes):
        b = raw
    elif isinstance(raw, np.ndarray):
        b = raw.tobytes()
    else:
        b = bytes(raw)

    # MATLAB often stores strings as UTF-16LE (null bytes every other byte)
    if b'\x00' in b:
        s = b.decode('utf-16le', errors='ignore')
    else:
        s = b.decode('utf-8', errors='ignore')

    return s.strip('\x00').strip()




#-------------------------------------------------
data_dir = '......./Data/Pharmaco_EEG/RawData'
out_dir = '......./Data/Pharmaco_EEG/processed/'


list_of_files = [file for file in os.listdir(data_dir)]
rng = random.Random(41)
rng.shuffle(list_of_files)

dico_files_meta_infos = {}
for file in list_of_files:
    meta_infos = file.split("_")

    # dico = {"file_name":[animal_num, treatment_received]}
    dico_files_meta_infos[file]=[meta_infos[0],meta_infos[-1].split(".")[0]]

print(dico_files_meta_infos)
print("======================================\n")



dico_electrodes_corresp = {
    "303":['PFCd', 'dHC', 'vHCd', 'MD', 'CA3'],
    "304":['PFCu', 'dHC', 'vHCd', 'MD', 'CA3'],
    "305":['PFCd', 'dHC', 'vHCd', 'MD', 'CA3'],
    "336":['PFCu', 'dHC', 'vHCd', 'MD', 'CA3'],
    "340":['PFCu', 'dHC', 'vHCu', 'MD', 'CA3'],
    "518":['PFCd', 'dHC', 'vHCd', 'MD', 'CA3'],
    "530":['PFCu', 'dHC', 'vHCd', 'MD', 'CA3'],
    "574":['PFCu', 'dHC', 'vHCd', 'MD', 'CA3'],
    "587":['PFCd', 'dHC', 'vHCd', 'MD', 'CA3'],
    "591":['PFCd', 'dHC', 'vHCd', 'MD', 'CA3']
}




def print_selected_frequencies(eeg, fs=200):
    """
    eeg: numpy array [nb_channels, seq_len]
    fs: sampling frequency (default 200 Hz)
    """

    nb_channels, L = eeg.shape

    eeg = eeg - eeg.mean(axis=1, keepdims=True)

    # FFT
    X = np.fft.rfft(eeg, axis=1)
    power = np.abs(X) ** 2

    freqs = np.fft.rfftfreq(L, d=1/fs)

    # Frequencies to inspect
    target_freqs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    print("\nSelected Frequency Powers (per channel)\n")

    for ch in range(nb_channels):
        print(f"Channel {ch}:")
        for f in target_freqs:
            if f <= fs/2:
                idx = np.argmin(np.abs(freqs - f))
                print(f"  {f:>3} Hz : {power[ch, idx]:.4e}")
        print()


for fi in dico_files_meta_infos.keys():
    with h5py.File(os.path.join(data_dir, fi), 'r') as f:

        animal_num = dico_files_meta_infos[fi][0]
        treatment  = dico_files_meta_infos[fi][1]
        channels_to_keep = dico_electrodes_corresp[animal_num]
        print("----------------")
        print(animal_num)
        print(treatment)
        print("---")

        sfreq = f["Ephys"]["srate"][0][0]
        Post_Inj_Start = int(f["Ephys"]["TimeStamps"]["Post_Inj_Start"][0][0])
        raw_eeg = f["Ephys"]["RawData"][:]

        channels_to_keep_ind = []
        chnlist = f["Ephys"]["ChnList"]
        for i, ref in enumerate(chnlist[:, 0]):
            ds = f[ref]  # dereference object
            channel_i = read_matlab_hdf5_string(ds)
            if channel_i in channels_to_keep:
                print(channel_i)
                channels_to_keep_ind.append(i)
        
        # Filter the raw eeg data to keep only 5 channels (remove ref and ground) and collect 40 mn post injection starting at 5mn post-inj
        new_start = int(Post_Inj_Start + 5*60*sfreq)
        new_end   = int(new_start + 40*60*sfreq)
        raw_eeg = raw_eeg[channels_to_keep_ind,new_start:new_end]

        # Resample and filter 50 Hz
        raw_eeg = signal.resample(raw_eeg, 480000, axis=-1) 
        #print("======================================\n")

        b, a = iirnotch(w0=50.0, Q=30.0, fs=200.0)
        raw_eeg = filtfilt(b, a, raw_eeg, axis=-1)   # zero-phase filtering
        #print("======================================\n")

        print(raw_eeg.shape)



        saved_file_prefix = 'A_'+animal_num+"_"+treatment+"_"
        for bloc in range(0,40):
            saved_file_name = saved_file_prefix+str(bloc+1)+".pkl"
            start = bloc * (60*200)
            end   =  start + (60*200)
            #print(f"{start}--{end}")
            #print(raw_eeg[:,start:end].shape[1]/200)

            pickle.dump(
                {"X": raw_eeg[:,start:end], "y": treatment},
                open(os.path.join(out_dir,saved_file_name), "wb"),
            )

        print("----------------\n")








