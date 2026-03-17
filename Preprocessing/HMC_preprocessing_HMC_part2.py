import json
import os
import pickle
from natsort import natsorted
import numpy as np


data_folder       = "......./Data/HMC_processed_pkl_256Hz"
save_folder_train = '......./Data/HMC_cross_json/train.json'
save_folder_test  = '......./Data/HMC_cross_json/test.json'
save_folder_val   = '......./Data/HMC_cross_json/val.json'

# Basic dataset information
sampling_rate = 256
ch_names = ["F4", "C4", "O2", "C3"]
num_channels = len(ch_names)
total_mean = np.zeros(num_channels)
total_std = np.zeros(num_channels)
num_all = 0
max_value = -1
min_value = 1e6

tuples_list_train = []
tuples_list_test = []
tuples_list_val = []
error_list = []

all_pkl_files = [f for f in os.listdir(data_folder)]
all_subjects  = [f.split("_")[0] for f in all_pkl_files if f.endswith('.pkl')]
all_subjects  = list(set(all_subjects))
all_subjects  = natsorted(all_subjects)
print(all_subjects)
print("-------------------\n\n")

#subject_folder = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]
#subject_folder = natsorted(subject_folder)

# Split subject_folder into training, validation, and test sets
train_subjects = all_subjects[:100]
val_subjects = all_subjects[100:100 + 25]
test_subjects = all_subjects[100 + 25:]
print(train_subjects)
print("-------------------")
print(val_subjects)
print("-------------------")
print(test_subjects)
print("-------------------\n\n\n")



# Process each folder and assign data to training, validation, and test sets
for subject_id, subject_name in enumerate(all_subjects):
    print("Processing ", subject_id, '/', len(all_subjects) - 1)
    pkl_files = [os.path.join(data_folder, f) for f in all_pkl_files if subject_name in f]
    pkl_files = natsorted(pkl_files)
    #print(pkl_files)
    #print("-------------------")

    for pkl_id, pkl_file in enumerate(pkl_files):
        try:
            eeg_data = pickle.load(open(pkl_file, "rb"))
            eeg = eeg_data['X']
            label = eeg_data['Y']
        except Exception as e:
            print(f"Error loading file {pkl_file}: {e}")
            error_list.append(pkl_file)
            continue  # Skip this file

        data = {
            "subject_id": subject_name.split("_")[-1], #int(per_folder.split("/")[-1]),
            "subject_name": subject_name,
            "file": pkl_file,
            "label": label
        }

        if subject_name in train_subjects:
            per_max_value = max(eeg.reshape(-1))
            if per_max_value > max_value:
                max_value = per_max_value
            per_min_value = min(eeg.reshape(-1))
            if per_min_value < min_value:
                min_value = per_min_value
            for j in range(num_channels):
                total_mean[j] += eeg[j].mean()
                total_std[j] += eeg[j].std()
            num_all += 1
            tuples_list_train.append(data)
        elif subject_name in val_subjects:
            tuples_list_val.append(data)
        elif subject_name in test_subjects:
            tuples_list_test.append(data)

# Compute mean and standard deviation (based on training set)
data_mean = (total_mean / num_all).tolist()
data_std = (total_std / num_all).tolist()

# Create dataset dictionaries
train_dataset = {
    "subject_data": tuples_list_train,
    "dataset_info": {
        "sampling_rate": sampling_rate,
        "ch_names": ch_names,
        "min": min_value,
        "max": max_value,
        "mean": data_mean,
        "std": data_std
    }
}

test_dataset = {
    "subject_data": tuples_list_test,
    "dataset_info": {
        "sampling_rate": sampling_rate,
        "ch_names": ch_names,
        "min": min_value,
        "max": max_value,
        "mean": data_mean,
        "std": data_std
    }
}

val_dataset = {
    "subject_data": tuples_list_val,
    "dataset_info": {
        "sampling_rate": sampling_rate,
        "ch_names": ch_names,
        "min": min_value,
        "max": max_value,
        "mean": data_mean,
        "std": data_std
    }
}

# Save datasets as JSON files
formatted_json_train = json.dumps(train_dataset, indent=2)
with open(save_folder_train, 'w') as f:
    f.write(formatted_json_train)

formatted_json_test = json.dumps(test_dataset, indent=2)
with open(save_folder_test, 'w') as f:
    f.write(formatted_json_test)

formatted_json_val = json.dumps(val_dataset, indent=2)
with open(save_folder_val, 'w') as f:
    f.write(formatted_json_val)

print("List of error files: ", error_list)




