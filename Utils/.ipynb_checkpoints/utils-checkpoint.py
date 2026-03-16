import shutil


# --- Utility: clean pycache
def delete_directory(directory_path):
    try:
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' successfully deleted.")
    except Exception as e:
        print(f"Error deleting directory '{directory_path}': {e}")
delete_directory("__pycache__/")  








import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
import math
import random


from fastprogress import progress_bar


from itertools import cycle


def load_folds_data(config, nb_Structures,class_encoding_dict,signals_path,post_treatment_ranges_path,metadata_DF_path,Class_Molec_Ani_dict_path,Class_Molecule_Dose_dict,Test_Animals_index, seed=41):
    logger = config.get_logger2()
 
    ALL_files = [fi for fi in os.listdir(signals_path) if fi.endswith(".h5")][:2000]                          #  TO DELEEEEETE
    df_all = pd.read_csv(metadata_DF_path, sep=" ")

    signal_files = [(item,df_all["Signal_ID"][b],df_all["Class"][b],df_all["Animal"][b],df_all["Molecule"][b],df_all["Dose"][b]) for b in range(df_all.shape[0]) for item in ALL_files if df_all["Signal_ID"][b] in item]

    df_all = pd.DataFrame(signal_files, columns=["Signal_File", "Signal_ID", "Class", "Animal", "Molecule", "Dose"])
    #np.savetxt(config.save_dir / "df_all.txt", df_all, fmt="%s", delimiter="\t")
    #print("SAAAAAAAAAAAAVED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    # Verification that each signal from Prefrontal has its corresponding Parietal signal
    #nb_Structures = 3                                                                                           # TO DELEEEEETE
    if nb_Structures == 2:
        df_Prefrontal = df_all[df_all["Signal_ID"].str.contains("Prefont")]
        df_Parietal = df_all[df_all["Signal_ID"].str.contains("Par")]

        df_Prefrontal["Signal_ID"] = df_Prefrontal["Signal_ID"].str.replace(r'_Cx.*', '', regex=True)
        df_Parietal["Signal_ID"] = df_Parietal["Signal_ID"].str.replace(r'_Cx.*', '', regex=True)

        df_all = pd.merge(df_Prefrontal, df_Parietal, on=["Signal_ID","Animal", "Class", "Molecule", "Dose"], suffixes=('_Pref', '_Par'), how='inner')

    if nb_Structures == 3:
        df_Prefrontal = df_all[df_all["Signal_ID"].str.contains("Prefont")]
        df_Parietal = df_all[df_all["Signal_ID"].str.contains("Par")]
        df_Hippocampus = df_all[df_all["Signal_ID"].str.contains("Hippo")]

        df_Prefrontal["Signal_ID"] = df_Prefrontal["Signal_ID"].str.replace(r'_Cx.*', '', regex=True)
        df_Parietal["Signal_ID"] = df_Parietal["Signal_ID"].str.replace(r'_Cx.*', '', regex=True)
        df_Hippocampus["Signal_ID"] = df_Hippocampus["Signal_ID"].str.replace(r'_Hippo.*', '', regex=True)

        df_Pref_Par = pd.merge(df_Prefrontal, df_Parietal, on=["Signal_ID","Animal", "Class", "Molecule", "Dose"], suffixes=('_Pref', '_Par'), how='inner')
        df_all = pd.merge(df_Pref_Par, df_Hippocampus, on=["Signal_ID","Animal", "Class", "Molecule", "Dose"], how='inner')
        df_all = df_all.rename(columns={'Signal_File': 'Signal_File_Hippo'})

    #print("******************* STOOOOOOOOOOOOOOOOOOOOOOOOOOOP *******************")



    Class_Molec_Ani_dict_path = Path(Class_Molec_Ani_dict_path)
    with Class_Molec_Ani_dict_path.open("rt") as handle:
        Class_Molec_Ani_dict = json.load(handle, object_hook=OrderedDict)

    df_Post_T_ranges = pd.read_csv(post_treatment_ranges_path, sep=" ")
    df_Post_T_ranges = df_Post_T_ranges.drop_duplicates()
    dict_Post_T_ranges = {row['EEG']: [min(row['Prefrontal_First_Bloc'],row['Parietal_First_Bloc']), max(row['Prefrontal_Last_Bloc'], row['Parietal_Last_Bloc'])] for _, row in df_Post_T_ranges.iterrows()}

    # Code to create the dict_Test_Animals and dict_Train_Animals according to the classes and molecules selected for the experiment
    list_Therapeutic_Class = list(Class_Molecule_Dose_dict.keys())
    dict_Train_Metadata, dict_Test_Metadata = {},{}
    list_dfs_Train_Val, list_dfs_Test = [],[] 
    pairs_ani_molec_test = []  
    for class_i in list_Therapeutic_Class:
        list_molecs_class_i = list(Class_Molecule_Dose_dict[class_i].keys())
        dict_Temp_1, dict_Temp_2 = {},{}
        list_Test_Animals, list_Molec_Dose = [],[]
        for molec_i in list_molecs_class_i:
            list_Molec_Dose.append((molec_i, Class_Molecule_Dose_dict[class_i][molec_i])) # [(molec_A, Dose_i), (molec_B, Dose_j), ...]
            all_animals_Class_i_molec_i = Class_Molec_Ani_dict[class_i][molec_i]
            if Test_Animals_index != "None":
                #test_animals_Class_i_molec_i = [all_animals_Class_i_molec_i[Test_Animals_index[0]], all_animals_Class_i_molec_i[Test_Animals_index[1]]]
                test_animals_Class_i_molec_i = [all_animals_Class_i_molec_i[test_ani_i] for test_ani_i in Test_Animals_index]
            else:
                test_animals_Class_i_molec_i = ["None"]
            #------- Tuples of (molec_i, ani_i) to filter test metadata and infer train_val_metadata
            for test_ani in test_animals_Class_i_molec_i:
                pairs_ani_molec_test.append((molec_i, test_ani))
            #---------------------------------------------------------------------------------------
            train_animals_Class_i_molec_i = [ani for ani in all_animals_Class_i_molec_i if ani not in test_animals_Class_i_molec_i]
            
            list_Test_Animals += test_animals_Class_i_molec_i # Incrementing the list of all test animals related to class_i
            
            dict_Temp_1[molec_i] = test_animals_Class_i_molec_i
            dict_Temp_2[molec_i] = [Class_Molecule_Dose_dict[class_i][molec_i]]+train_animals_Class_i_molec_i # [Dose, Ani_1, Ani_2, ..., Ani_x]
        #*****************************************************************************************


        list_Vehicle_Dose = [("Vehicle_"+p[0][1:],0) if p[0] == "Sertraline" else ("Vehicle_"+p[0],0) for p in list_Molec_Dose]
        #random.seed(seed)
        #random.shuffle(list_Vehicle_Dose)
        list_Selection = list_Molec_Dose + list_Vehicle_Dose

        # Filtering the DataFrame to keep only the rows with the selected Molecules and Doses
        df_Filtered = df_all[df_all[['Molecule', 'Dose']].apply(tuple, axis=1).isin([p for p in list_Selection])]
        df_Filtered["EEG"] = df_Filtered["Signal_ID"].apply(lambda x: x.split("SC")[0][:-3])
        df_Filtered = df_Filtered.sort_values(by=['Class', 'Molecule', 'Dose', 'Animal']).reset_index(drop=True)
        
        expanded_rows = []
        for _, row in df_Filtered.iterrows():
            eeg_name = row['EEG']
            list_start_end = dict_Post_T_ranges[eeg_name]

            for i in range(list_start_end[0], list_start_end[1]+1):
                new_row = row.copy()
                new_row['BlocIndex'] = i
                expanded_rows.append(new_row)

        df_Filtered = pd.DataFrame(expanded_rows).reset_index(drop=True)

        #Adding pairs of (vehic, ani_i)
        pairs_ani_vehic_test = [("Vehicle_"+elt[0][1:],elt[1]) if elt[0] == "Sertraline" else ("Vehicle_"+elt[0],elt[1]) for elt in pairs_ani_molec_test]
        pairs_ani_molec_test = pairs_ani_molec_test + pairs_ani_vehic_test
        random.seed(seed)
        random.shuffle(pairs_ani_molec_test)
        
        #df_Pref_Par_Hippo_Train_Val = df_Filtered[~df_Filtered["Animal"].isin(list_Test_Animals)]
        #df_Pref_Par_Hippo_Test = df_Filtered[df_Filtered["Animal"].isin(list_Test_Animals)]
        df_Pref_Par_Hippo_Train_Val = df_Filtered[~df_Filtered[['Molecule', 'Animal']].apply(tuple, axis=1).isin(pairs_ani_molec_test)]
        df_Pref_Par_Hippo_Test = df_Filtered[df_Filtered[['Molecule', 'Animal']].apply(tuple, axis=1).isin(pairs_ani_molec_test)]
        


        del df_Filtered

        list_dfs_Train_Val.append(df_Pref_Par_Hippo_Train_Val)
        list_dfs_Test.append(df_Pref_Par_Hippo_Test)

        #Dictionary of Test Metadata To Print
        dict_Test_Metadata[class_i]=dict_Temp_1
        #Dictionary of Train Metadata To Print
        dict_Train_Metadata[class_i]=dict_Temp_2
    print("\n --------------- dict_Test_Metadata ----------------------------------------- ")
    logger.info(dict_Test_Metadata)
    print("------------------ dict_Train_Metadata -----------------------------------")
    logger.info(dict_Train_Metadata)
    print("----------------------------------------------------------------------\n\n\n")
    


    if len(list_Therapeutic_Class) > 1:
        df_metadata = pd.concat(list_dfs_Train_Val, ignore_index=True, axis=0)
        df_metadata_Test = pd.concat(list_dfs_Test, ignore_index=True, axis=0)
    else:
        df_metadata = list_dfs_Train_Val[0]
        df_metadata_Test = list_dfs_Test[0]
    del list_dfs_Train_Val, list_dfs_Test, df_Pref_Par_Hippo_Test, df_Pref_Par_Hippo_Train_Val

    df_metadata = df_metadata.sort_values(by=['Class', 'Molecule', 'Dose', 'Animal']).reset_index(drop=True)
    df_Vehicle = df_metadata[df_metadata["Class"].isin(["Vehicle"])]
    df_metadata = df_metadata[~df_metadata["Class"].isin(["Vehicle"])]
    
    max_nb_blocs = max(list(df_metadata["Class"].value_counts())) # The maximum number of blocs among all classes
    df_Vehicle = df_Vehicle.sample(n=max_nb_blocs, random_state=seed)

    df_metadata = pd.concat([df_metadata,df_Vehicle], ignore_index=True, axis=0).reset_index(drop=True)
    del df_Vehicle


    

    print(f"\n\n\n------------ {(df_metadata.shape[0]*10)/60} hours of Recorded  data included --------------\n")
    print(f"------------ NO Synthetic data included --------------\n\n\n")


    df_metadata = df_metadata.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Replacing the class names by their corresponding numbers 
    for class_i in df_metadata["Class"].unique():
        class_i_replacement = class_encoding_dict[class_i]
        df_metadata["Class"] = df_metadata["Class"].replace(class_i, class_i_replacement)
        df_metadata_Test["Class"] = df_metadata_Test["Class"].replace(class_i, class_i_replacement)


    #df_metadata.to_csv("/bettik/PROJECTS/pr-eeg/darankod/PART_3/Exp_MACO_July_25/CoSupFormer_MACO_batchExp4/utils/df_metadata_TrainVal.txt", index=False)
    #df_metadata_Test.to_csv("/bettik/PROJECTS/pr-eeg/darankod/PART_3/Exp_MACO_July_25/CoSupFormer_MACO_batchExp4/utils/df_metadata_Test.txt", index=False)
    df_metadata.to_csv(config.save_dir / "df_metadata_TrainVal.txt", index=False)
    df_metadata_Test.to_csv(config.save_dir / "df_metadata_Test.txt", index=False)

    if nb_Structures == 2:
        df_metadata = df_metadata[["Signal_File_Pref","Signal_File_Par",'Molecule', 'Dose',"Class","BlocIndex"]]
        df_metadata_Test = df_metadata_Test[["Signal_File_Pref","Signal_File_Par",'Molecule', 'Dose',"Class","BlocIndex"]]
    if nb_Structures == 3:
        df_metadata = df_metadata[["Signal_File_Pref","Signal_File_Par","Signal_File_Hippo",'Molecule', 'Dose',"Class","BlocIndex"]]
        df_metadata_Test = df_metadata_Test[["Signal_File_Pref","Signal_File_Par","Signal_File_Hippo",'Molecule', 'Dose',"Class","BlocIndex"]]
    
    print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("Training and Validation :")
    print(df_metadata["Class"].value_counts(normalize=True))
    print("Test :")
    print(df_metadata_Test["Class"].value_counts(normalize=True))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")

    
    nb_blocs_Validation = int(np.ceil(df_metadata.shape[0] * 0.1))

    """
    df_Training = df_metadata[:-nb_blocs_Validation]       
    df_Validation = df_metadata[-nb_blocs_Validation:]
    df_Test = df_metadata_Test
    """
    df_Training = df_metadata[:600]       #[:10]
    df_Validation = df_metadata[600:900]
    df_Test = df_metadata_Test[:200]
    
    

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    logger.info(f' TRAINING FILES ({df_Training.shape[0]} blocs of 10mn * {nb_Structures} nb_Struct)  \n\n')
    logger.info(f' VALIDATION FILES ({df_Validation.shape[0]} blocs of 10mn * {nb_Structures} nb_Struct) \n\n')
    logger.info(f' TEST FILES ({df_Test.shape[0]} blocs of 10mn * {nb_Structures} nb_Struct) \n\n')
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n")
    

    return [df_Training, df_Validation, df_Test]
