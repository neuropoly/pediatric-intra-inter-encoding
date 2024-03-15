# tf2 environment
import os
import random 
import glob
import zipfile
import numpy as np
import nibabel as nib
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import re
from tensorflow import keras
from tensorflow.keras import layers

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # No normalization want to keep values from -1 to 1, only converts to float32
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume


# Folder "intra" consists of MRI scans from intra-registration and "inter" from inter-registration pairs.
def read_paths(path_intra, path_inter, experiment_name_intra, experiment_name_inter):
    """
    Retrieve file paths for MRI scans based on the specified experiment_name.

    Parameters:
    - path_intra (str): path to folders of pairs containing the intra logJacobian volumes 
    - path_inter (str): path to folders of pairs containing the inter logJacobian volumes 
    - experiment_name_intra (str): The name of the experiment, determining the specific set of MRI scans to be processed. 
                            Possible values include inter_mix_ra, inter_mix_r, intra_mix_r, intra_mix_ra, 
                            and all combinations from inter_f_ra and inter_m_r.
    - experiment_name_inter (str): Here, 'inter' and 'intra' represent the types of pairs, 'mix' indicates the presence 
                            of both males (m) and females (f), 'r' implies rigid registration, and 'ra' indicates both rigid 
                            and affine registration before.

    Returns:
    - intra_scan_paths (list): A list of file paths for MRI scans with the specified experiment_name and "intra" type.
    - nbr_intra_pairs (int): The number of pairs of MRI scans for the "intra" type.
    - inter_scan_paths (list): A list of file paths for MRI scans with the specified experiment_name and "inter" type.
    - nbr_inter_pairs (int): The number of pairs of MRI scans for the "inter" type.
    """
    
    intra_scan_paths = glob.glob(f"{path_intra}/{experiment_name_intra}/intra/**/logJacobian.nii.gz", recursive=True)

    inter_scan_paths = glob.glob(f"{path_inter}/{experiment_name_inter}/**/logJacobian.nii.gz", recursive=True)

    print("Nbr of MRI scans with intra log Jacobian: " + str(len(intra_scan_paths)))
    print("Nbr of MRI scans with inter log Jacobian: " + str(len(inter_scan_paths)))

    nbr_intra_pairs = len(intra_scan_paths)
    nbr_inter_pairs = len(inter_scan_paths)

    return intra_scan_paths, nbr_intra_pairs, inter_scan_paths, nbr_inter_pairs

def read_tsv_and_create_dict(file_path):
    """
    Reads a TSV (Tab-Separated Values) file containing participant information and creates a dictionary
    mapping participant IDs to their unique identifiers, associated scan IDs, sex, and age.

    Parameters:
    - file_path (str): The path to the TSV file containing participant information.

    Returns:
    - participant_mapping (dict): A dictionary where participant IDs are keys, and the corresponding values
                                  are dictionaries containing 'id' (unique identifier), 'scan_ids' (list of scan IDs),
                                  'sex' (list of participant sexes), and 'age' (list of participant ages).

    Example Usage:
    participant_info = read_tsv_and_create_dict("/path/to/all-participants.tsv")
    # Access information for a participant with ID '10006'
    p001_info = participant_info.get('10006', {})
    print(f"Participant P001 has unique ID: {p001_info.get('id')}, Scan IDs: {p001_info.get('scan_ids')}, "
          f"Sex: {p001_info.get('sex')}, Age: {p001_info.get('age')}")
    """
    participant_mapping = {}
    
    with open(file_path, 'r') as file:
        header = file.readline()  # Skip the header line
        for line in file:
            participant_id, sub_id_bids, scan_id, session, age, sex, group = line.strip().split('\t')
            
            # Convert age to float
            age = float(age)
            
            # Check if the participant_id is already in the mapping, if not, assign it a unique identifier
            if participant_id not in participant_mapping:
                unique_id = len(participant_mapping)
                participant_mapping[participant_id] = {'id': unique_id, 'scan_ids': [], 'sex':[], 'age':[]}
            
            # Append the scan_id to the participant's list
            participant_mapping[participant_id]['scan_ids'].append(scan_id)
            # Add the sex information if it hasn't been added yet
            if len(participant_mapping[participant_id]['sex']) == 0:
                participant_mapping[participant_id]['sex'].append(sex)
            # Append the age
            if len(participant_mapping[participant_id]['age']) == 0:
                participant_mapping[participant_id]['age'].append(age)
    
    return participant_mapping

def read_tsv_and_create_dataframe(file_path):
    """
    Reads a TSV (Tab-Separated Values) file containing participant information and creates a pandas DataFrame
    with columns for participant ID, scan ID, age, and sex.

    Parameters:
    - file_path (str): The path to the TSV file containing participant information.

    Returns:
    - df (pandas.DataFrame): A DataFrame containing participant data with columns 'participant_id', 'scan_id',
                             'age', and 'sex'.

    Example Usage:
    participant_df = read_tsv_and_create_dataframe("/path/to/participant_info.tsv")
    # Access the DataFrame columns and rows as needed
    print(participant_df.head())
    """
    participant_data = []
    
    with open(file_path, 'r') as file:
        header = file.readline()  # Skip the header line
        for line in file:
            participant_id, sub_id_bids, scan_id, session, age, sex, group = line.strip().split('\t')
            
            # Convert age to float
            age = float(age)
            
            # Append data to participant_data list
            participant_data.append([participant_id, scan_id, age, sex])
            
    # Create a DataFrame with the participant data
    columns = ['participant_id', 'scan_id', 'age', 'sex']
    df = pd.DataFrame(participant_data, columns=columns)
    
    return df

def find_scan_ids(name):
    """
    Extracts patient numbers, moving and fixed image scan IDs from a given folder name for further analysis.

    Parameters:
    - name (str): The folder name containing information about patient numbers and scan IDs.

    Returns:
    - tuple: A tuple containing two strings representing the moving and fixed image scan IDs.

    Example Usage:
    folder_name = 'CL_Dev_004_CL_Dev_008'
    scan_ids = find_scan_ids(folder_name)
    print(f"Patient Number: {scan_ids[0]}, Moving Image Scan ID: {scan_ids[1]}")

    Description:
    The function takes a folder name as input, which typically represents a pair of scans in medical imaging.
    It extracts relevant information such as patient numbers, moving, and fixed image scan IDs.
    The folder name may follow various formats, and the function handles different cases, returning a tuple with
    the moving and fixed image scan IDs.

    Possible folder name formats:
    - 'CL_Dev_004_CL_Dev_008'
    - 'CL_Dev_004_PS15_048'
    - 'PS15_048_CL_Dev_004'
    - 'PS15_048_PS17_017'
    - 'PS16_058_PS1183-10-1'

    The function supports various lengths and formats of folder names, providing flexibility for different naming conventions.
    """

    #idx contains a list of strings of a given name
    idx = [s for s in name.split("_")]
    

    if len(idx) == 6:
        
        mov = f'{idx[0]}_{idx[1]}_{idx[2]}'
        fix = f'{idx[3]}_{idx[4]}_{idx[5]}'
        return(mov, fix)

    elif len(idx) == 5:
        if 'CL' in idx[0]:
            mov = f'{idx[0]}_{idx[1]}_{idx[2]}'
            fix = f'{idx[3]}_{idx[4]}'
            
            return(mov, fix)
        elif 'PS' in idx[0]:
            mov = f'{idx[0]}_{idx[1]}'
            fix = f'{idx[2]}_{idx[3]}_{idx[4]}'
            
            return(mov, fix)

    elif len(idx) == 4:
        mov = f'{idx[0]}_{idx[1]}'
        fix = f'{idx[2]}_{idx[3]}'
        return(mov, fix)

    elif len(idx) == 3 and '-' not in idx[2]:
        mov = f'{idx[0]}'
        fix = f'{idx[1]}_{idx[2]}'
        return(mov, fix)

    elif len(idx) == 3 and '-' in idx[2]:
        mov = f'{idx[0]}_{idx[1]}'
        fix = f'{idx[2]}'
        return(mov, fix)

    else:
        print('Not a corresponding folder name', name)

def extract_all_participant_ids(intra_scan_paths, inter_scan_paths, participant_mapping, full_df):
    """
    Extracts participant ID pairs, scan ID pairs, sex information pairs, and age interval pairs from intra and inter scan paths.

    Parameters:
    - intra_scan_paths (list): List of file paths for intra scans.
    - inter_scan_paths (list): List of file paths for inter scans.
    - participant_mapping (dict): Dictionary mapping participant IDs to their information.
    - full_df (pd.DataFrame): DataFrame containing information about scans, including age.

    Returns:
    - list: A list containing tuples representing participant ID pairs extracted from intra and inter scan paths.

    Example Usage:
    intra_paths = [...]
    inter_paths = [...]
    participant_mapping = {...}
    full_data_frame = pd.DataFrame(...)  # DataFrame with scan information
    result = extract_all_participant_ids(intra_paths, inter_paths, participant_mapping, full_data_frame)
    print(f"All Participant ID Pairs: {result}")

    Description:
    The function takes lists of intra and inter scan paths, a participant mapping dictionary, and a DataFrame
    containing scan information. It iterates through the paths, extracts participant and scan IDs, and looks up
    corresponding information such as sex and age from the provided mappings. The resulting participant ID pairs
    are stored in a list and returned.

    Note: Ensure the 'find_scan_ids' function is defined and available in your code.
    """
    all_participant_ids = []
    all_participant_ids_and_scan_ids = []
    all_participant_ids_sex = []
    all_participant_ids_age = []

    # Initialize variables before the loop
    participant_id_1 = participant_id_2 = sex_1 = sex_2 = age_1 = age_2 = None



    all_paths = np.concatenate((intra_scan_paths, inter_scan_paths), axis=0)

    # Iterate over the intra and inter _scan_paths
    for path in all_paths:
        # Extract the scan_id from the path
        scan_ids = path.split('/')[-2]
        scan_id_1, scan_id_2 = find_scan_ids(scan_ids)
        
        # Find participant_ids for scan_id_1 and scan_id_2
        participant_id_1 = None
        participant_id_2 = None

        for participant_id, info in participant_mapping.items():
            if scan_id_1 in info['scan_ids']:
                participant_id_1 = participant_id
                sex_1 = info['sex']
                age_1 = full_df.loc[full_df['scan_id'] == scan_id_1, 'age'].values[0]
            if scan_id_2 in info['scan_ids']:
                participant_id_2 = participant_id
                sex_2 = info['sex']
                age_2 = full_df.loc[full_df['scan_id'] == scan_id_2, 'age'].values[0]
            if age_1 and age_2 != None:
                # Calculate the age interval
                age_interval = age_1 - age_2
            else:
                continue
        
                

        # Create pairs
        if participant_id_1 is not None and participant_id_2 is not None:
            all_participant_ids.append((participant_id_1, participant_id_2))
            all_participant_ids_and_scan_ids.append(((participant_id_1, participant_id_2), (scan_id_1, scan_id_2)))
            all_participant_ids_sex.append(((participant_id_1, participant_id_2), (sex_1, sex_2)))
            all_participant_ids_age.append(((participant_id_1, participant_id_2), (age_interval)))


    print("all_participant_ids:", all_participant_ids)
    print("all_participant_ids_and_scan_ids:", all_participant_ids_and_scan_ids)
    print("all_participant_ids_and_scan_ids:", all_participant_ids_sex)
    print("all_participant_ids_and_scan_ids:", all_participant_ids_age)
    print('Number of pairs in total (both intra and inter):', len(all_participant_ids))

    return all_participant_ids

def shuffle_keys(dictionary, random_seed):
    """
    Shuffles the keys of a given dictionary, providing a new dictionary with keys in a random order.

    Parameters:
    - dictionary (dict): The input dictionary whose keys need to be shuffled.
    - random_seed (int): An integer used to seed the random number generator, ensuring reproducibility.

    Returns:
    - dict: A new dictionary with keys shuffled in a random order.

    Example Usage:
    original_dict = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    seed_value = 42
    shuffled_dict = shuffle_keys(original_dict, seed_value)
    print(f"Original Dictionary: {original_dict}")
    print(f"Shuffled Dictionary: {shuffled_dict}")

    Description:
    The function takes a dictionary and a random seed as inputs. It shuffles the keys of the input dictionary using
    the provided random seed to ensure reproducibility. The result is a new dictionary with the same values as the
    original dictionary but with keys rearranged in a random order. This can be useful for scenarios where the order
    of dictionary keys needs to be randomized for further processing or analysis.
    """
    # Set the random seed for reproducibility
    random.seed(random_seed)
    
    # Extract the keys from the dictionary
    keys = list(dictionary.keys())
    
    # Shuffle the keys
    random.shuffle(keys)
    
    # Create a new dictionary with shuffled keys
    shuffled_dict = {key: dictionary[key] for key in keys}
    
    return shuffled_dict

def custom_split_indices(data, ratio, nbr_sets, random_state=None):
    """
    Custom function to split indices based on participant IDs while considering desired ratios for each set.

    Parameters:
    - data (list): List of tuples representing pairs of participant IDs to be split into sets.
    - ratio (float): The desired ratio for each set.
    - nbr_sets (int): The number of sets to be created.
    - random_state (int): An integer used to seed the random number generator for reproducibility.

    Returns:
    - tuple: A tuple containing two elements - a list of sets representing the split indices (all_sets),
             and a dictionary mapping participant IDs to corresponding indices (indices_by_subject).

    Example Usage:
    data_pairs = [(10006, 10006), (10006, 10007), (10010, 10032), (10064, 10064)]
    result_sets, result_indices = custom_split_indices(data_pairs, ratio=0.2, nbr_sets=3, random_state=42)
    print(f"Resulting Sets: {result_sets}")
    print(f"Indices by Subject: {result_indices}")

    Description:
    The function takes a list of pairs of participant IDs and aims to split these pairs into sets while maintaining
    desired ratios. It uses a custom algorithm that shuffles the order of subjects and assigns each pair of indices
    to the least populated set, avoiding overlap where possible. The function returns a list of sets (all_sets) and a
    dictionary (indices_by_subject) mapping participant IDs to their corresponding indices.

    Note: Ensure the 'shuffle_keys' function is defined and available in your code.
    """

    # Create dictionaries to track subjects by participant_id
    indices_by_subject = {}
    
    # Define a list to hold the sets
    all_sets = [set() for _ in range(nbr_sets)]
    
    
    # Iterate through the data and group indices by participant_id (subject_number)
    for i, (participant_id_1, participant_id_2) in enumerate(data):
        if participant_id_1 not in indices_by_subject:
            indices_by_subject[participant_id_1] = []
        
        if participant_id_2 not in indices_by_subject:
            indices_by_subject[participant_id_2] = []
            
        
        indices_by_subject[participant_id_1].append(i)
        if participant_id_2 != participant_id_1:
            indices_by_subject[participant_id_2].append(i)

    # Shuffle all subject keys in the created dictionnary mapping indices
    if random_state is not None:
        shuffled_indices_by_subject = shuffle_keys(indices_by_subject, random_state)

    wanted_ratios = [ratio] * nbr_sets
    iteration = 0
    # Iterate through the subjects, shuffling the order of subjects for each participant_id (subject)
    for subject, indices in shuffled_indices_by_subject.items():
        # print(f"Subject: {subject}")
        iteration += 1  # Increment iteration count
        subject_overlapping = []  # Overlapping for each subject
        for a_set in all_sets:
            overlapping = bool(a_set.intersection(set(indices)))
            subject_overlapping.append(overlapping)

        # Skip that pair if there's overlapping with at least 2 sets, if there is only 1, then update indices where it overlaps, else if there are no
        # overlaps, choose the less populated set to be updated
        if sum(subject_overlapping) == 1:
            for i, a_set in enumerate(all_sets):
                if subject_overlapping[i]:
                    a_set.update(set(indices))
                    break
            continue
        elif sum(subject_overlapping) >= 2:
            continue

        else:
        
            total_indices_all_sets = sum(len(a_set) for a_set in all_sets)
            if total_indices_all_sets == 0:
                all_sets[0].update(set(indices))
                continue

            # Calculate current ratios 
            current_ratios = [len(a_set_indices) / total_indices_all_sets for a_set_indices in all_sets]

            # Find where there is the bigger difference between the wanted ratios and the current ratios
            max_idx = np.argmax((np.array(wanted_ratios)-np.array(current_ratios))/np.array(wanted_ratios)) 
            # Update the set based on max_idx
            if max_idx < len(all_sets):
                # Update the set at the specific index indicated by max_idx
                all_sets[max_idx].update(set(indices))
            else:
                print("max_idx is out of range for all_sets")
        # Verify when 2 sets have overlapping indices
        flag = False
        for i in range(len(all_sets)):
            for j in range(i + 1, len(all_sets)):
                if all_sets[i].intersection(all_sets[j]):
                    overlapping = True
                    print(f"Sets {i} and {j} have overlapping indices.")
                    print(subject)
                    flag = True
                    break
                if flag:
                    break
        if flag:
            print(f"Condition met at iteration {iteration}")  # Print the iteration when the condition is met
            break
    
    return all_sets, indices_by_subject

def verify_nbr_per_sets(all_participant_ids):
    """
    Verifies the number of indices in each set and checks for overlapping indices.

    Parameters:
    - all_participant_ids (list): List of all participant IDs or indices to be split into sets.

    Example Usage:
    participant_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    verify_nbr_per_sets(participant_ids)

    Description:
    The function utilizes the 'custom_split_indices' function to split the provided participant IDs or indices into sets
    with a specified ratio. It then calculates and prints the total number of indices across all sets and the percentage
    of indices in each set relative to the total. Additionally, the function checks for overlapping indices between sets
    and prints a message if any overlap is found.

    Note: Ensure the 'custom_split_indices' function is defined and available in your code.
    """
    all_sets, indices_by_subject = custom_split_indices(all_participant_ids, ratio=0.1, nbr_sets=10, random_state=42)


    # Calculate the total length of all indices in all_sets
    total_indices = sum(len(a_set) for a_set in all_sets)

    print("Total number of indices in all sets:", total_indices)

    # Calculate the percentage of indices in each set
    for i, a_set in enumerate(all_sets):
        print(len(a_set))
        set_length = len(a_set)
        percentage = (set_length / total_indices) * 100
        print(f"Set {i + 1} has {set_length} indices, which is {percentage:.2f}% of the total.")

    overlapping = False

    for i in range(len(all_sets)):
        for j in range(i + 1, len(all_sets)):
            if all_sets[i].intersection(all_sets[j]):
                overlapping = True
                print(f"Sets {i} and {j} have overlapping indices.")

    if not overlapping:
        print("No sets have overlapping indices.")

def select_random_state(all_participant_ids, nbr_sets=10, ratio=0.1, max_to_search=5000):
    """
    Selects a random state to achieve desired ratio splits for a given number of sets.

    Parameters:
    - all_participant_ids (list): List of all participant IDs or indices to be split into sets.
    - nbr_sets (int): The number of sets to be created.
    - ratio (float): The desired ratio for each set.
    - max_to_search (int): The maximum number of random states to search for.

    Returns:
    - int: The selected random state that results in the closest achieved ratios to the desired ratios.

    Example Usage:
    participant_ids = [10006, 10008, 10003, 10001, 10010]
    random_state = select_random_state(participant_ids, nbr_sets=5, ratio=0.2, max_to_search=10000)
    print(f"Selected Random State: {random_state}")
    """
    wanted_ratios =  [ratio] * nbr_sets
    lowest_norm = np.linalg.norm(np.array(wanted_ratios))
    selected_indices = [[] for _ in range(nbr_sets)]
    for i in range(max_to_search):
        all_sets, indices_by_subject = custom_split_indices(all_participant_ids, ratio=0.1, nbr_sets=10, random_state=i)
        total_indices_all_sets = sum(len(a_set) for a_set in all_sets)
        current_ratios = [len(a_set_indices) / total_indices_all_sets for a_set_indices in all_sets]

        diff_norm = np.linalg.norm(np.array(wanted_ratios) - np.array(current_ratios))
        if diff_norm < lowest_norm:
            lowest_norm = diff_norm
            for x, a_set in enumerate(all_sets):
                selected_indices[x] = a_set
            selected_random_state = i

    return selected_random_state



def split_into_indices(all_sets, nbr_sets):
    """
    Split indices into training, validation, and test sets for cross-validation.

    Parameters:
    - all_sets (list): List containing sets of indices.
    - nbr_sets (int): Number of sets to split the data into.

    Returns:
    - tuple: A tuple containing three lists -
             - List of training indices for each iteration (train_indices_list).
             - List of validation indices for each iteration (val_indices_list).
             - List of test indices for each iteration (test_indices_list).

    Example Usage:
    all_sets = [...]
    nbr_sets = 5
    result = split_into_indices(all_sets, nbr_sets)
    print(f"Train Indices List: {result[0]}")
    print(f"Validation Indices List: {result[1]}")
    print(f"Test Indices List: {result[2]}")

    Description:
    The function takes a list of sets of indices and the number of sets (nbr_sets) to split the data into.
    It iteratively assigns each set as a test set and determines corresponding validation and training sets.
    The resulting indices for each iteration are stored in three separate lists and returned as a tuple.

    Note: The function prints example sets for each iteration as an illustration. Ensure to use or modify
    the sets as needed for your specific use case.
    """
    train_indices_list = []
    val_indices_list = []
    test_indices_list = []

    for i in range(nbr_sets):
        test_indices = list(all_sets[i])  # Set the test set

        # Determine the start of the validation indices based on the test index
        val_start = i * 2 + 1

        val_set_1 = list(all_sets[val_start % nbr_sets])  # First validation set
        val_set_2 = list(all_sets[(val_start + 1) % nbr_sets])  # Second validation set

        # Train indices are the rest of the sets not used for test and validation
        train_indices = []
        for j in range(nbr_sets):
            if j not in [i, val_start % nbr_sets, (val_start + 1) % nbr_sets]:
                train_indices.extend(list(all_sets[j]))

        train_indices_list.append(train_indices)
        val_indices_list.append(val_set_1 + val_set_2)
        test_indices_list.append(test_indices)

        # Use the sets as needed (train, validation, test)
        # For example:
        print(f"Iteration {i + 1}:")
        print("Train sets:", train_indices)
        print("Validation sets:", val_set_1 + val_set_2)
        print("Test set:", test_indices)
        print()

        return train_indices_list, val_indices_list, test_indices_list

def count_pairs_per_set(indices_list):
    """
    Count the number of unique pairs in each set and the total number of pairs across all sets.

    Parameters:
    - indices_list (list): List containing sets of indices.

    Returns:
    - None: Prints the size of each set and the total number of unique pairs.

    Example Usage:
    test_sets = [...]
    count_pairs_per_set(test_sets)

    Description:
    The function takes a list of sets of indices (test_indices_list) and iterates through each set to count
    the number of unique pairs. It prints the size of each set and the total number of unique pairs across
    all sets.

    Note: The function prints the size of each set along with its index and the total number of unique pairs.
    You can use this information to verify the distribution and size of sets in your specific use case.
    """
    sum = 0 
    list_total = []
    for i in range(len(indices_list)):
        elements = indices_list[i]
        size = len(set(indices_list[i])) 
        print(i, ':', size)
        sum += size
        list_total.extend(elements)
    print('total number of pairs:', sum)
    
def process_selected_sets(split_nbr, intra_scan_paths, inter_scan_paths, train_indices_list, val_indices_list, test_indices_list):
    """
    Read and process selected MRI scan sets for training, validation, and testing.

    Parameters:
    - split_nbr (int): Split currently used from the sets
    - intra_scan_paths (list): List of file paths for intra scans.
    - inter_scan_paths (list): List of file paths for inter scans.
    - train_indices_list (list): List containing sets of training indices.
    - val_indices_list (list): List containing sets of validation indices.
    - test_indices_list (list): List containing sets of test indices.

    Returns:
    - tuple: A tuple containing selected paths, processed sets, and their corresponding labels for training, validation, and testing.

    Description:
    The function takes lists of intra and inter scan paths along with sets of training, validation, and test indices.
    It processes the selected paths by resizing each scan and assigns labels (1 for intra and 0 for inter).
    The function returns selected paths, processed sets, and their corresponding labels as a tuple.

    Note: The 'process_scan' function should be defined and available in your code for the function to work correctly.
    Ensure that the processed sets are used for subsequent model training or evaluation.
    """
    # Read and process the scans only when the train, val, test sets are selected.
    # Each scan is then resized across height, width, and depth and rescaled.
    intra_scans = np.array(intra_scan_paths)
    inter_scans = np.array(inter_scan_paths)

    # For the MRI scans having a logJacobian derived from intra reg assign 1, 
    # for inter assign 0.
    intra_labels = np.array([1 for _ in range(len(intra_scans))])
    inter_labels = np.array([0 for _ in range(len(inter_scans))])

    X = np.concatenate((intra_scans, inter_scans), axis=0)
    y = np.concatenate((intra_labels, inter_labels), axis=0)

    # Read and process only the selected paths for the train, val and test sets
    selected_train_paths = [X[i] for i in train_indices_list[split_nbr]]
    x_train, y_train = np.array([process_scan(path) for path in selected_train_paths]), y[train_indices_list[split_nbr]]
    selected_val_paths = [X[i] for i in val_indices_list[split_nbr]]
    x_val, y_val = np.array([process_scan(path) for path in selected_val_paths]), y[val_indices_list[split_nbr]]
    selected_test_paths = [X[i] for i in test_indices_list[split_nbr]]
    x_test, y_test = np.array([process_scan(path) for path in selected_test_paths]), y[test_indices_list[split_nbr]]

    print(
        "Number of samples in train, validation and test are %d, %d and %d."
        % (x_train.shape[0], x_val.shape[0], x_test.shape[0])
    )

    return selected_train_paths, x_train, y_train, selected_val_paths, x_val, y_val, selected_test_paths, x_test, y_test

import random

from scipy import ndimage


@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < -1] = -1
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def data_loaders(x_train, y_train, x_val, y_val, batch_size):
    # Define data loaders.
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    # Augment the data on the fly during training.
    train_dataset = (
        train_loader.shuffle(len(x_train))
        .map(train_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )
    # Only rescale.
    validation_dataset = (
        validation_loader.shuffle(len(x_val))
        .map(validation_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )
    return train_dataset, validation_dataset

if __name__ == "__main__":
    # For the NO OVERLAP scenario
    # Specify the path to your TSV file
    file_path = '/home/andjela/Documents/intra-inter-ddfs/src/all-participants.tsv'

    # Call the function to create the dictionary for all 64 patients
    participant_mapping = read_tsv_and_create_dict(file_path)

    # Obtain dataframe with subject, scan_id, age and sex
    full_df = read_tsv_and_create_dataframe(file_path)

    path_inter = '/home/andjela/Documents/intra-inter-ddfs'
    path_intra = '/home/andjela/joplin-intra-inter/'
    experiment_name = 'mix_ra'
    experiment_name_inter = f'inter_{experiment_name}'
    experiment_name_intra = f'intra_{experiment_name}'
    intra_scan_paths, nbr_intra_pairs, inter_scan_paths, nbr_inter_pairs = read_paths(path_intra, path_inter, experiment_name_inter, experiment_name_intra)

    all_participant_ids = extract_all_participant_ids(intra_scan_paths, inter_scan_paths, participant_mapping, full_df)
    selected_random_state = select_random_state(all_participant_ids, nbr_sets=10, ratio=0.1, max_to_search=5000)
    all_sets, indices_by_subject = custom_split_indices(all_participant_ids, ratio=0.1, nbr_sets=10, random_state=selected_random_state)

    # Calculate the total length of all indices in all_sets
    total_indices = sum(len(a_set) for a_set in all_sets)

    print("Total number of indices in all sets:", total_indices)

    # Calculate the percentage of indices in each set
    for i, a_set in enumerate(all_sets):
        set_length = len(a_set)
        percentage = (set_length / total_indices) * 100
        print(f"Set {i + 1} has {set_length} indices, which is {percentage:.2f}% of the total.")

    print('The chosen random_state is:', selected_random_state)