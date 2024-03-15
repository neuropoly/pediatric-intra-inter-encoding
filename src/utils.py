import pandas as pd
import numpy as np


def find_pairs_by_scan_ids(scan_ids, pairs):
    """
    Find pairs of participants based on specified scan_ids.

    This function takes a list of tuples containing participant_id, scan_id, and additional data.
    It iterates through the data and finds all unique pairs of participants where the first participant's scan_id
    matches `scan_id_1` and the second participant's scan_id matches `scan_id_2`.

    Parameters:
        scan_ids (list): A list of tuples, where each tuple contains two scan_ids (str).
        pairs (list): A list of tuples, where each tuple contains two pairs of scan_id data.

    Returns:
        list: A list of tuples, where each tuple contains two tuples representing the pairs.
            Each inner tuple contains the participant_id (str), scan_id (str), and additional data of a participant.

    Example:
        >>> scan_ids_to_find = [("S2", "S5"), ("P4", "PS15_081_CL_Dev_008")]
        >>> pairs = find_pairs_by_scan_ids(scan_ids_to_find, list_of_pairs)
        >>> print(pairs)
        [(('P2', 'S2', 30), ('P5', 'S5', 35)), (('P4', 'PS15_081_CL_Dev_008', 28), ...)]
    """
    result_pairs = []
    for scan_id_1, scan_id_2 in scan_ids:
        for pair in pairs:
            if scan_id_1 in pair[0] and scan_id_2 in pair[1]:
                result_pairs.append(pair)
    return result_pairs

def find_pairs(data, type):
    """
    Find pairs of participants with age difference less than 0.5 years.

    This function takes a list of tuples containing participant_id, scan_id, and age data.
    It iterates through the data and finds all unique pairs of participants whose age difference is less than 0.5 years.

    Parameters:
        data (list): A list of tuples, where each tuple contains the participant_id (str), scan_id (str), and age (float) data.

    Returns:
        list: A list of tuples, where each tuple contains two tuples representing the pairs.
              Each inner tuple contains the participant_id (str), scan_id (str), and age (float) of a participant.

    Example:
        >>> data = [('sub-001', 'PS14_001', 4.1389), ('sub-002', 'PS14_002', 3.789), ...]
        >>> pairs = find_pairs(data)
        >>> print(pairs)
        [(('sub-001', 'PS14_001', 4.1389), ('sub-002', 'PS14_002', 3.789)), ...]
    """
    pairs = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            participant_id_1, scan_id_1, age_1 = data[i]
            participant_id_2, scan_id_2, age_2 = data[j]

            if type == 'intra':
                if participant_id_1 == participant_id_2:
                    pairs.append(((participant_id_1, scan_id_1, age_1), (participant_id_2, scan_id_2, age_2)))
            else:
                if participant_id_1 != participant_id_2 and abs(age_1 - age_2) < 0.025:
                    pairs.append(((participant_id_1, scan_id_1, age_1), (participant_id_2, scan_id_2, age_2)))
    
    return pairs

def read_tsv_file(file_path):
    """
    Read a TSV file and extract participant_id, scan_id, and age data.

    This function reads a TSV (Tab-Separated Values) file and extracts the relevant data, i.e., participant_id, scan_id, and age.
    
    Parameters:
        file_path (str): The path to the TSV file.

    Returns:
        list: A list of tuples, where each tuple contains the participant_id (str), scan_id (str), and age (float) for each entry in the TSV file.

    Example:
        >>> tsv_file_path = "path_to_your_tsv_file.tsv"
        >>> data = read_tsv_file(tsv_file_path)
        >>> print(data)
        [('sub-001', 'PS14_001', 'ses-001', 4.1389), ('sub-002', 'PS14_002', 'ses-001', 3.789), ...]
    """
    data = []
    with open(file_path, 'r') as file:
        header = file.readline()  # Skip the header line
        for line in file:
            participant_id, sub_id_bids, scan_id, session, age, sex, group = line.strip().split('\t')
            data.append((participant_id, scan_id, float(age)))
    return data

def calculate_average_log_jacobian(volume):
    # Iterate through the image_3d and store the absolute values of all elements in abs_values
    abs_values = np.abs(volume)

    # Calculate the average of the absolute values
    average_absolute = np.mean(abs_values)

    return average_absolute

def create_df_log_jacobian_data(all_participant_ids_and_scan_ids, all_sets, full_df):
    df_279 = pd.DataFrame(columns=['scan_id_1', 'scan_id_2', 'age_interval', 'avg_abs_log_jacobian', 'group'])
    # Combine all sets into a single list
    combined_list_of_sets = []
    for a_set in all_sets:
        combined_list_of_sets.extend(list(a_set))

    print(combined_list_of_sets)

    for i in combined_list_of_sets:
        participant_ids_scan_ids = all_participant_ids_and_scan_ids[i]
        # Intra case
        if participant_ids_scan_ids[0][0] == participant_ids_scan_ids[0][1]:
            group = 'intra'
            specific_scan_id_1 = participant_ids_scan_ids[1][0]
            specific_scan_id_2 = participant_ids_scan_ids[1][1]
            # Filter the DataFrame to retrieve the ages for the specific scan IDs
            age_for_scan_id_1 = full_df.loc[full_df['scan_id'] == specific_scan_id_1, 'age'].iloc[0]
            age_for_scan_id_2 = full_df.loc[full_df['scan_id'] == specific_scan_id_2, 'age'].iloc[0]
            age_interval = np.abs(age_for_scan_id_2 - age_for_scan_id_1)
            df_279 = df_279.append({'scan_id_1': specific_scan_id_1, 'scan_id_2': specific_scan_id_2, 'age_interval': age_interval, 'group': group}, ignore_index=True)
        # Inter case
        else:
            group = 'inter'
            specific_scan_id_1 = participant_ids_scan_ids[1][0]
            specific_scan_id_2 = participant_ids_scan_ids[1][1]
            # Filter the DataFrame to retrieve the ages for the specific scan IDs
            age_for_scan_id_1 = full_df.loc[full_df['scan_id'] == specific_scan_id_1, 'age'].iloc[0]
            age_for_scan_id_2 = full_df.loc[full_df['scan_id'] == specific_scan_id_2, 'age'].iloc[0]
            age_interval = np.abs(age_for_scan_id_2 - age_for_scan_id_1)
            df_279 = df_279.append({'scan_id_1': specific_scan_id_1, 'scan_id_2': specific_scan_id_2, 'age_interval': age_interval, 'group': group}, ignore_index=True)
        if i in train_indices_list[split_nbr]:
            index_of_i = train_indices_list[split_nbr].index(i)
            volume = x_train[index_of_i]
            avg_abs_log_jacobian = calculate_average_log_jacobian(volume)
            # This line gets the row index where 'scan_id_1' and 'scan_id_2' match the specified values
            row_index = df_279[(df_279['scan_id_1'] == specific_scan_id_1) & (df_279['scan_id_2'] == specific_scan_id_2)].index

            # You can access the index, assuming only one row matches these conditions
            if not row_index.empty:
                # Update the 'age_interval' column for the identified row
                df_279.at[row_index, 'avg_abs_log_jacobian'] = avg_abs_log_jacobian
            else:
                print('Row index train empty')
        elif i in val_indices_list[split_nbr]:
            index_of_i = val_indices_list[split_nbr].index(i)
            volume = x_val[index_of_i]
            avg_abs_log_jacobian = calculate_average_log_jacobian(volume)
            # This line gets the row index where 'scan_id_1' and 'scan_id_2' match the specified values
            row_index = df_279[(df_279['scan_id_1'] == specific_scan_id_1) & (df_279['scan_id_2'] == specific_scan_id_2)].index

            # You can access the index, assuming only one row matches these conditions
            if not row_index.empty:
                # Update the 'age_interval' column for the identified row
                df_279.at[row_index, 'avg_abs_log_jacobian'] = avg_abs_log_jacobian
            else:
                print('Row index val empty')
        else:
            index_of_i = test_indices_list[split_nbr].index(i)
            volume = x_test[index_of_i]
            avg_abs_log_jacobian = calculate_average_log_jacobian(volume)
            # This line gets the row index where 'scan_id_1' and 'scan_id_2' match the specified values
            row_index = df_279[(df_279['scan_id_1'] == specific_scan_id_1) & (df_279['scan_id_2'] == specific_scan_id_2)].index

            # You can access the index, assuming only one row matches these conditions
            if not row_index.empty:
                # Update the 'age_interval' column for the identified row
                df_279.at[row_index, 'avg_abs_log_jacobian'] = avg_abs_log_jacobian
            else:
                print('Row index test empty')

    
