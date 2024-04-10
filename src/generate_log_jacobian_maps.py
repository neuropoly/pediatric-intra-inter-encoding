# previous file name: inter_reg.py
import os
from collections import defaultdict
from timeit import default_timer as timer
from datetime import timedelta
from datetime import datetime
import pandas as pd
import csv
import numpy as np
# from matplotlib import pyplot as plt
import random
import shutil
import nibabel as nib

def subj_to_scan_id(tsv_file_path):
    #Read the all-participants.tsv file into a DataFrame
    df = pd.read_csv(tsv_file_path, sep='\t')

    # Initialize an empty dictionary to store participant_id and associated scan_ids
    participant_to_scan = {}

    # Iterate through the DataFrame and populate the dictionary
    for index, row in df.iterrows():
        participant_id = row['participant_id']
        scan_id = row['scan_id']
        if participant_id not in participant_to_scan:
            participant_to_scan[participant_id] = []
        participant_to_scan[participant_id].append(scan_id)

    # print(participant_to_scan)
    return participant_to_scan

def scan_id_to_sub_id(tsv_file_path):
    #Read the all-participants.tsv file into a DataFrame
    df = pd.read_csv(tsv_file_path, sep='\t')

    # Initialize an empty dictionary to store participant_id and associated scan_ids
    scan_to_sub = {}

    # Iterate through the DataFrame and populate the dictionary
    for index, row in df.iterrows():
        sub_id = row['sub_id_bids']
        scan_id = row['scan_id']
        if scan_id not in scan_to_sub:
            scan_to_sub[scan_id] = []
        scan_to_sub[scan_id].append(sub_id)

    # print(scan_to_cdsub)
    return scan_to_sub

def scan_id_to_session(tsv_file_path):
    #Read the all-participants.tsv file into a DataFrame
    df = pd.read_csv(tsv_file_path, sep='\t')

    # Initialize an empty dictionary to store participant_id and associated scan_ids
    scan_to_session = {}

    # Iterate through the DataFrame and populate the dictionary
    for index, row in df.iterrows():
        scan_id = row['scan_id']
        session = row['session']
        if scan_id not in scan_to_session:
            scan_to_session[scan_id] = []
        scan_to_session[scan_id].append(session)

    # print(scan_to_session)
    return scan_to_session

def find_participant_id(scan_id, participant_scan_ids):
    for participant_id, scan_ids in participant_scan_ids.items():
        if scan_id in scan_ids:
            return participant_id
    return None

def extract_non_longitudinal_scan_ids(tsv_file_path):

    # Read the all-participants.tsv file into a DataFrame
    df = pd.read_csv(tsv_file_path, sep='\t')

    # Count the occurrences of each participant_id
    participant_counts = df['participant_id'].value_counts()

    # Filter scan_ids associated with participant_id occurring only once
    unique_participants = participant_counts[participant_counts == 1].index
    unique_scan_ids = df[df['participant_id'].isin(unique_participants)]['scan_id'].tolist()

    return unique_scan_ids

def skull_strip_non_longitudinal(tsv_file_path):

    p_dict = subj_to_scan_id(tsv_file_path)
    # p = '10136'
    # scanID = p_dict[p][0]
    non_longitudinal_scan_ids = extract_non_longitudinal_scan_ids(tsv_file_path)

    scan_to_session = scan_id_to_session(tsv_file_path)
    scan_to_sub = scan_id_to_sub_id(tsv_file_path)

    scan_ids_intra_to_do = ["PS16_002", "PS17_024", "PS16_054", "PS17_007", "PS0322-10-2", "PS1477-10-1", "PS16_006"]

    for scan_id in scan_ids_intra_to_do:
        p = find_participant_id(scan_id, p_dict)
        sub_id = scan_to_sub[scan_id][0]
        session = scan_to_session[scan_id][0]
        # os.system(f"python reg_t12mni_N4corr.py {p} {scan_id} {sub_id} {session}")
        os.system(f"python cbf2mni.py {p} {scan_id} {sub_id} {session}")
    
    # Test with one of the 32 subjects
    # scan_id = 'PS14_006'
    # p = 10001
    # sub_id = scan_to_sub[scan_id][0]
    # session = scan_to_session[scan_id][0]
    # print(sub_id, session)
    # # os.system(f"python reg_t12mni_N4corr.py {p} {scan_id} {sub_id} {session}")
    # os.system(f"python cbf2mni.py {p} {scan_id} {sub_id} {session}")

    # not_wanted_ps = [10001, 10002, 10011, 10012, 10024]
    # for p, scan_ids in p_dict.items():
    #     if not p in not_wanted_ps:
    #         for scan_id in scan_ids:
    #             if scan_id in non_longitudinal_scan_ids:
                    
    #                 sub_id = scan_to_sub[scan_id][0]
    #                 session = scan_to_session[scan_id][0]
    #                 os.system(f"python reg_t12mni_N4corr.py {p} {scan_id} {sub_id} {session}")
    #                 # os.system(f"python cbf2mni.py {p} {scan_id} {sub_id} {session}")
    #     else:
    #         print('Skipping:', p)
    # not_wanted_p = [10001]
    # for p, scan_ids in p_dict.items():
    #     if not p in not_wanted_p:
    #         for scan_id in scan_ids:
    #             if scan_id in non_longitudinal_scan_ids:
                    
    #                 sub_id = scan_to_sub[scan_id][0]
    #                 session = scan_to_session[scan_id][0]
    #                 # os.system(f"python reg_t12mni_N4corr.py {p} {scan_id} {sub_id} {session}")
    #                 os.system(f"python cbf2mni.py {p} {scan_id} {sub_id} {session}")

    #     else:
    #         print('Skipping:', p)

def create_participant_file(output_dir):
    mapping = subj_to_scan_id()
    filename = 'participants.tsv'

    xl_path = '/media/andjela/SeagatePor1/CP/Calgary_Preschool_Dataset_Updated_20200213_copy.xlsx' 
    df = pd.read_excel(xl_path)

    with open(f'{output_dir}/{filename}', "w") as file:
        writer = csv.writer(file, delimiter='\t')
    
        # Write header row
        writer.writerow(['participant_id', 'sub_id_bids', 'scan_id', 'session', 'age', 'sex', 'group'])
        group = 'control'
        
        # Write data rows
        count_sub = 0
        count_ses = 0
        for sub, scan_ids in mapping.items():
            for scan_id in scan_ids:
                p = df.loc[df['ScanID'] == scan_id, 'PreschoolID'].values[0]
                age = df.loc[df['ScanID'] == scan_id, 'Age (Years)'].values[0]
                sex = df.loc[df['ScanID'] == scan_id, 'Biological Sex (Female = 0; Male = 1)'].values[0]
                writer.writerow([f'{p}', 'sub-{:03d}'.format(count_sub+1), f'{scan_id}', 'ses-{:03d}'.format(count_ses+1), f'{age}', f'{sex}', f'{group}' ])
                count_ses += 1
            count_ses = 0
            count_sub += 1

def find_intra_pairs(data):
    pairs = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            participant_id_1, scan_id_1, age_1, sex_1 = data[i]
            participant_id_2, scan_id_2, age_2, sex_2 = data[j]

            if participant_id_1 == participant_id_2:
                pairs.append(((participant_id_1, scan_id_1, age_1, sex_1), (participant_id_2, scan_id_2, age_2, sex_2)))

    return pairs

def find_inter_pairs(data):
    pairs = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            participant_id_1, scan_id_1, age_1, sex_1 = data[i]
            participant_id_2, scan_id_2, age_2, sex_2 = data[j]

            
            if participant_id_1 != participant_id_2:
                pairs.append(((participant_id_1, scan_id_1, age_1, sex_1), (participant_id_2, scan_id_2, age_2, sex_2)))
    
    return pairs

def find_closest_values_with_indices(target_value, array):
    
    # Calculate the absolute differences between the target value and each element in the array
    differences = np.abs(array - target_value)
    
    # Find the indices of the two closest values
    closest_indices = np.argsort(differences)[:2]
    
    # Get the corresponding closest values
    closest_values = array[closest_indices]
    closest_values = sorted(closest_values)
    
    return closest_values, closest_indices[0]

def find_inter_pairs_with_matching_distribution_init(data, n_bins, sex=None, same_sex=False):
    # Calculate histogram distribution of intra_array
    intra_pairs_info = find_intra_pairs(data)
    
    intra_age_intervals = np.abs([item[0][2]-item[1][2] for item in intra_pairs_info])
    hist_distribution_intra_age_intervals, bin_edges_intra_age_intervals = np.histogram(intra_age_intervals, bins=n_bins) # Mean: 1.152, std: 0.684

    intra_init_ages = [min(item[0][2],item[1][2]) for item in intra_pairs_info]
    hist_distribution_intra_init_ages, bin_edges_intra_init_ages = np.histogram(intra_init_ages, bins=n_bins) # Mean: 1.152, std: 0.684

    if same_sex:
        intra_sexes = [item[0][3] for item in intra_pairs_info]

        count_0 = intra_sexes.count(0)
        count_1 = intra_sexes.count(1)

        print("Number of 0's:", count_0)
        print("Number of 1's:", count_1)
    
    # Known values for the new group
    inter_pairs_info = find_inter_pairs(data)
    
    inter_age_intervals = np.abs([item[0][2]-item[1][2] for item in inter_pairs_info])

    inter_init_ages = [min(item[0][2],item[1][2]) for item in inter_pairs_info]
    
    inter_pairs_to_keep = []
    
    for idx_intra, intra_pair in enumerate(intra_pairs_info):
        sex_intra = intra_pair[0][3]
        if sex is not None:
            if sex != sex_intra:
                continue
        
        limits_intervals, idx_bin_intervals  = find_closest_values_with_indices(intra_age_intervals[idx_intra], bin_edges_intra_age_intervals)
        limits_init, idx_bin_init = find_closest_values_with_indices(intra_init_ages[idx_intra], bin_edges_intra_init_ages)
        
        # Inter pairs where the age interval or initial age are within the same bin as intra pair
        inter_set_age_intervals = set()
        inter_set_init_ages = set()
        if same_sex:
            inter_set_sex = set()
        
        for idx_inter, inter_pair in enumerate(inter_pairs_info):
            if inter_age_intervals[idx_inter] >= limits_intervals[0] and inter_age_intervals[idx_inter] < limits_intervals[1]:
                inter_set_age_intervals.add(inter_pair)
            if inter_init_ages[idx_inter] >= limits_init[0] and inter_init_ages[idx_inter] < limits_init[1]:
                inter_set_init_ages.add(inter_pair)
            if same_sex:
                if inter_pair[0][3] == sex_intra and inter_pair[1][3] == sex_intra: 
                    inter_set_sex.add(inter_pair)
        if same_sex:
            intersection = inter_set_age_intervals & inter_set_init_ages & inter_set_sex
            # Sort the intersection by age interval in reverse to keep more inter-pairs not within intra-pairs
            sorted_intersection = sorted(intersection, reverse=True)
            for element in sorted_intersection:

                if element not in inter_pairs_to_keep:
                    inter_pairs_to_keep.append(element)

                    break
        else:
            intersection = inter_set_age_intervals & inter_set_init_ages
            # Sort the intersection by age interval in reverse to keep more inter-pairs not within intra-pairs
            sorted_intersection = sorted(intersection, reverse=True)
            for element in sorted_intersection:

                if element not in inter_pairs_to_keep:
                    inter_pairs_to_keep.append(element)

                    break

    inter_age_intervals_to_keep = np.abs([item[0][2]-item[1][2] for item in inter_pairs_to_keep])
    inter_init_ages_to_keep = [min(item[0][2], item[1][2]) for item in inter_pairs_to_keep]

    # plt.hist(inter_age_intervals_to_keep, range=(bin_edges_intra_age_intervals.min(), bin_edges_intra_age_intervals.max()), bins=n_bins, alpha=0.5, label='Inter Group')
    # plt.hist(intra_age_intervals, bins=n_bins, alpha=0.5, label='Intra Group')
    # plt.legend()
    # plt.show()

    # plt.hist(inter_init_ages_to_keep, range=(bin_edges_intra_init_ages.min(), bin_edges_intra_init_ages.max()), bins=n_bins, alpha=0.5, label='Inter Group')
    # plt.hist(intra_init_ages, bins=n_bins, alpha=0.5, label='Intra Group')
    # plt.legend()
    # plt.show()
        
    return inter_pairs_to_keep

    

def run_ants_registration(pair, transfo, img_dir, img_dir_out, img_dir_init_transfo, scan_to_sub, scan_to_session):
    """
    Run ANTs image registration workflow and generates deformed images and its corresponding h5 file.

    This function performs ANTs image registration based on the specified transformation type.

    Parameters:
        pair (list): A list containing tuples of image data. Each tuple contains participant_id, image_filename, and age for both moving and fixed images.
        transfo (str): The type of transformation to apply during image registration.
        img_dir (str): The directory path where the image files are stored.
        img_dir_out (str): The output directory where the registration results will be stored.

    Returns:
        tuple: A tuple containing the age interval between moving and fixed images, the filenames of the moving and fixed images.

    Example:
        >>> tsv_file_path = "participants.tsv"
        >>> data = read_tsv_file(tsv_file_path)
        >>> pairs = find_pairs(data)
        >>> transfo ='rigid_inter'
        >>> img_dir = '/media/andjela/SeagatePor/PairRegData/images/'
        >>> img_dir_out = f'/media/andjela/SeagatePor/PairReg/{transfo}/images/'

        >>> for pair in pairs:
        ...     interval, mov, fix = run_ants_registration(pair, transfo, img_dir, img_dir_out)
        ...     print(f"Pair: {mov}_{fix}, Age Interval: {interval}")
    """

    start = timer()
    p_mov, mov, age_mov, _ = pair[0]
    p_fix, fix, age_fix, _ = pair[1]

    # Grab equivalent sub_id and session
    
    sub_id_mov = scan_to_sub[mov][0]
    session_mov = scan_to_session[mov][0]

    sub_id_fix = scan_to_sub[fix][0]
    session_fix = scan_to_session[fix][0]

    # Ensure mov2fix is from younger subject to older
    if age_mov > age_fix:
        mov, fix = fix, mov
        age_mov, age_fix = age_fix, age_mov
        p_mov, p_fix = p_fix, p_mov
        sub_id_mov, sub_id_fix = sub_id_fix, sub_id_mov
        session_mov, session_fix = session_fix, session_mov

    interval = age_fix-age_mov
    
    if not os.path.exists(f'{img_dir_out}{mov}_{fix}/'): 
        os.mkdir(f'{img_dir_out}{mov}_{fix}/')
    if transfo=='affine_inter':
        print('Affine')
        #f"--use-estimate-learning-rate-once 1 " \ (invalid flag given by ANTs)
        os.system(f"antsRegistration --float --collapse-output-transforms 1 --dimensionality 3 " \
                f"--initial-moving-transform [ {img_dir}{p_fix}/{fix}.nii.gz, {img_dir}{p_mov}/{mov}.nii.gz, 0 ] " \
                f"--interpolation Linear " \
                f"--output [ {img_dir_out}{mov}_{fix}/mov2fix_, {img_dir_out}{mov}_{fix}/mov2fix_warped_image.nii.gz, {img_dir_out}{mov}_{fix}/fix2mov_warped_image.nii.gz ] " \
                f"--transform Affine[ 0.1 ] " \
                f"--metric Mattes[ {img_dir}{p_fix}/{fix}.nii.gz, {img_dir}{p_mov}/{mov}.nii.gz, 1, 32, Regular, 0.3 ] " \
                f"--convergence [850x250x250,1e-7,25] " \
                f"--shrink-factors 4x2x1 " \
                f"--smoothing-sigmas 2x1x0vox " \
                f"--use-histogram-matching 1 " \
                f"--winsorize-image-intensities [ 0.005, 0.995 ] " \
                f"--verbose 1"
                )
        # warp subject calculated mov mask to fix
        os.system(f"antsApplyTransforms --default-value 0 --float 0 " \
            f"--input /media/andjela/SeagatePor/work_dir2/cbf2mni_wdir/{p_mov}/{mov}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz " \
            f"--input-image-type 0 --interpolation Linear --output {img_dir_out}{mov}_{fix}/mask_trans.nii " \
            f"--reference-image {img_dir}{p_fix}/{fix}.nii.gz " \
            f"--transform {img_dir_out}{mov}_{fix}/mov2fix_0GenericAffine.mat")
        #brain extraction for mov image which becomes mov2fix_warped
        os.system(f'fslmaths {img_dir_out}{mov}_{fix}/mov2fix_warped_image.nii.gz -mul {img_dir_out}{mov}_{fix}/mask_trans.nii {img_dir_out}{mov}_{fix}/{mov}_dtype.nii.gz -odt float')
        #brain extraction for fix image which stays fix
        os.system(f'fslmaths {img_dir}{p_fix}/{fix}.nii.gz -mul /media/andjela/SeagatePor/work_dir2/cbf2mni_wdir/{p_fix}/{fix}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz {img_dir_out}{mov}_{fix}/{fix}_dtype.nii.gz -odt float')
        end = timer()
        print('TIME TAKEN:', timedelta(seconds=end-start))
        return interval, mov, fix
    elif transfo=='rigid_affine_intra':
        print('Rigid-affine')
        os.system(
                f"antsRegistration --dimensionality 3 --float 0 " \
                f"--output [ {img_dir_out}{mov}_{fix}/mov2fix_, {img_dir_out}{mov}_{fix}/mov2fix_warped_image.nii.gz, {img_dir_out}{mov}_{fix}/fix2mov_warped_image.nii.gz ] " \
                #f"--initialize-transforms-per-stage 1 " \
                f"--transform Affine[0.1] " \
                f"--initial-moving-transform {img_dir_init_transfo}{mov}_{fix}/mov2fix_Composite.h5 " \
                f"--interpolation Linear " \
                f"--metric MI[ {img_dir_init_transfo}{mov}_{fix}/{fix}_dtype.nii.gz, {img_dir_init_transfo}{mov}_{fix}/{mov}_dtype.nii.gz, 1, 32, Regular, 0.3 ] " \
                f"--convergence [850x250x250, 1e-7, 25] " \
                f"--shrink-factors 4x2x1 " \
                f"--smoothing-sigmas 2x1x0vox " \
                f"--use-histogram-matching 1 " \
                f"--winsorize-image-intensities [0.005, 0.995] " \
                f"--write-composite-transform 1 " \
                #f"--help " \
                f"--verbose 1"
                )
        end = timer()
        print('TIME TAKEN:', timedelta(seconds=end-start))
        return interval, mov, fix
    elif transfo=='rigid_affine_inter':
        print('Rigid-affine')
        os.system(
                f"antsRegistration --dimensionality 3 --float 0 " \
                f"--output [ {img_dir_out}{mov}_{fix}/mov2fix_, {img_dir_out}{mov}_{fix}/mov2fix_warped_image.nii.gz, {img_dir_out}{mov}_{fix}/fix2mov_warped_image.nii.gz ] " \
                #f"--initialize-transforms-per-stage 1 " \
                f"--transform Affine[0.1] " \
                f"--initial-moving-transform {img_dir_init_transfo}{mov}_{fix}/mov2fix_Composite.h5 " \
                f"--interpolation Linear " \
                f"--metric MI[ {img_dir_init_transfo}{mov}_{fix}/{fix}_dtype.nii.gz, {img_dir_init_transfo}{mov}_{fix}/{mov}_dtype.nii.gz, 1, 32, Regular, 0.3 ] " \
                f"--convergence [850x250x250, 1e-7, 25] " \
                f"--shrink-factors 4x2x1 " \
                f"--smoothing-sigmas 2x1x0vox " \
                f"--use-histogram-matching 1 " \
                f"--winsorize-image-intensities [0.005, 0.995] " \
                f"--write-composite-transform 1 " \
                #f"--help " \
                f"--verbose 1"
                )
        end = timer()
        print('TIME TAKEN:', timedelta(seconds=end-start))
        return interval, mov, fix
    elif 'syn' in transfo:
        os.system(f"antsRegistration --dimensionality 3 --float 0 " \
                f"--output [ {img_dir_out}{mov}_{fix}/mov2fix_, {img_dir_out}{mov}_{fix}/mov2fix_warped_image.nii.gz, {img_dir_out}{mov}_{fix}/fix2mov_warped_image.nii.gz ] " \
                f"--interpolation Linear " \
                f"--winsorize-image-intensities [0.005,0.995] " \
                f"--use-histogram-matching 1 " \
                f"--write-composite-transform 1 " \
                f"--transform SyN[0.1] " \
                f"--metric Mattes[ {img_dir}{p_fix}/{fix}.nii.gz, {img_dir}{p_mov}/{mov}.nii.gz, 1, 32, Regular, 0.3 " \
                f"--convergence [500x250x100,1e-6,10] " \
                f"--shrink-factors 4x2x1 " \
                f"--smoothing-sigmas 2x1x0vox " \
                f"--verbose 1"
                )
        end = timer()
        print('TIME TAKEN:', timedelta(seconds=end-start))
        return interval, mov, fix

    else:
        
        # Rigid Transfo
        if os.path.exists(f'{img_dir}{p_fix}/{fix}/wf/n4/{fix}_corrected.nii.gz') and os.path.exists(f'{img_dir}{p_mov}/{mov}/wf/n4/{mov}_corrected.nii.gz'):
            os.system(f"antsRegistration --dimensionality 3 --float 0 " \
                f"--output [ {img_dir_out}{mov}_{fix}/mov2fix_, {img_dir_out}{mov}_{fix}/mov2fix_warped_image.nii.gz, {img_dir_out}{mov}_{fix}/fix2mov_warped_image.nii.gz ] " \
                f"--interpolation Linear " \
                f"--winsorize-image-intensities [0.005,0.995] " \
                f"--use-histogram-matching 1 " \
                f"--write-composite-transform 1 " \
                f"--transform Rigid[0.1] " \
                f"--metric Mattes[ {img_dir}{p_fix}/{fix}/wf/n4/{fix}_corrected.nii.gz, {img_dir}{p_mov}/{mov}/wf/n4/{mov}_corrected.nii.gz, 1, 32, Regular, 0.3 ] " \
                f"--convergence [500x250x100,1e-6,10] " \
                f"--shrink-factors 4x2x1 " \
                f"--smoothing-sigmas 2x1x0vox " \
                f"--verbose 1"
                )
            # warp subject calculated mov mask to fix
            os.system(f"antsApplyTransforms --default-value 0 --float 0 " \
                f"--input {os.path.dirname(os.path.dirname(img_dir))}2/cbf2mni_wdir/{p_mov}/{mov}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz " \
                f"--input-image-type 0 --interpolation NearestNeighbor --output {img_dir_out}{mov}_{fix}/mask_trans.nii " \
                f"--reference-image {img_dir}{p_fix}/{fix}/wf/n4/{fix}_corrected.nii.gz " \
                f"--transform {img_dir_out}{mov}_{fix}/mov2fix_Composite.h5")
            
            #brain extraction for mov image which becomes mov2fix_warped
            os.system(f'fslmaths {img_dir_out}{mov}_{fix}/mov2fix_warped_image.nii.gz -mul {img_dir_out}{mov}_{fix}/mask_trans.nii {img_dir_out}{mov}_{fix}/{mov}_dtype.nii.gz -odt float')
            #brain extraction for fix image which stays fix
            os.system(f'fslmaths {img_dir}{p_fix}/{fix}/wf/n4/{fix}_corrected.nii.gz -mul {os.path.dirname(os.path.dirname(img_dir))}2/cbf2mni_wdir/{p_fix}/{fix}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz {img_dir_out}{mov}_{fix}/{fix}_dtype.nii.gz -odt float')
            end = timer()
            print('TIME TAKEN:', timedelta(seconds=end-start))

            return interval, mov, fix
        elif os.path.exists(f'{img_dir}{p_fix}/{fix}/wf/n4/{sub_id_fix}-{session_fix}-T1w_corrected.nii.gz') and os.path.exists(f'{img_dir}{p_mov}/{mov}/wf/n4/{sub_id_mov}-{session_mov}-T1w_corrected.nii.gz'):
            os.system(f"antsRegistration --dimensionality 3 --float 0 " \
                f"--output [ {img_dir_out}{mov}_{fix}/mov2fix_, {img_dir_out}{mov}_{fix}/mov2fix_warped_image.nii.gz, {img_dir_out}{mov}_{fix}/fix2mov_warped_image.nii.gz ] " \
                f"--interpolation Linear " \
                f"--winsorize-image-intensities [0.005,0.995] " \
                f"--use-histogram-matching 1 " \
                f"--write-composite-transform 1 " \
                f"--transform Rigid[0.1] " \
                f"--metric Mattes[ {img_dir}{p_fix}/{fix}/wf/n4/{sub_id_fix}-{session_fix}-T1w_corrected.nii.gz, {img_dir}{p_mov}/{mov}/wf/n4/{sub_id_mov}-{session_mov}-T1w_corrected.nii.gz, 1, 32, Regular, 0.3 ] " \
                f"--convergence [500x250x100,1e-6,10] " \
                f"--shrink-factors 4x2x1 " \
                f"--smoothing-sigmas 2x1x0vox " \
                f"--verbose 1"
                )
            # warp subject calculated mov mask to fix
            os.system(f"antsApplyTransforms --default-value 0 --float 0 " \
                f"--input {os.path.dirname(os.path.dirname(img_dir))}2/cbf2mni_wdir/{p_mov}/{mov}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz " \
                f"--input-image-type 0 --interpolation NearestNeighbor --output {img_dir_out}{mov}_{fix}/mask_trans.nii " \
                f"--reference-image {img_dir}{p_fix}/{fix}/wf/n4/{sub_id_fix}-{session_fix}-T1w_corrected.nii.gz " \
                f"--transform {img_dir_out}{mov}_{fix}/mov2fix_Composite.h5")

            #brain extraction for mov image which becomes mov2fix_warped
            os.system(f'fslmaths {img_dir_out}{mov}_{fix}/mov2fix_warped_image.nii.gz -mul {img_dir_out}{mov}_{fix}/mask_trans.nii {img_dir_out}{mov}_{fix}/{mov}_dtype.nii.gz -odt float')
            #brain extraction for fix image which stays fix
            os.system(f'fslmaths {img_dir}{p_fix}/{fix}/wf/n4/{sub_id_fix}-{session_fix}-T1w_corrected -mul {os.path.dirname(os.path.dirname(img_dir))}2/cbf2mni_wdir/{p_fix}/{fix}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz {img_dir_out}{mov}_{fix}/{fix}_dtype.nii.gz -odt float')
            end = timer()
            print('TIME TAKEN:', timedelta(seconds=end-start))

            return interval, mov, fix
        elif os.path.exists(f'{img_dir}{p_fix}/{fix}/wf/n4/{sub_id_fix}-{session_fix}-T1w_corrected.nii.gz') and os.path.exists(f'{img_dir}{p_mov}/{mov}/wf/n4/{mov}_corrected.nii.gz'):
            os.system(f"antsRegistration --dimensionality 3 --float 0 " \
                f"--output [ {img_dir_out}{mov}_{fix}/mov2fix_, {img_dir_out}{mov}_{fix}/mov2fix_warped_image.nii.gz, {img_dir_out}{mov}_{fix}/fix2mov_warped_image.nii.gz ] " \
                f"--interpolation Linear " \
                f"--winsorize-image-intensities [0.005,0.995] " \
                f"--use-histogram-matching 1 " \
                f"--write-composite-transform 1 " \
                f"--transform Rigid[0.1] " \
                f"--metric Mattes[ {img_dir}{p_fix}/{fix}/wf/n4/{sub_id_fix}-{session_fix}-T1w_corrected.nii.gz, {img_dir}{p_mov}/{mov}/wf/n4/{mov}_corrected.nii.gz, 1, 32, Regular, 0.3 ] " \
                f"--convergence [500x250x100,1e-6,10] " \
                f"--shrink-factors 4x2x1 " \
                f"--smoothing-sigmas 2x1x0vox " \
                f"--verbose 1"
                )
            # warp subject calculated mov mask to fix
            os.system(f"antsApplyTransforms --default-value 0 --float 0 " \
                f"--input {os.path.dirname(os.path.dirname(img_dir))}2/cbf2mni_wdir/{p_mov}/{mov}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz " \
                f"--input-image-type 0 --interpolation NearestNeighbor --output {img_dir_out}{mov}_{fix}/mask_trans.nii " \
                f"--reference-image {img_dir}{p_fix}/{fix}/wf/n4/{sub_id_fix}-{session_fix}-T1w_corrected.nii.gz " \
                f"--transform {img_dir_out}{mov}_{fix}/mov2fix_Composite.h5")

            #brain extraction for mov image which becomes mov2fix_warped
            os.system(f'fslmaths {img_dir_out}{mov}_{fix}/mov2fix_warped_image.nii.gz -mul {img_dir_out}{mov}_{fix}/mask_trans.nii {img_dir_out}{mov}_{fix}/{mov}_dtype.nii.gz -odt float')
            #brain extraction for fix image which stays fix
            os.system(f'fslmaths {img_dir}{p_fix}/{fix}/wf/n4/{sub_id_fix}-{session_fix}-T1w_corrected -mul {os.path.dirname(os.path.dirname(img_dir))}2/cbf2mni_wdir/{p_fix}/{fix}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz {img_dir_out}{mov}_{fix}/{fix}_dtype.nii.gz -odt float')
            end = timer()
            print('TIME TAKEN:', timedelta(seconds=end-start))

            return interval, mov, fix
        else:
            #case where fix_corrected and mov is in the sub-session format
            os.system(f"antsRegistration --dimensionality 3 --float 0 " \
                f"--output [ {img_dir_out}{mov}_{fix}/mov2fix_, {img_dir_out}{mov}_{fix}/mov2fix_warped_image.nii.gz, {img_dir_out}{mov}_{fix}/fix2mov_warped_image.nii.gz ] " \
                f"--interpolation Linear " \
                f"--winsorize-image-intensities [0.005,0.995] " \
                f"--use-histogram-matching 1 " \
                f"--write-composite-transform 1 " \
                f"--transform Rigid[0.1] " \
                f"--metric Mattes[ {img_dir}{p_fix}/{fix}/wf/n4/{fix}_corrected.nii.gz, {img_dir}{p_mov}/{mov}/wf/n4/{sub_id_mov}-{session_mov}-T1w_corrected.nii.gz, 1, 32, Regular, 0.3 ] " \
                f"--convergence [500x250x100,1e-6,10] " \
                f"--shrink-factors 4x2x1 " \
                f"--smoothing-sigmas 2x1x0vox " \
                f"--verbose 1"
                )
            # warp subject calculated mov mask to fix
            os.system(f"antsApplyTransforms --default-value 0 --float 0 " \
                f"--input {os.path.dirname(os.path.dirname(img_dir))}2/cbf2mni_wdir/{p_mov}/{mov}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz " \
                f"--input-image-type 0 --interpolation NearestNeighbor --output {img_dir_out}{mov}_{fix}/mask_trans.nii " \
                f"--reference-image {img_dir}{p_fix}/{fix}/wf/n4/{fix}_corrected.nii.gz " \
                f"--transform {img_dir_out}{mov}_{fix}/mov2fix_Composite.h5")
            
            #brain extraction for mov image which becomes mov2fix_warped
            os.system(f'fslmaths {img_dir_out}{mov}_{fix}/mov2fix_warped_image.nii.gz -mul {img_dir_out}{mov}_{fix}/mask_trans.nii {img_dir_out}{mov}_{fix}/{mov}_dtype.nii.gz -odt float')
            #brain extraction for fix image which stays fix
            os.system(f'fslmaths {img_dir}{p_fix}/{fix}/wf/n4/{fix}_corrected.nii.gz -mul {os.path.dirname(os.path.dirname(img_dir))}2/cbf2mni_wdir/{p_fix}/{fix}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz {img_dir_out}{mov}_{fix}/{fix}_dtype.nii.gz -odt float')
            end = timer()
            print('TIME TAKEN:', timedelta(seconds=end-start))

            return interval, mov, fix


def run_ants_intra_reg(pair, img_dir, img_dir_out):
    start = timer()
    p, mov, age_mov = pair[0]
    p, fix, age_fix = pair[1]

    # Ensure mov2fix is from younger subject to older
    if age_mov > age_fix:
        mov, fix = fix, mov
        age_mov, age_fix = age_fix, age_mov

    interval = age_fix-age_mov
    out_path = f"{img_dir_out}/{mov}_{fix}/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if age_mov > age_fix:
        print(mov, fix, 'Age mov < Age fix')
        # os.system(f"antsRegistration -d 3 -m Demons[{img_dir}{p}_{fix}_{mov}/{fix}_dtype.nii.gz, {img_dir}{p}_{fix}_{mov}/{mov}_dtype.nii.gz, 1, 4] " \
        #     f"-t SyN[0.1, 3, 0] -f 2x1 -s 1x0vox -u 0 " \
        #     f"-c [10x5] -o [{out_path}movingToFixed, {out_path}movingToFixedDeformed.nii.gz] -v 1"
        #     )
        os.system(f"antsRegistration -d 3 -m Demons[{img_dir}{fix}_{mov}/{fix}_dtype.nii.gz, {img_dir}{fix}_{mov}/{mov}_dtype.nii.gz, 1, 4] " \
            f"-t SyN[0.1, 3, 0] -f 2x1 -s 1x0vox -u 0 " \
            f"-c [10x5] -o [{out_path}movingToFixed, {out_path}movingToFixedDeformed.nii.gz] -v 1"
            )
    else:
        # os.system(f"antsRegistration -d 3 -m Demons[{img_dir}{p}_{mov}_{fix}/{fix}_dtype.nii.gz, {img_dir}{p}_{mov}_{fix}/{mov}_dtype.nii.gz, 1, 4] " \
        #     f"-t SyN[0.1, 3, 0] -f 2x1 -s 1x0vox -u 0 " \
        #     f"-c [10x5] -o [{out_path}movingToFixed, {out_path}movingToFixedDeformed.nii.gz] -v 1"
        #     )
        os.system(f"antsRegistration -d 3 -m Demons[{img_dir}{mov}_{fix}/{fix}_dtype.nii.gz, {img_dir}{mov}_{fix}/{mov}_dtype.nii.gz, 1, 4] " \
            f"-t SyN[0.1, 3, 0] -f 2x1 -s 1x0vox -u 0 " \
            f"-c [10x5] -o [{out_path}movingToFixed, {out_path}movingToFixedDeformed.nii.gz] -v 1"
            )
    os.system(f"CreateJacobianDeterminantImage 3 {out_path}movingToFixed0Warp.nii.gz {out_path}jacobian.nii.gz 0 1")
    os.system(f"CreateJacobianDeterminantImage 3 {out_path}movingToFixed0Warp.nii.gz {out_path}logJacobian.nii.gz 1 1")

    end = timer()
    print('TIME TAKEN:', timedelta(seconds=end-start))

    return interval, mov, fix

def run_intra_reg_per_pair(p, mov, fix, img_dir, img_dir_out):
    start = timer()
    out_path = f"{img_dir_out}{mov}_{fix}/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    os.system(f"antsRegistration -d 3 -m Demons[{img_dir}{p}_{fix}_{mov}/{fix}_dtype.nii.gz, {img_dir}{p}_{fix}_{mov}/{mov}_dtype.nii.gz, 1, 4] " \
        f"-t SyN[0.1, 3, 0] -f 2x1 -s 1x0vox -u 0 " \
        f"-c [10x5] -o [{out_path}movingToFixed, {out_path}movingToFixedDeformed.nii.gz] -v 1"
        )
    os.system(f"CreateJacobianDeterminantImage 3 {out_path}movingToFixed0Warp.nii.gz {out_path}jacobian.nii.gz 0 1")
    os.system(f"CreateJacobianDeterminantImage 3 {out_path}movingToFixed0Warp.nii.gz {out_path}logJacobian.nii.gz 1 1")
    end = timer()
    print('TIME TAKEN:', timedelta(seconds=end-start))

def run_intra_rigid_reg_per_pair(p, mov, fix, img_dir, img_dir_out):
    start = timer()
    out_path = f"{img_dir_out}{mov}_{fix}/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    

    end = timer()
    print('TIME TAKEN:', timedelta(seconds=end-start))

def run_ants_intra_syn_reg(pair, img_dir_fix, img_dir_mov, img_dir_out):
    start = timer()
    p, mov, age_mov = pair[0]
    p, fix, age_fix = pair[1]

    # Ensure mov2fix is from younger subject to older
    if age_mov > age_fix:
        mov, fix = fix, mov
        age_mov, age_fix = age_fix, age_mov
        print(mov, fix, 'age_mov', age_mov, 'age_fix', age_fix, 'Age mov < Age fix')

    interval = age_fix-age_mov
    out_path = f"{img_dir_out}/{mov}_{fix}/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    os.system(f"antsRegistration -d 3 -m Demons[{img_dir_fix}{p}/{fix}/wf/brainextraction/{fix}_dtype.nii.gz, {img_dir_mov}{mov}_{fix}/mov2fix_warped_image.nii.gz, 1, 4] " \
    f"-t SyN[0.1, 3, 0] -f 2x1 -s 1x0vox -u 0 " \
    f"-c [10x5] -o [{out_path}movingToFixed, {out_path}movingToFixedDeformed.nii.gz] -v 1"
    )
    
    os.system(f"CreateJacobianDeterminantImage 3 {out_path}movingToFixed0Warp.nii.gz {out_path}jacobian.nii.gz 0 1")
    os.system(f"CreateJacobianDeterminantImage 3 {out_path}movingToFixed0Warp.nii.gz {out_path}logJacobian.nii.gz 1 1")

    end = timer()
    print('TIME TAKEN:', timedelta(seconds=end-start))

    return interval, mov, fix

def run_ants_inter_reg(pair, img_dir_fix, img_dir_mov, img_dir_out):
    start = timer()
    p_mov, mov, age_mov, _ = pair[0]
    p_fix, fix, age_fix, _ = pair[1]

    # Ensure mov2fix is from younger subject to older
    if age_mov > age_fix:
        mov, fix = fix, mov
        age_mov, age_fix = age_fix, age_mov
        p_mov, p_fix = p_fix, p_mov

    interval = age_fix-age_mov
    out_path = f"{img_dir_out}{mov}_{fix}/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # For rigid-inter
    os.system(f"antsRegistration -d 3 -m Demons[{img_dir_fix}{mov}_{fix}/{fix}_dtype.nii.gz, {img_dir_mov}{mov}_{fix}/{mov}_dtype.nii.gz, 1, 4] " \
        f"-t SyN[0.1, 3, 0] -f 2x1 -s 1x0vox -u 0 " \
        f"-c [10x5] -o [{out_path}movingToFixed, {out_path}movingToFixedDeformed.nii.gz] -v 1"
        )
    # For rigid-affine-inter
    # os.system(f"antsRegistration -d 3 -m Demons[{img_dir_fix}{p_fix}/{fix}/wf/brainextraction/{fix}_dtype.nii.gz, {img_dir_mov}{mov}_{fix}/mov2fix_warped_image.nii.gz, 1, 4] " \
    #     f"-t SyN[0.1, 3, 0] -f 2x1 -s 1x0vox -u 0 " \
    #     f"-c [10x5] -o [{out_path}movingToFixed, {out_path}movingToFixedDeformed.nii.gz] -v 1"
    #     )
    os.system(f"CreateJacobianDeterminantImage 3 {out_path}movingToFixed0Warp.nii.gz {out_path}jacobian.nii.gz 0 1")
    os.system(f"CreateJacobianDeterminantImage 3 {out_path}movingToFixed0Warp.nii.gz {out_path}logJacobian.nii.gz 1 1")

    end = timer()
    print('TIME TAKEN:', timedelta(seconds=end-start))

    return interval, mov, fix

def create_ddf_file(pair, transfo, img_dir, img_dir_out):
    """
    Create a deformation field (DDF) using ANTs registration.

    This function generates a deformation field (DDF) by applying a transformation to a moving image using the ANTs software.

    Parameters:
        pair (list): A list containing tuples of image data. Each tuple contains participant_id, image_filename, and age for both moving and fixed images.
        transfo (str): The transformation applied to the images (not explicitly used in the function).
        img_dir (str): The directory path where the image files are stored.
        img_dir_out (str): The output directory where the generated DDF and related files will be stored.

    Example:
        >>> pair = [('sub-001', 'PS14_001', 2.5), ('sub-002', 'PS14_002', 3.0)]
        >>> transfo = 'transform_file.h5'
        >>> img_dir = '/path/to/image/directory/'
        >>> img_dir_out = '/path/to/output/directory/'
        >>> create_ddf_file(pair, transfo, img_dir, img_dir_out)
    """
    p_mov, mov, age_mov = pair[0]
    p_fix, fix, age_fix = pair[1]

    os.system(f"antsApplyTransforms -d 3 -o [{img_dir_out}{mov}_{fix}/ddf_ANTs.nii.gz, 1] -v 1 -t {img_dir_out}{mov}_{fix}/mov2fix_Composite.h5 " \
                f"-r  {img_dir}{p_fix}/{fix}.nii.gz")
            
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
            data.append((participant_id, scan_id, float(age), int(sex)))
    return data

def find_pairs_per_age(data, type):
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
            participant_id_1, scan_id_1, age_1, _ = data[i]
            participant_id_2, scan_id_2, age_2, _ = data[j]

            if type == 'intra':
                if participant_id_1 == participant_id_2:
                    if age_1 < age_2:
                        pairs.append(((participant_id_1, scan_id_1, age_1), (participant_id_2, scan_id_2, age_2)))
                    else:
                        pairs.append(((participant_id_2, scan_id_2, age_2), (participant_id_1, scan_id_1, age_1)))
            else:
                if participant_id_1 != participant_id_2 and abs(age_1 - age_2) < 0.025:
                    pairs.append(((participant_id_1, scan_id_1, age_1), (participant_id_2, scan_id_2, age_2)))
    
    return pairs

def find_inter_pairs_per_sex(data, sex):
    """
    Find pairs of participants with age difference less than 0.5 years and with same sex pairs.

    This function takes a list of tuples containing participant_id, scan_id, and age data.
    It iterates through the data and finds all unique pairs of participants whose age difference is less than 0.5 years.

    Parameters:
        data (list): A list of tuples, where each tuple contains the participant_id (str), scan_id (str), and age (float) data.
        sex (int)

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
            participant_id_1, scan_id_1, age_1, sex_1 = data[i]
            participant_id_2, scan_id_2, age_2, sex_2 = data[j]

            if participant_id_1 != participant_id_2 and abs(age_1 - age_2) < 0.025 and sex_1 == sex_2:
                if sex_1 == sex:
                    pairs.append(((participant_id_1, scan_id_1, age_1), (participant_id_2, scan_id_2, age_2)))
    
    return pairs

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

    elif len(idx) == 4 and '-' not in idx[3]:
        mov = f'{idx[0]}_{idx[1]}'
        fix = f'{idx[2]}_{idx[3]}'
        return(mov, fix)

    elif len(idx) == 4 and '-' in idx[3]:
        mov = f'{idx[0]}_{idx[1]}_{idx[2]}'
        fix = f'{idx[3]}'
        return(mov, fix)

    elif len(idx) == 3 and '-' not in idx[2]:
        mov = f'{idx[0]}'
        fix = f'{idx[1]}_{idx[2]}'
        return(mov, fix)

    elif len(idx) == 3 and '-' in idx[2]:
        mov = f'{idx[0]}_{idx[1]}'
        fix = f'{idx[2]}'
        return(mov, fix)
    
    elif len(idx) == 2 and '-' in idx[1]:
        mov = f'{idx[0]}'
        fix = f'{idx[1]}'
        return(mov, fix)
    
    else:
        print('Not a corresponding folder name', name)

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
            if (scan_id_1 in pair[0] and scan_id_2 in pair[1]) or (scan_id_1 in pair[1] and scan_id_2 in pair[0]):
                result_pairs.append(pair)
    return result_pairs

def find_folders_without_file(root_path, file_to_check):
    folders_without_file = []

    for foldername, subfolders, filenames in os.walk(root_path):
        # Extract scan_id from the foldername
        folder_scan_id = os.path.basename(foldername)
        
        if file_to_check not in filenames:
            folders_without_file.append(folder_scan_id)
    
    # Create pairs of scan_id_1 and scan_id_2 from the list of folders
    scan_id_pairs = [(find_scan_ids(s)) for s in folders_without_file[1:]]

    return scan_id_pairs

def find_pairs_with_matching_distribution(data, target_distribution):
    pairs = []
    n_pairs = len(target_distribution)
    
    for i in range(n_pairs):
        # Use random.choice to select pairs based on the target distribution
        selected_pair = random.choice(data)
        pairs.append(selected_pair)
    
    return pairs

def separate_intra_by_sex(folders_path, data_path, folders_dest, type):
    """
    Separate intra-subject pairs by sex. 1 is for male and 0 for female.
    type: ra or r
    """
    data = pd.read_csv(data_path, sep='\t')
    for folder in os.listdir(folders_path):
        if '.csv' not in folder:
            mov, fix = find_scan_ids(folder)
            row = data[data['scan_id'] == mov]

            # Check if a row is found
            if not row.empty:
                # Extract the sex value from the found row
                sex = row['sex'].iloc[0]
            else:
                print("Scan ID not found")

            if sex == 1:
                # Copy the logJacobian.nii.gz file in folder called intra_m_ra/r
                experiment_name = 'intra_m_ra' if type == 'ra' else 'intra_m_r'
                if not os.path.exists(f'{folders_dest}/{experiment_name}/{folder}/'):
                    os.makedirs(f'{folders_dest}/{experiment_name}/{folder}/')
                
                shutil.copy(f'{folders_path}{folder}/logJacobian.nii.gz', f'{folders_dest}/{experiment_name}/{folder}/logJacobian.nii.gz')
            else:
                experiment_name = 'intra_f_ra' if type == 'ra' else 'intra_f_r'
                if not os.path.exists(f'{folders_dest}/{experiment_name}/{folder}/'):
                    os.makedirs(f'{folders_dest}/{experiment_name}/{folder}/')
                shutil.copy(f'{folders_path}{folder}/logJacobian.nii.gz', f'{folders_dest}/{experiment_name}/{folder}/logJacobian.nii.gz')
        else:
            print('Not a folder, but a file')

def intra_count_by_sex(folders_path, data_path):
    data = pd.read_csv(data_path, sep='\t')
    
    current_participant_ids = []

    # Iterate through each row in the sorted DataFrame
    for folder in os.listdir(folders_path):
        mov, fix = find_scan_ids(folder)
        row = data[data['scan_id'] == mov]
        # Get the current participant ID
        participant_id = row['participant_id'].iloc[0]
        
        # If it's the first row or the participant ID changed, increment the counter
        if participant_id not in current_participant_ids:
            current_participant_ids.append(participant_id)
        
        # Store the current participant ID as the previous one
        prev_participant_id = participant_id
    print(f'There are {len(current_participant_ids)} subjects of the specified sex')
    print(current_participant_ids)

def find_count_inter_pairs_per_sex(pairs, sex):


    same_sex_pairs = []
    count = 0
    for pair in pairs:
        participant_id_1, scan_id_1, age_1, sex_1 = pair[0]
        participant_id_2, scan_id_2, age_2, sex_2 = pair[1]

        if sex_1 == sex and sex_2 == sex:
            count += 1
            same_sex_pairs.append(pair)

    
    print(count)
    return same_sex_pairs

def find_failed_rigid_pairs(csv_file, index):

    # Read the CSV file and filter pairs based on time in seconds
    pairs_less_than_15s = []
    with open(csv_file, newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for i, row in enumerate(reader):
            if i >= index:
                break
            
            pair, time_seconds, _ = row
            time_seconds_float = float(time_seconds.split(':')[1]) * 60 + float(time_seconds.split(':')[2])
            if time_seconds_float < 15:
                pairs_less_than_15s.append(pair)
    scan_id_pairs = [(find_scan_ids(s)) for s in pairs_less_than_15s]
    return scan_id_pairs
def mask_log_jacobian(log_JD_path, df_path, group):
    """
    Calculate the log jacobian of the deformation field and apply a mask to it.

    This function calculates the log jacobian of the deformation field and applies a mask to it.
    The mask is generated using the brain extracted images of the moving and fixed images.

    Parameters:
        log_JD_path (str): The path to the directory containing the log Jacobian maps and mask.
        df_path (str): The path to the directory containing the pairs information.

    Example:
        >>> log_JD_path = "/path/to/inter-subject-reg/"
        >>> df_path = "/path/to/intra-subject-reg/"
    """
    df= pd.read_csv(df_path)
    scan_id_1_list = []
    scan_id_2_list = []
    avg_log_jac_list = []
    age_interval_list = []
    init_age_list = []
    group_list = []

    # Iterate through the folders in the inter_path directory
    for foldername in os.listdir(log_JD_path):
        if '.csv' not in foldername:
            # Extract the scan_id from the foldername
            scan_id_1, scan_id_2 = find_scan_ids(foldername)
            print(scan_id_1, scan_id_2)
            # Filter the DataFrame based on the given scan_id_1 and scan_id_2
            filtered_df = df[(df['scan_id_1'] == scan_id_1) & (df['scan_id_2'] == scan_id_2)]

            # Check if there is a match
            if not filtered_df.empty:
                age_interval = filtered_df['age_interval'].iloc[0]
                init_age = filtered_df['init_age'].iloc[0]
            else:
                age_interval = df[(df['scan_id_1'] == scan_id_2) & (df['scan_id_2'] == scan_id_1)]['age_interval'].iloc[0]
                init_age = df[(df['scan_id_1'] == scan_id_2) & (df['scan_id_2'] == scan_id_1)]['init_age'].iloc[0]
            # Check if the folder contains the logJacobian.nii.gz file
            if 'logJacobian.nii.gz' in os.listdir(os.path.join(log_JD_path, foldername)):
                # Load the logJacobian.nii.gz file
                log_jacobian = nib.load(os.path.join(log_JD_path, foldername, 'logJacobian.nii.gz'))
                # Load the image brain mask
                mask = nib.load(os.path.join(log_JD_path, foldername, 'mask_trans.nii'))
                # Apply the transformed mask to the log jacobian
                log_jacobian_masked = log_jacobian.get_fdata() * mask.get_fdata()
                # Save the masked log jacobian to a new file
                # nib.save(nib.Nifti1Image(log_jacobian_masked, log_jacobian.affine), os.path.join(foldername, 'logJacobian_masked.nii.gz'))
                # Exclude zeros outside the mask
                masked_log_jacobian_values = log_jacobian_masked[mask != 0]

                # Calculate the average absolute log Jacobian value on the masked areas
                avg_log_jac = np.mean(np.abs(masked_log_jacobian_values))

                scan_id_1_list.append(scan_id_1)
                scan_id_2_list.append(scan_id_2)
                avg_log_jac_list.append(avg_log_jac)
                age_interval_list.append(age_interval)
                init_age_list.append(init_age)
                group_list.append(group)

    return scan_id_1_list, scan_id_2_list, avg_log_jac_list, age_interval_list, init_age_list, group_list

def calculate_log_jacobian_masked(inter_path, intra_path, df_path):
    """
    Generates a dataframe for both intra and inter log Jacobian values

    """
    # Calculate the log Jacobian values for the intra-subject registration
    scan_id_1_list_intra, scan_id_2_list_intra, avg_log_jac_list_intra, age_interval_list_intra, init_age_list_intra, group_list_intra = mask_log_jacobian(intra_path, df_path, 'intra')

    # Calculate the log Jacobian values for the inter-subject registration
    scan_id_1_list_inter, scan_id_2_list_inter, avg_log_jac_list_inter, age_interval_list_inter, init_age_list_inter, group_list_inter = mask_log_jacobian(inter_path, df_path, 'inter')

    # Create a DataFrame from the collected lists
    avg_log_jac_df = pd.DataFrame({
        'scan_id_1': scan_id_1_list_intra + scan_id_1_list_inter,
        'scan_id_2': scan_id_2_list_intra + scan_id_2_list_inter,
        'avg_log_jac': avg_log_jac_list_intra + avg_log_jac_list_inter,
        'age_interval': age_interval_list_intra + age_interval_list_inter,
        'init_age': init_age_list_intra + init_age_list_inter,
        'group': group_list_intra + group_list_inter
    })

    mean_avg_log_jac = avg_log_jac_df['avg_log_jac'].mean()
    std_avg_log_jac = avg_log_jac_df['avg_log_jac'].std()

    print("Mean of avg_log_jac:", mean_avg_log_jac)
    print("Standard deviation of avg_log_jac:", std_avg_log_jac)

if __name__ == "__main__":

    tsv_file_path = "all-participants.tsv"
    
    data = read_tsv_file(tsv_file_path)

    # # # # # Separate intra-sub by sex # # # # #
    # folders_path = "/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/intra_mix_ra/intra/"
    # folders_dest = "/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs"
    # separate_intra_by_sex(folders_path, tsv_file_path, folders_dest, 'ra')

    # folders_path = "/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/intra_f_ra/"
    # intra_count_by_sex(folders_path, tsv_file_path)

    # Separate inter-sub by sex
    # sexes = [0, 1]
    # for sex in sexes:
    #     pairs = find_inter_pairs_per_sex(data, sex)
    #     print(f'There are {len(pairs)} pairs of sex {sex}')

    # Separate inter-sub by age interval and initial age
    # selected_inter_pairs = find_inter_pairs_with_matching_distribution_init(data, n_bins=25)
    # print(f'There are {len(selected_inter_pairs)} pairs with matching distribution')
    # print(selected_inter_pairs[:5])
    # And again by sex
    # for sex in sexes:
    #     selected_sex_pairs = find_inter_pairs_with_matching_distribution_init(data, 25, sex, same_sex=True)
    # #     pairs_per_sex = find_count_inter_pairs_per_sex(selected_inter_pairs, sex)
    #     print(f'There are {len(selected_sex_pairs)} pairs of sex {sex}')

    # # # # # Skull stripping # # # # #
    # skull_strip_non_longitudinal(tsv_file_path)

    # # # # # Paths & Transformations # # # # #
    # # transfo ='rigid_inter'
    # # transfo = 'affine_inter'
    # transfo = 'rigid_affine_inter'
    # init_transfo = 'rigid_inter'
    # img_dir = '/media/andjela/SeagatePor/PairRegData/images/'
    # img_dir_out = f'/media/andjela/SeagatePor/PairReg/{transfo}/images/'
    # img_dir_init_transfo = f'/media/andjela/SeagatePor/PairReg/{init_transfo}/images/'

    # img_dir = "/media/andjela/SeagatePor/PairReg/rigid/images/"
    # img_dir_out = '/home/andjela/Documents/intra-inter-ddfs/intra/'

    # # # # # Extract pairs # # # # #
    # type = 'intra'
    # pairs = find_pairs(data, type)
    # print(len(pairs))

    # # # # # Correction of intra-subject-reg # # # # #
    # run_intra_reg_per_pair(p= '10117', mov= 'PS15_070', fix= 'CL_Dev_015', img_dir = img_dir, img_dir_out = img_dir_out)

    # all_failed_pairs = [('10117', 'PS15_070', 'CL_Dev_012'), ('10117', 'PS15_070', 'CL_Dev_012'), ('10117', 'PS15_070', 'CL_Dev_010'), 
    #                     ('10117', 'PS15_070', 'CL_Dev_004'), ('10117', 'PS15_070', 'CL_Dev_007'), ('10117', 'PS15_070', 'CL_Dev_005'),
    #                     ('10117', 'PS15_070', 'CL_Dev_017'), ('10117', 'PS15_081', 'CL_Dev_015'), ('10117', 'PS15_081', 'CL_Dev_012'),
    #                     ('10117', 'PS15_081', 'CL_Dev_004'), ('10117', 'PS15_070', 'CL_Dev_008'), ('10117', 'PS15_081', 'CL_Dev_008'),
    #                     ('10117', 'PS15_081', 'CL_Dev_010'), ('10117', 'PS15_081', 'CL_Dev_017'), ('10117', 'PS15_081', 'CL_Dev_005'),
    #                     ('10117', 'PS15_081', 'CL_Dev_007'), ('10136', 'CL_Dev_009', 'CL_DEV_023'), ('10136', 'CL_Dev_018', 'CL_DEV_023'),
    #                     ('10136', 'CL_Dev_016', 'CL_DEV_023'), ('10136', 'CL_Dev_011', 'CL_DEV_023')]
    
    # for pair in all_failed_pairs:
    #     run_intra_reg_per_pair(p= pair[0], mov= pair[1], fix= pair[2], img_dir = img_dir, img_dir_out = img_dir_out)

    # root_path = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/intra_mix_ra/intra/'
    # file_to_check = 'logJacobian.nii.gz'
    # folders_to_redo = find_folders_without_file(root_path, file_to_check)
    # pairs_to_redo = find_pairs_by_scan_ids(folders_to_redo, pairs)
    # filtered_pairs_to_redo = [item for item in pairs_to_redo if item[0][0] != '10136']

    # # # # # For rigid only # # # # #
    # img_dir = '/media/andjela/SeagatePor/work_dir/reg_n4_wdir/'
    # img_dir_out = '/home/andjela/joplin-intra-inter/rigid_intra/'
    # img_dir_init_transfo = ''
    # transfo = 'rigid'
    # for pair in filtered_pairs_to_redo:
    #     run_ants_registration(pair, transfo, img_dir, img_dir_out, img_dir_init_transfo)

    # # # # # For rigid-affine # # # # #
    #img_dir = '/media/andjela/SeagatePor/work_dir/reg_n4_wdir/'
    #img_dir_out = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/rigid_affine_corr/images/'
    #img_dir_init_transfo = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/rigid_intra/'
    #transfo = 'rigid_affine_intra'
    #for pair in pairs_to_redo:
    #    run_ants_registration(pair, transfo, img_dir, img_dir_out, img_dir_init_transfo)

    # # # # # For inter_reg # # # # #
    # img_dir_fix = '/media/andjela/SeagatePor/work_dir2/cbf2mni_wdir/'
    # img_dir_mov = '/media/andjela/SeagatePor/PairReg/rigid_inter/images/'
    # img_dir_out = '/home/andjela/Documents/intra-inter-ddfs/inter/'
    # type = 'inter'
    # pairs = find_pairs(data, type)
    # print(len(pairs))

    # # Inter reg on pairs with matching init age, age interval and sex (ias) # #
    # img_dir = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/work_dir/reg_n4_wdir/'
    # img_dir_out = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/inter_ias_r/'
    # img_dir_init_transfo = ''
    # transfo = 'rigid'

    # selected_ias_pairs = find_inter_pairs_with_matching_distribution_init(data, 25, same_sex=True)
    # print('IAS', len(selected_ias_pairs))

    # df_inter_ias_r = pd.DataFrame([(pair[0][1], pair[1][1], min(pair[0][2], pair[1][2]), abs(pair[0][2] - pair[1][2]), [pair[0][3], pair[1][3]], 'inter') for pair in selected_ias_pairs],
    #                   columns=['scan_id_1', 'scan_id_2', 'init_age', 'age_interval', 'sex', 'group'])

    # # Concatenate the two DataFrames
    # df_ias = pd.concat([df_intra, df_inter_ias_r], ignore_index=True)
    # print(df_ias)
    # df_ias.to_csv('C:\\Users\\andje\\Downloads\\pairs_ias_r.csv')

    # scan_to_sub = scan_id_to_sub_id(tsv_file_path)
    # scan_to_session = scan_id_to_session(tsv_file_path)

    # scan_ids_to_find = [("PS16_002", "PS17_024"), ("PS16_054", "PS17_007"), ("PS0322-10-2", "PS17_007"), ("PS1477-10-1", "PS17_007"), ("PS16_006", "PS17_007")]
    # inter_last_pairs_to_redo = find_pairs_by_scan_ids(scan_ids_to_find, selected_ias_pairs)
    # # print(inter_last_pairs_to_redo)
    # for pair in inter_last_pairs_to_redo:
    #     run_ants_registration(pair, transfo, img_dir, img_dir_out, img_dir_init_transfo, scan_to_sub, scan_to_session)
    
    # # # # TESTING A PAIR
    # tryout_pair = selected_ias_pairs[0]
    # run_ants_registration(tryout_pair, transfo, img_dir, img_dir_out, img_dir_init_transfo, scan_to_sub, scan_to_session)

    # # # SAVING RESULTS
    # csv_output_path = f'{img_dir_out}timing_results.csv'
    # if not os.path.exists(csv_output_path):
    #     with open(csv_output_path, 'w', newline='') as csvfile:
    #         csv_writer = csv.writer(csvfile)
    #         csv_writer.writerow(['Pair (mov_fix)', 'Time (seconds)', 'Age interval (fix-mov)'])

    # with open(csv_output_path, 'a', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     for pair in selected_ias_pairs:
    #         start = timer()
    #         interval, mov, fix = run_ants_registration(pair, transfo, img_dir, img_dir_out, img_dir_init_transfo, scan_to_sub, scan_to_session)
    #         end = timer()
    #         time_taken = timedelta(seconds=end-start)
    #         csv_writer.writerow([f'{mov}_{fix}', time_taken, interval])

    # csv_file = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/inter_ias_r/timing_results.csv'
    # # csv_file = 'C:\\Users\\andje\\Downloads\\timing_results_inter_ias.csv'
    # failed_inter_rigid_pairs = find_failed_rigid_pairs(csv_file, 432)
    
    
    # inter_pairs_to_redo = find_pairs_by_scan_ids(failed_inter_rigid_pairs, selected_ias_pairs)
    
    # tryout_pair = inter_pairs_to_redo[0]
    # for pair in inter_pairs_to_redo:
    #     run_ants_registration(pair, transfo, img_dir, img_dir_out, img_dir_init_transfo, scan_to_sub, scan_to_session)


    # # Inter reg on pairs with matching init age, age interval (ia) # #
    # img_dir_out = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/inter_ia_r/'

    # selected_ia_pairs = find_inter_pairs_with_matching_distribution_init(data, 25, same_sex=False)
    # print('IA', len(selected_ia_pairs))
    
    # df_inter_ia_r = pd.DataFrame([(pair[0][1], pair[1][1], min(pair[0][2], pair[1][2]), abs(pair[0][2] - pair[1][2]), [pair[0][3], pair[1][3]], 'inter') for pair in selected_ia_pairs],
    #                   columns=['scan_id_1', 'scan_id_2', 'init_age', 'age_interval', 'sex', 'group'])

    # # Concatenate the two DataFrames
    # df_ia = pd.concat([df_intra, df_inter_ia_r])
    # print(df_ia)
    # df_ia.to_csv('C:\\Users\\andje\\Downloads\\pairs_ia_r.csv')

    # # Convert lists to sets and find their intersection
    # common_elements = set(selected_ias_pairs).intersection(set(selected_ia_pairs))

    # # Convert the result back to a list if needed
    # common_elements_list = list(common_elements)
    # print(common_elements_list)
    # print(len(selected_ia_pairs), len(selected_ias_pairs))

    # csv_file = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/inter_ia_r/timing_results.csv'
    # csv_file = 'C:\\Users\\andje\\Downloads\\timing_results_inter_ias.csv'
    # failed_inter_rigid_pairs = find_failed_rigid_pairs(csv_file, 433)
    # csv_file = 'C:\\Users\\andje\\Downloads\\timing_results_inter_ia.csv'
    # failed_inter_rigid_pairs = find_failed_rigid_pairs(csv_file, 433)
    # print(len(failed_inter_rigid_pairs))
    # print(failed_inter_rigid_pairs)
    
    
    # inter_pairs_to_redo = find_pairs_by_scan_ids(failed_inter_rigid_pairs, selected_ia_pairs)
    # print(len(inter_pairs_to_redo))
    # print(inter_pairs_to_redo)
    # print(len(selected_ia_cd pairs))

    # for count, pair in enumerate(inter_pairs_to_redo):
    #     print('ITERATION: ', count)
    #     run_ants_registration(pair, transfo, img_dir, img_dir_out, img_dir_init_transfo, scan_to_sub, scan_to_session)

    # # # SAVING RESULTS
    # csv_output_path = f'{img_dir_out}timing_results.csv'
    # if not os.path.exists(csv_output_path):
    #     with open(csv_output_path, 'w', newline='') as csvfile:
    #         csv_writer = csv.writer(csvfile)
    #         csv_writer.writerow(['Pair (mov_fix)', 'Time (seconds)', 'Age interval (fix-mov)'])

    # with open(csv_output_path, 'a', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     for pair in selected_ia_pairs:
    #         start = timer()
    #         interval, mov, fix = run_ants_registration(pair, transfo, img_dir, img_dir_out, img_dir_init_transfo, scan_to_sub, scan_to_session)
    #         end = timer()
    #         time_taken = timedelta(seconds=end-start)
    #         csv_writer.writerow([f'{mov}_{fix}', time_taken, interval])

    # # # # # For SyN # # # # #
    # img_dir_fix = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/work_dir2/cbf2mni_wdir/'
    # img_dir_mov = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/rigid_affine_corr/images/'
    # img_dir_out = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/intra_mix_ra/intra'

    # csv_output_path = f'{img_dir_out}timing_results.csv'
    # if not os.path.exists(csv_output_path):
    #     with open(csv_output_path, 'w', newline='') as csvfile:
    #         csv_writer = csv.writer(csvfile)
    #         csv_writer.writerow(['Pair (mov_fix)', 'Time (seconds)', 'Age interval (fix-mov)'])

    # with open(csv_output_path, 'a', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     for pair in pairs_to_redo:
    #         start = timer()
    #         interval, mov, fix = run_ants_intra_syn_reg(pair, img_dir_fix, img_dir_mov, img_dir_out)
    #         end = timer()
    #         time_taken = timedelta(seconds=end-start)
    #         csv_writer.writerow([f'{mov}_{fix}', time_taken, interval])

    
    # # # # # For inter_ias_r # # # # #
    # img_dir_fix = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/inter_ias_r/'
    # img_dir_mov = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/inter_ias_r/'
    # img_dir_out = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/inter_ias_r/'

    # for pair in inter_pairs_to_redo:
    #     run_ants_inter_reg(pair, img_dir_fix, img_dir_mov, img_dir_out)

    # scan_ids_to_find = [("PS16_002", "PS17_024"), ("PS16_054", "PS17_007"), ("PS0322-10-2", "PS17_007"), ("PS1477-10-1", "PS17_007"), ("PS16_006", "PS17_007")]
    # inter_last_pairs_to_redo = find_pairs_by_scan_ids(scan_ids_to_find, selected_ias_pairs)
    # # print(inter_last_pairs_to_redo)
    # for pair in inter_last_pairs_to_redo:
    #     # run_ants_registration(pair, transfo, img_dir, img_dir_out, img_dir_init_transfo, scan_to_sub, scan_to_session)
    #     run_ants_inter_reg(pair, img_dir_fix, img_dir_mov, img_dir_out)

    # # # TRYOUT A PAIR # #
    # # tryout_pair = selected_ias_pairs[0]
    # # run_ants_inter_reg(tryout_pair, img_dir_fix, img_dir_mov, img_dir_out)

    # csv_output_path = f'{img_dir_out}timing_results.csv'
    # if not os.path.exists(csv_output_path):
    #     with open(csv_output_path, 'w', newline='') as csvfile:
    #         csv_writer = csv.writer(csvfile)
    #         csv_writer.writerow(['Pair (mov_fix)', 'Time (seconds)', 'Age interval (fix-mov)'])

    # with open(csv_output_path, 'a', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     for pair in selected_ias_pairs:
    #         start = timer()
    #         interval, mov, fix = run_ants_inter_reg(pair, img_dir_fix, img_dir_mov, img_dir_out)
    #         end = timer()
    #         time_taken = timedelta(seconds=end-start)
    #         csv_writer.writerow([f'{mov}_{fix}', time_taken, interval])

    # # # # # For inter_ia_r # # # # #
    # img_dir_fix = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/inter_ia_r/'
    # img_dir_mov = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/inter_ia_r/'
    # img_dir_out = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/inter_ia_r/'

    # file_to_check = 'logJacobian.nii.gz'
    # folders_to_redo = find_folders_without_file(img_dir_fix, file_to_check)
    # pairs_to_redo = find_pairs_by_scan_ids(folders_to_redo, selected_ia_pairs)
    # # print(len(pairs_to_redo))
    # for count, pair in enumerate(pairs_to_redo):
    #     print('ITERATION: ', count)
    #     run_ants_inter_reg(pair, img_dir_fix, img_dir_mov, img_dir_out)

    # for count, pair in enumerate(inter_pairs_to_redo):
    #     print('ITERATION: ', count)
    #     run_ants_inter_reg(pair, img_dir_fix, img_dir_mov, img_dir_out)

    #     # Get the current date and time
    #     current_time = datetime.now()

    #     # Check if the current time meets a condition
    #     if current_time.hour == 1 and current_time.minute == 58:
    #         # If the condition is met, write a certain variable to a text file
    #         with open("output_inter_ia_r.txt", "w") as file:
    #             file.write(count, pair)
    #         break

    # with open('output_inter_ia_r.txt', 'r') as file:
    #     # Read the first line and split it into two values
    #     values = file.readline().split()
    #     # Assign the values to variables
    #     count = int(values[0])
    #     pair = values[1]
    
    # for pair in inter_pairs_to_redo[count:]:
    #     run_ants_inter_reg(pair, img_dir_fix, img_dir_mov, img_dir_out)

    # csv_output_path = f'{img_dir_out}timing_results.csv'
    # if not os.path.exists(csv_output_path):
    #     with open(csv_output_path, 'w', newline='') as csvfile:
    #         csv_writer = csv.writer(csvfile)
    #         csv_writer.writerow(['Pair (mov_fix)', 'Time (seconds)', 'Age interval (fix-mov)'])

    # with open(csv_output_path, 'a', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     for pair in selected_ia_pairs:
    #         start = timer()
    #         interval, mov, fix = run_ants_inter_reg(pair, img_dir_fix, img_dir_mov, img_dir_out)
    #         end = timer()
    #         time_taken = timedelta(seconds=end-start)
    #         csv_writer.writerow([f'{mov}_{fix}', time_taken, interval])

    # # # # # For inter_affine_reg # # # # #
    # img_dir_fix = '/media/andjela/SeagatePor/work_dir2/cbf2mni_wdir/'
    
    # img_dir_mov = '/media/andjela/SeagatePor/PairReg/rigid_affine_inter/images/'
    # img_dir_out = '/home/andjela/Documents/intra-inter-ddfs/inter_mix_ra/'
        
    # img_dir = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/rigid_affine_corr/images/'
    # img_dir_out = '/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/intra_mix_ra/'
    
    # type = 'intra'
    # pairs = find_pairs(data, type)
    # print(len(pairs))
    
    # # TESTING A PAIR   
    # tryout_pair = pairs[0]
    # print(tryout_pair)
    # run_ants_intra_reg(tryout_pair, img_dir, img_dir_out)
    # run_ants_registration(tryout_pair, transfo, img_dir, img_dir_out, img_dir_init_transfo)
    # create_ddf_file(tryout_pair, transfo, img_dir, img_dir_out)

    # # # SAVING RESULTS
    # csv_output_path = f'{img_dir_out}timing_results.csv'
    # if not os.path.exists(csv_output_path):
    #     with open(csv_output_path, 'w', newline='') as csvfile:
    #         csv_writer = csv.writer(csvfile)
    #         csv_writer.writerow(['Pair (mov_fix)', 'Time (seconds)', 'Age interval (fix-mov)'])

    # with open(csv_output_path, 'a', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     for pair in pairs:
    #         start = timer()
    #         # interval, mov, fix = run_ants_registration(pair, transfo, img_dir, img_dir_out, img_dir_init_transfo)
    #         interval, mov, fix = run_ants_intra_reg(pair, img_dir, img_dir_out)
    #         # interval, mov, fix = run_ants_inter_reg(pair, img_dir_fix, img_dir_mov, img_dir_out)
    #         end = timer()
    #         time_taken = timedelta(seconds=end-start)
    #         csv_writer.writerow([f'{mov}_{fix}', time_taken, interval])

    # # # # # Calculate Log Jacobians # # # # #
    # inter_path = "/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/inter_ia_r/"
    # intra_path = "/home/andim/intra-inter-ddfs/intra_mix_r/"
    # df_path = "/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/pairs_ia_r.csv"
    # calculate_log_jacobian_masked(inter_path, intra_path, df_path)
    
