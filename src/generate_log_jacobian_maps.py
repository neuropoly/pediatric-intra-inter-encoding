# previous file name: inter_reg.py
import os
from collections import defaultdict
from timeit import default_timer as timer
from datetime import timedelta
import pandas as pd
import csv
import numpy as np
import random
import shutil

def subj_to_scan_id():
    p_file = "PatientDict.txt"
    mapping = {} #defaultdict(list)
    # associates list of scanID (value) to key (patient number)
    
    with open(p_file, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace and newline characters
            key, value = line.split('\t')
            if key in mapping:
                mapping[key].append(value)
            else:
                mapping[key] = [value]

    return mapping

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
            participant_id_1, scan_id_1, age_1 = data[i]
            participant_id_2, scan_id_2, age_2 = data[j]

            if participant_id_1 == participant_id_2:
                pairs.append(((participant_id_1, scan_id_1, age_1), (participant_id_2, scan_id_2, age_2)))

    return pairs

def find_inter_pairs(data):
    pairs = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            participant_id_1, scan_id_1, age_1 = data[i]
            participant_id_2, scan_id_2, age_2 = data[j]

            
            if participant_id_1 != participant_id_2:
                pairs.append(((participant_id_1, scan_id_1, age_1), (participant_id_2, scan_id_2, age_2)))
    
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

def find_inter_pairs_with_matching_distribution_init(data, n_bins):
    # Calculate histogram distribution of intra_array
    intra_pairs_info = find_intra_pairs(data)
    intra_age_intervals = np.abs([item[0][2]-item[1][2] for item in intra_pairs_info])
    hist_distribution_intra_age_intervals, bin_edges_intra_age_intervals = np.histogram(intra_age_intervals, bins=n_bins) # Mean: 1.152, std: 0.684

    intra_init_ages = [min(item[0][2],item[1][2]) for item in intra_pairs_info]
    hist_distribution_intra_init_ages, bin_edges_intra_init_ages = np.histogram(intra_init_ages, bins=n_bins) # Mean: 1.152, std: 0.684


    # Known values for the new group
    inter_pairs_info = find_inter_pairs(data)
    inter_age_intervals = np.abs([item[0][2]-item[1][2] for item in inter_pairs_info])

    inter_init_ages = [min(item[0][2],item[1][2]) for item in inter_pairs_info]

    
    inter_pairs_to_keep = []
    # hist_distribution_inter_age_intervals = np.zeros(len(hist_distribution_intra_age_intervals))
    # hist_distribution_inter_init_ages = np.zeros(len(hist_distribution_intra_init_ages))
    # counter_intervals = 0
    # counter_init = 0
    for idx, intra_pair in enumerate(intra_pairs_info):
        limits_intervals, idx_bin_intervals  = find_closest_values_with_indices(intra_age_intervals[idx], bin_edges_intra_age_intervals)
        limits_init, idx_bin_init = find_closest_values_with_indices(intra_init_ages[idx], bin_edges_intra_init_ages)
        inter_set_age_intervals = set()
        inter_set_init_ages = set()
        # target_bin_count_intervals = hist_distribution_intra_age_intervals[idx_bin_intervals]
        # target_bin_count_init = hist_distribution_intra_init_ages[idx_bin_init]
        for idx, inter_pair in enumerate(inter_pairs_info):
            if inter_age_intervals[idx] >= limits_intervals[0] and inter_age_intervals[idx] < limits_intervals[1]:
                inter_set_age_intervals.add(inter_pair)
            if inter_init_ages[idx] >= limits_init[0] and inter_init_ages[idx] < limits_init[1]:
                inter_set_init_ages.add(inter_pair)
        
        # actual_bin_count_intervals = hist_distribution_inter_age_intervals[idx_bin_intervals]
        # actual_bin_count_init = hist_distribution_inter_init_ages[idx_bin_init]
        intersection = inter_set_age_intervals & inter_set_init_ages
        for element in intersection:

            if element not in inter_pairs_to_keep:
                inter_pairs_to_keep.append(element)

                break


    # for lim_inf_intervals, lim_sup_intervals, bin_count_intervals, lim_inf_init, lim_sup_init, bin_count_init in zip(bin_edges_intra_age_intervals, bin_edges_intra_age_intervals[1:], hist_distribution_intra_age_intervals, bin_edges_intra_init_ages, bin_edges_intra_init_ages[1:], hist_distribution_intra_init_ages):
    #     count_added = 0
    #     for idx, inter_pair in enumerate(inter_pairs_info):
    #         if inter_age_intervals[idx] >= lim_inf_intervals and inter_age_intervals[idx] < lim_sup_intervals:
    #             inter_set_age_intervals.add(inter_pair)
                
    #         if inter_init_ages[idx] >= lim_inf_init and inter_init_ages[idx] < lim_sup_init:
    #             inter_set_init_ages.add(inter_pair)
                
    #     intersection = inter_set_age_intervals & inter_set_init_ages
    #     for element in intersection:
    #         if element not in inter_pairs_to_keep and count_added <:
    #             count_added += 1
    #             inter_pairs_to_keep.append(element)
    #     counter_intervals = 0
    #     counter_init = 0

    inter_age_intervals_to_keep = np.abs([item[0][2]-item[1][2] for item in inter_pairs_to_keep])
    inter_init_ages_to_keep = [min(item[0][2],item[1][2]) for item in inter_pairs_to_keep]

    # plt.hist(inter_age_intervals_to_keep, range=(bin_edges_intra_age_intervals.min(), bin_edges_intra_age_intervals.max()), bins=n_bins, alpha=0.5, label='Inter Group')
    # plt.hist(intra_age_intervals, bins=n_bins, alpha=0.5, label='Intra Group')
    # plt.legend()
    # plt.show()

    # plt.hist(inter_init_ages_to_keep, range=(bin_edges_intra_init_ages.min(), bin_edges_intra_init_ages.max()), bins=n_bins, alpha=0.5, label='Inter Group')
    # plt.hist(intra_init_ages, bins=n_bins, alpha=0.5, label='Intra Group')
    # plt.legend()
    # plt.show()
        
    return inter_pairs_to_keep

    

def run_ants_registration(pair, transfo, img_dir, img_dir_out, img_dir_init_transfo):
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
    p_mov, mov, age_mov = pair[0]
    p_fix, fix, age_fix = pair[1]

    # Ensure mov2fix is from younger subject to older
    if age_mov > age_fix:
        mov, fix = fix, mov
        age_mov, age_fix = age_fix, age_mov
        p_mov, p_fix = p_fix, p_mov

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
        #f"--use-estimate-learning-rate-once 1 " \ (invalid flag given by ANTs)
        # os.system(f"antsRegistration --float --collapse-output-transforms 1 --dimensionality 3 " \
        #         f"--initial-moving-transform [ {img_dir}{p_fix}/{fix}.nii.gz, {img_dir}{p_mov}/{mov}.nii.gz, 0 ] " \
        #         f"--initialize-transforms-per-stage 0 " \
        #         f"--interpolation Linear " \
        #         f"--output [ {img_dir_out}{mov}_{fix}/mov2fix_, {img_dir_out}{mov}_{fix}/mov2fix_warped_image.nii.gz, {img_dir_out}{mov}_{fix}/fix2mov_warped_image.nii.gz ] " \
        #         f"--transform Rigid[0.1] " \
        #         f"--metric MI[ {img_dir}{p_fix}/{fix}.nii.gz, {img_dir}{p_mov}/{mov}.nii.gz, 1, 32, Regular, 0.3 ] " \
        #         f"--convergence [850x250x250,1e-7,25] " \
        #         f"--shrink-factors 4x2x1 " \
        #         f"--smoothing-sigmas 2x1x0vox " \
        #         f"--use-histogram-matching 1 " \
        #         f"--verbose 1 " \
        #         f"--transform Affine[0.1] " \
        #         f"--metric MI[ {img_dir}{p_fix}/{fix}.nii.gz, {img_dir}{p_mov}/{mov}.nii.gz, 1, 32, Regular, 0.3 ] " \
        #         f"--convergence [850x250x250, 1e-7, 25] " \
        #         f"--shrink-factors 4x2x1 " \
        #         f"--smoothing-sigmas 2x1x0vox " \
        #         f"--use-histogram-matching 1 " \
        #         f"--winsorize-image-intensities [0.005, 0.995] " \
        #         f"--write-composite-transform 1" \
        #         f"--verbose 1"
        #         )
        # warp subject calculated mov mask to fix
        # os.system(f"antsApplyTransforms --default-value 0 --float 0 " \
        #     f"--input /media/andjela/SeagatePor/work_dir2/cbf2mni_wdir/{p_mov}/{mov}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz " \
        #     f"--input-image-type 0 --interpolation Linear --output {img_dir_out}{mov}_{fix}/mask_trans.nii " \
        #     f"--reference-image {img_dir}{p_fix}/{fix}.nii.gz " \
        #     f"--transform {img_dir_out}{mov}_{fix}/mov2fix_Composite.h5")
        # #brain extraction for mov image which becomes mov2fix_warped
        # os.system(f'fslmaths {img_dir_out}{mov}_{fix}/mov2fix_warped_image.nii.gz -mul {img_dir_out}{mov}_{fix}/mask_trans.nii {img_dir_out}{mov}_{fix}/{mov}_dtype.nii.gz -odt float')
        # #brain extraction for fix image which stays fix
        # os.system(f'fslmaths {img_dir}{p_fix}/{fix}.nii.gz -mul /media/andjela/SeagatePor/work_dir2/cbf2mni_wdir/{p_fix}/{fix}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz {img_dir_out}{mov}_{fix}/{fix}_dtype.nii.gz -odt float')
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
        # # warp subject calculated mov mask to fix
        # os.system(f"antsApplyTransforms --default-value 0 --float 0 " \
        #     f"--input /media/andjela/SeagatePor/work_dir2/cbf2mni_wdir/{p_mov}/{mov}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz " \
        #     f"--input-image-type 0 --interpolation Linear --output {img_dir_out}{mov}_{fix}/mask_trans.nii " \
        #     f"--reference-image {img_dir}{p_fix}/{fix}.nii.gz " \
        #     f"--transform {img_dir_out}{mov}_{fix}/mov2fix_Composite.h5")
        # #brain extraction for mov image which becomes mov2fix_warped
        # os.system(f'fslmaths {img_dir_out}{mov}_{fix}/mov2fix_warped_image.nii.gz -mul {img_dir_out}{mov}_{fix}/mask_trans.nii {img_dir_out}{mov}_{fix}/{mov}_dtype.nii.gz -odt float')
        # #brain extraction for fix image which stays fix
        # os.system(f'fslmaths {img_dir}{p_fix}/{fix}.nii.gz -mul /media/andjela/SeagatePor/work_dir2/cbf2mni_wdir/{p_fix}/{fix}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz {img_dir_out}{mov}_{fix}/{fix}_dtype.nii.gz -odt float')
        end = timer()
        print('TIME TAKEN:', timedelta(seconds=end-start))
        return interval, mov, fix

    else:
        # Rigid Transfo
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
            f"--input /media/andjela/SeagatePor/work_dir2/cbf2mni_wdir/{p_mov}/{mov}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz " \
            f"--input-image-type 0 --interpolation NearestNeighbor --output {img_dir_out}{mov}_{fix}/mask_trans.nii " \
            f"--reference-image {img_dir}{p_fix}/{fix}/wf/n4/{fix}_corrected.nii.gz " \
            f"--transform {img_dir_out}{mov}_{fix}/mov2fix_Composite.h5")
        #brain extraction for mov image which becomes mov2fix_warped
        os.system(f'fslmaths {img_dir_out}{mov}_{fix}/mov2fix_warped_image.nii.gz -mul {img_dir_out}{mov}_{fix}/mask_trans.nii {img_dir_out}{mov}_{fix}/{mov}_dtype.nii.gz -odt float')
        #brain extraction for fix image which stays fix
        os.system(f'fslmaths {img_dir}{p_fix}/{fix}/wf/n4/{fix}_corrected.nii.gz -mul /media/andjela/SeagatePor/work_dir2/cbf2mni_wdir/{p_fix}/{fix}/wf/biniraze_mask/nihpd_asym_04.5-08.5_mask_trans_dtype.nii.gz {img_dir_out}{mov}_{fix}/{fix}_dtype.nii.gz -odt float')
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
    p_mov, mov, age_mov = pair[0]
    p_fix, fix, age_fix = pair[1]

    # Ensure mov2fix is from younger subject to older
    if age_mov > age_fix:
        mov, fix = fix, mov
        age_mov, age_fix = age_fix, age_mov
        p_mov, p_fix = p_fix, p_mov

    interval = age_fix-age_mov
    out_path = f"{img_dir_out}{mov}_{fix}/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # os.system(f"antsRegistration -d 3 -m Demons[{img_dir_fix}{p_fix}/{fix}/wf/brainextraction/{fix}_dtype.nii.gz, {img_dir_mov}{mov}_{fix}/{mov}_dtype.nii.gz, 1, 4] " \
    #     f"-t SyN[0.1, 3, 0] -f 2x1 -s 1x0vox -u 0 " \
    #     f"-c [10x5] -o [{out_path}movingToFixed, {out_path}movingToFixedDeformed.nii.gz] -v 1"
    #     )
    # For rigid-affine-inter
    os.system(f"antsRegistration -d 3 -m Demons[{img_dir_fix}{p_fix}/{fix}/wf/brainextraction/{fix}_dtype.nii.gz, {img_dir_mov}{mov}_{fix}/mov2fix_warped_image.nii.gz, 1, 4] " \
        f"-t SyN[0.1, 3, 0] -f 2x1 -s 1x0vox -u 0 " \
        f"-c [10x5] -o [{out_path}movingToFixed, {out_path}movingToFixedDeformed.nii.gz] -v 1"
        )
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

if __name__ == "__main__":
    # output_directory = "/home/andjela/Documents/intra-inter-ddfs/src/"
    # create_participant_file(output_directory)

    tsv_file_path = "participants.tsv"
    data = read_tsv_file(tsv_file_path)

    # Separate intra-sub by sex
    # folders_path = "/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/intra_mix_ra/intra/"
    # folders_dest = "/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs"
    # separate_intra_by_sex(folders_path, tsv_file_path, folders_dest, 'ra')

    # folders_path = "/home/GRAMES.POLYMTL.CA/andim/intra-inter-ddfs/intra_f_ra/"
    # intra_count_by_sex(folders_path, tsv_file_path)

    # Separate inter-sub by sex
    sex = 1
    pairs = find_inter_pairs_per_sex(data, sex)
    print(f'There are {len(pairs)} pairs of sex {sex}')

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

    
        
        

    # # # # # For inter_reg # # # # #
    # img_dir_fix = '/media/andjela/SeagatePor/work_dir2/cbf2mni_wdir/'
    # img_dir_mov = '/media/andjela/SeagatePor/PairReg/rigid_inter/images/'
    # img_dir_out = '/home/andjela/Documents/intra-inter-ddfs/inter/'
    # type = 'inter'
    # pairs = find_pairs(data, type)
    # print(len(pairs))

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
    
