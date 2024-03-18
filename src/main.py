from preprocessing import read_tsv_and_create_dict, read_tsv_and_create_dataframe, read_paths, extract_all_participant_ids, select_random_state, custom_split_indices
from train-and-test import .


def main(file_path, path_inter, path_intra, experiment_name):
    # Call the function to create the dictionary for all 64 patients
    participant_mapping = read_tsv_and_create_dict(file_path)

    # Obtain dataframe with subject, scan_id, age and sex
    full_df = read_tsv_and_create_dataframe(file_path)
    
    experiment_name_inter = f'inter_{experiment_name}'
    experiment_name_intra = f'intra_{experiment_name}'
    intra_scan_paths, nbr_intra_pairs, inter_scan_paths, nbr_inter_pairs = read_paths(path_intra, path_inter, experiment_name_intra, experiment_name_inter)

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

    
    train_no_overlap(experiment_name, intra_scan_paths, inter_scan_paths, all_sets)

    random_state = 42
    train_data, y_data = extract_train_data_overlap(intra_scan_paths, inter_scan_paths, nbr_intra_pairs, total_indices, random_state)
    train_overlap(train_data, y_data, random_state)


if __name__ == "__main__":
    # Specify the path to your TSV file
    file_path = '/home/andjela/Documents/intra-inter-ddfs/src/all-participants.tsv'

    path_inter = '/home/andjela/Documents/intra-inter-ddfs'
    path_intra = '/home/andjela/joplin-intra-inter'
    experiment_name = 'mix_ra'
    





