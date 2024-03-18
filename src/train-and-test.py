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
from sklearn.model_selection import KFold, train_test_split
from model import get_model
from visualization import visualize_model_performance
from utils import calculate_accuracy_test_set
from preprocessing import data_loaders, split_into_indices, process_selected_sets, process_scan


def train_loop_no_overlap(model, train_dataset, validation_dataset, epochs, train_iteration='_no_overlap_9'):
    # Train iteration number to be changed based on the experiment
    # train_iteration = '_no_overlap_9'
    # Define train_dir and log_dir
    train_dir = os.path.join(os.getcwd(), f'train_{train_iteration}')
    log_dir = os.path.join(train_dir, "logs")
    # Create both train_dir and log_dir if not already existent
    for directory in [train_dir, log_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # Compile model.
    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{train_dir}/model.h5', 
							monitor='val_loss', verbose=1, 
							save_best_only=True, mode='min')
    

    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

    # Add the TensorBoard callback to the list of callbacks
    callbacks_list = [checkpoint, early_stopping_cb]  # Include other callbacks as well

    # Train the model, doing validation at the end of each epoch
    # epochs = 100
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=2,
        callbacks=callbacks_list,
    )
    return train_dir, callbacks_list, model

def train_no_overlap(experiment_name, intra_scan_paths, inter_scan_paths, all_sets):
    epochs = 100
    train_iteration_names = [f'{experiment_name}_no_overlap_{i}' for i in range(10)]

    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []
    TEST_ACCURACY = []
    TEST_F1_SCORE = []

    TOTAL_NUM_SAMPLES = []

    nbr_sets = 10
    train_indices_list, val_indices_list, test_indices_list = split_into_indices(all_sets, nbr_sets)

    for train_iteration_name in train_iteration_names:
        split_nbr = int(re.search(r'\d+', train_iteration_name).group())
        selected_train_paths, x_train, y_train, \
        selected_val_paths, x_val, y_val, \
        selected_test_paths, x_test, y_test = process_selected_sets(
        split_nbr, intra_scan_paths, inter_scan_paths, 
        train_indices_list, val_indices_list, test_indices_list
        )
        print('X_TRAIN', x_train.shape)
        print('X_VAL', x_val.shape)
        total_num_samples = x_train.shape[3]+x_val.shape[3]+x_test.shape[3]
        train_dataset, validation_dataset = data_loaders(x_train, y_train, x_val, y_val, batch_size=2)
        # Create new model
        model = get_model(width=128, height=128, depth=64)
        
        train_dir, callbacks_list, model = train_loop_no_overlap(model, train_dataset, validation_dataset, epochs, train_iteration_name)

        visualize_model_performance(model, train_dir)
        # LOAD BEST MODEL to evaluate the performance of the model
        model.load_weights(f'{train_dir}/model.h5')
        
        results = model.evaluate(validation_dataset)
        results = dict(zip(model.metrics_names,results))

        accuracy, f1_score_calc = calculate_accuracy_test_set(model, x_test, y_test, train_dir)
        
        VALIDATION_ACCURACY.append(results['acc'])
        VALIDATION_LOSS.append(results['loss'])
        TEST_ACCURACY.append(accuracy)
        TEST_F1_SCORE.append(f1_score_calc)
        TOTAL_NUM_SAMPLES.append(total_num_samples)
        tf.keras.backend.clear_session()
    
    print('total num samples', TOTAL_NUM_SAMPLES)
    print('validation accuracy', VALIDATION_ACCURACY)
    print('test accuracy', TEST_ACCURACY)
    print('test f1_score', TEST_F1_SCORE)
    print('mean validation accuracy', np.mean(VALIDATION_ACCURACY), np.std(VALIDATION_ACCURACY))
    print('mean test accuracy', np.mean(TEST_ACCURACY), np.std(TEST_ACCURACY))
    print('mean f1 score', np.mean(TEST_F1_SCORE), np.std(TEST_F1_SCORE))
    

def get_model_name(k):
    return 'model_'+str(k)+'.h5'

def extract_train_data_overlap(intra_scan_paths, inter_scan_paths, nbr_intra_pairs, num_indices, random_state):
    
    # Set a random seed for reproducibility
    # random_state = 42
    np.random.seed(random_state)  # You can use any integer value

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

    # Number of indices to be selected to match the no_overlap training
    # num_indices = 305

    # Randomly select indices from the first 434 elements (intra part)
    first_indices = np.random.choice(nbr_intra_pairs, size=num_indices // 2, replace=False)

    # Randomly select indices from the last 421 elements
    last_indices = np.random.choice(range(nbr_intra_pairs, len(y)), size=num_indices - len(first_indices), replace=False)

    # Concatenate both sets of indices
    selected_indices = np.concatenate((first_indices, last_indices))

    # Shuffle the indices using the random state
    rng = np.random.default_rng(random_state)
    rng.shuffle(selected_indices)

    selected_train_paths = [X[i] for i in selected_indices]
    train_data = np.array([process_scan(path) for path in selected_train_paths])
    y_data = y[selected_indices]

    return train_data, y_data


def train_overlap(train_data, y_data, random_state):
    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []
    TEST_ACCURACY = []
    TEST_F1_SCORE = []


    fold_var = 0
    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
            
    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
                )
    for train_index, test_index in kf.split(train_data, y_data, random_state):
        train_iteration = f'_overlap_{fold_var}'
        train_dir = os.path.join(os.getcwd(), f'train{train_iteration}')
        # Create both train_dir if not already existent
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        # Further split the training data into 70% training and 30% validation
        new_train_index, val_index = train_test_split(train_index, test_size=0.3, random_state=random_state)
        
        x_train, y_train = train_data[new_train_index], y_data[new_train_index]
        x_val, y_val = train_data[val_index], y_data[val_index]
        x_test, y_test = train_data[test_index], y_data[test_index]      
        # Obtain augmented data for train set
        train_dataset, validation_dataset = data_loaders(x_train, y_train, x_val, y_val, batch_size=2)
        # Create new model
        model = get_model(width=128, height=128, depth=64)
        # Compile model
        model.compile(
            loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            metrics=["acc"],
        )
        # CREATE CALLBACKS
        early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{train_dir}/{get_model_name(fold_var)}', 
                                monitor='val_loss', verbose=1, 
                                save_best_only=True, mode='min')
        callbacks_list = [checkpoint, early_stopping_cb]
        # There can be other callbacks, but just showing one because it involves the model name
        # This saves the best model
        # FIT THE MODEL
        epochs = 100
        history = model.fit(train_dataset,
                    epochs=epochs,
                    callbacks=callbacks_list,
                    shuffle=True,
                    verbose=2,
                    validation_data=validation_dataset)
        
        visualize_model_performance(model, train_dir)
        # LOAD BEST MODEL to evaluate the performance of the model
        model.load_weights(f'{train_dir}/{get_model_name(fold_var)}')
        
        results = model.evaluate(validation_dataset)
        results = dict(zip(model.metrics_names,results))

        accuracy, f1_score_calc = calculate_accuracy_test_set(model, x_test, y_test, train_dir)
        
        VALIDATION_ACCURACY.append(results['acc'])
        VALIDATION_LOSS.append(results['loss'])
        TEST_ACCURACY.append(accuracy)
        TEST_F1_SCORE.append(f1_score_calc)
        tf.keras.backend.clear_session()
        fold_var += 1

    print('validation accuracy', VALIDATION_ACCURACY)
    print('test accuracy', TEST_ACCURACY)
    print('test f1_score', TEST_F1_SCORE)
    print('mean validation accuracy', np.mean(VALIDATION_ACCURACY), np.std(VALIDATION_ACCURACY))
    print('mean test accuracy', np.mean(TEST_ACCURACY), np.std(TEST_ACCURACY))
    print('mean f1 score', np.mean(TEST_F1_SCORE), np.std(TEST_F1_SCORE))
