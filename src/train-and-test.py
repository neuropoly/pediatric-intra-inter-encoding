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


class CustomModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def __init__(self, train_dir, **kwargs):
        self.train_dir = train_dir
        filepath = f"{self.train_dir}/3d_image_classification_epoch{{epoch:02d}}.h5"
        super().__init__(filepath=filepath, **kwargs)

def train_no_overlap(model, train_dataset, validation_dataset, epochs, train_iteration='_no_overlap_9'):
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

    # Define callbacks.
    # checkpoint_cb = CustomModelCheckpoint(
    #     train_dir,
    #     save_best_only=True,
    #     monitor='val_loss',  # Set the metric to monitor, typically validation loss
    #     verbose=1,  # Set to 1 to receive a message when a new best model is saved
    #     mode='min'  # Set to 'min' if you are monitoring a metric like validation loss
    # )

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

def train_overlap():
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
    for train_index, test_index in kf.split(train_data):
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
