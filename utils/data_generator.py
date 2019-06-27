import numpy as np
import h5py
import csv
import time
import logging
from functools import reduce 

from utilities import calculate_scalar, scale
import config


class DataGenerator(object):

    def __init__(self, hdf5_path, batch_size, validation_fold, total_folds=10, seed=1234):
        """
        Inputs:
          hdf5_path: str
          batch_size: int
          seed: int, random seed
        """

        self.batch_size = batch_size

        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)
        lb_to_ix = config.lb_to_ix

        # Load data
        load_time = time.time()
        hf = h5py.File(hdf5_path, 'r')

        self.x = []
        self.y = []
        for fold_id in range(1, total_folds+1):
            self.x.append(hf['features_fold{}'.format(fold_id)][:])
            self.y.append(hf['labels_fold{}'.format(fold_id)][:])

        hf.close()
        logging.info('Loading data time: {:.3f} s'.format(time.time() - load_time))
        
        # Assigning training and validation indices in stacked training data.
        total_trainig_example = reduce(lambda x, y: x+y, [self.x[fold_id].shape[0] for fold_id in range(0, total_folds)])
        if validation_fold == 1:
            validation_start = 0
        else:
            validation_start = reduce(lambda x, y: x+y, [self.x[fold_id].shape[0] for fold_id in range(0, validation_fold-1)])
        validation_end = validation_start + self.x[validation_fold-1].shape[0]
        
        self.validate_audio_indexes = np.arange(validation_start, validation_end)
        self.train_audio_indexes = np.concatenate((np.arange(0, validation_start), np.arange(validation_end, total_trainig_example)), axis=0)

        self.x = np.vstack(self.x)
        self.y = np.vstack(self.y)

        # Calculates mean and standard deviation for each audio file's feature.
        (self.mean, self.std) = calculate_scalar(self.x[self.train_audio_indexes])

    def generate_train(self):
        """Generate mini-batch data for training. 
        
        Returns:
          batch_x: (batch_size, seq_len, freq_bins)
          batch_y: (batch_size,)
        """

        batch_size = self.batch_size
        audio_indexes = np.array(self.train_audio_indexes)
        audios_num = len(audio_indexes)

        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        while True:

            # Reset pointer
            if pointer >= audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch indexes.
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size

            iteration += 1

            batch_x = self.x[batch_audio_indexes]
            batch_y = self.y[batch_audio_indexes]

            # Transform data.
            batch_x = self.transform(batch_x)

            yield batch_x, batch_y

    def generate_validate(self, data_type, shuffle, max_iteration=None):
        """Generate mini-batch data for evaluation. 
        
        Args:
          data_type: 'train' | 'validate'
          devices: list of devices, e.g. ['a'] | ['a', 'b', 'c']
          max_iteration: int, maximum iteration for validation
          shuffle: bool
          
        Returns:
          batch_x: (batch_size, seq_len, freq_bins)
          batch_y: (batch_size,)
          batch_audio_names: (batch_size,)
        """

        batch_size = self.batch_size

        if data_type == 'train':
            audio_indexes = np.array(self.train_audio_indexes)

        elif data_type == 'validate':
            audio_indexes = np.array(self.validate_audio_indexes)

        else:
            raise Exception('Invalid data_type!')
            
        if shuffle:
            self.validate_random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        audios_num = len(audio_indexes)
        while True:

            # If all validation indices are traversed.
            if pointer >= audios_num:
                break

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size

            iteration += 1

            batch_x = self.x[batch_audio_indexes]
            batch_y = self.y[batch_audio_indexes]

            # Transform data
            batch_x = self.transform(batch_x)

            yield batch_x, batch_y

    def transform(self, x):
        """Transform data. 
        
        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)
          
        Returns:
          Transformed data. 
        """

        return scale(x, self.mean, self.std)
