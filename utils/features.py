import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy as np
import pandas as pd
import argparse
import h5py
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import time
import csv
import random

from utilities import read_audio, create_folder
import config
from tqdm import tqdm

# Global flags and variables. 
PLOT_FEATURES = False

class LogMelExtractor():
    def __init__(self, sample_rate, window_size, overlap, mel_bins):
        
        self.window_size = window_size
        self.overlap = overlap
        
        # Loading hamming window and Mel-filters.
        self.ham_win = np.hamming(window_size)
        self.melW = librosa.filters.mel(sr=sample_rate, 
                                        n_fft=window_size, 
                                        n_mels=mel_bins, 
                                        fmin=50., 
                                        fmax=sample_rate // 2).T
    
    # transform: Assumes a numpy array representing raw audio-signal.
    def transform(self, audio):
    
        ham_win = self.ham_win
        window_size = self.window_size
        overlap = self.overlap
    
        # Compute a spectrogram with consecutive Fourier transforms.
        [f, t, x] = signal.spectral.spectrogram(
                        audio, 
                        window=ham_win,
                        nperseg=window_size, 
                        noverlap=overlap, 
                        detrend=False, 
                        return_onesided=True, 
                        mode='magnitude') 
        x = x.T
        
        # Applying mel-filters on sequence of fourier transforms.    
        x = np.dot(x, self.melW)

        # Applying log on mel-filters.
        x = np.log(x + 1e-8)
        
        x = x.astype(np.float32)
        return x


def calculate_logmel(audio_path, sample_rate, feature_extractor):
    
    # Read audio (first 4 seconds only).
    (audio, fs) = read_audio(audio_path, target_fs=sample_rate)
    
    # Extract feature
    feature = feature_extractor.transform(audio)
    
    return feature


def read_meta(meta_csv):
    
    df = pd.read_csv(meta_csv, sep=',')
    df = pd.DataFrame(df)
    
    audio_names = [] # The name of the audio file.
    fs_IDs = [] # The Freesound ID of the recording from which this excerpt (slice) is taken
    start_times = [] # The start time of the slice in the original Freesound recording
    end_times = [] # The end time of slice in the original Freesound recording
    saliences = [] # A (subjective) salience rating of the sound. 1 = foreground, 2 = background.
    folds = [] # The fold number (1-10) to which this file has been allocated.
    class_IDs = [] # A numeric identifier of the sound class.
    classes = [] # The class name: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music.
    
    for row in df.iterrows():
        audio_name = row[1]['slice_file_name']
        fs_ID = row[1]['fsID']
        start_time = row[1]['start']
        end_time = row[1]['end']
        salience = row[1]['salience']
        fold = row[1]['fold']
        class_ID = row[1]['classID']
        classs = row[1]['class']
        
        audio_names.append(audio_name)
        fs_IDs.append(fs_ID)
        start_times.append(start_time)
        end_times.append(end_time)
        saliences.append(salience)
        folds.append(fold)
        class_IDs.append(class_ID)
        classes.append(classs)
        
    return audio_names, fs_IDs, start_times, end_times, saliences, folds, class_IDs, classes
    
    
def calculate_features(args):
    
    # Arguments. 
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    features_type = args.mode
    features_file_name = args.features_file_name

    # Parameters for feature extraction.
    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins

    # Dislaying arguments and parameters.
    print("Arguments and Parameters:")
    print("Dataset Directory: {}".format(dataset_dir))
    print("Workspace: {}".format(workspace))
    print("Sample Rate: {}".format(sample_rate))
    print("Window Size: {}".format(window_size))
    print("Overlapping Frames: {}".format(overlap))
    print("Sequence Length: {}".format(seq_len)) # Dimension of feature corresponding to each audio file: (seq_len, mel_bins)
    print("Mel Bins: {}".format(mel_bins))

    # Paths
    audio_dir = os.path.join(dataset_dir, 'audio') 
    meta_csv = os.path.join(dataset_dir, 'metadata', 'UrbanSound8K.csv') 
    hdf5_path = os.path.join(workspace, 'features', features_type, features_file_name) 

    # Displaying paths.
    print("Reading audio from: {}".format(audio_dir))
    print("Reading meatadata file form: {}".format(meta_csv))
    print("Saving the extracted features at: {}".format(hdf5_path))
        
    create_folder(os.path.dirname(hdf5_path))    
    
    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)

    audio_names, fs_IDs, start_times, end_times, saliences, folds, class_IDs, classes = read_meta(meta_csv)

    # Create hdf5 file
    hf = h5py.File(hdf5_path, 'w')
    
    # Intialising hdf5 file to store audios/labels of all folds.
    for fold_id in range(1, 11):
        hf.create_dataset(
            name='features_fold{}'.format(fold_id), 
            shape=(0, seq_len, mel_bins), 
            maxshape=(None, seq_len, mel_bins), 
            dtype=np.float32)
        hf.create_dataset(
            name='labels_fold{}'.format(fold_id), 
            shape=(0, 1), 
            maxshape=(None, 1), 
            dtype=np.float32)

    # To remember number of audio files processed in each fold. 
    fold_count = [0] * 11               
    
    for (n, audio_name) in tqdm(enumerate(audio_names)):

        # Calculate feature.
        audio_path = os.path.join(audio_dir, 'fold{}'.format(folds[n]), audio_name) 
        
        # Extract feature.
        feature = calculate_logmel(audio_path=audio_path, 
                                    sample_rate=sample_rate, 
                                    feature_extractor=feature_extractor)

        hf['features_fold{}'.format(folds[n])].resize((fold_count[folds[n]] + 1, seq_len, mel_bins))
        hf['features_fold{}'.format(folds[n])][fold_count[folds[n]]] = feature

        hf['labels_fold{}'.format(folds[n])].resize((fold_count[folds[n]] + 1, 1))
        hf['labels_fold{}'.format(folds[n])][fold_count[folds[n]]] = class_IDs[n]   

        fold_count[folds[n]] += 1     
        
        # Plot log-Mel for debug.
        if PLOT_FEATURES:
            plt.matshow(feature.T, origin='lower', aspect='auto', cmap='jet')
            plt.show()

    hf.close()

    # Displaying total files processed from each fold.
    print("Files Processed from each fold:")
    for fold_id in range(1, 11):
        print("Fold {}: {} files.".format(fold_id, fold_count[fold_id]))



# USAGE: python features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode') # Different modes can be added to extract different type of features.

    parser_logmel = subparsers.add_parser('logmel')
    parser_logmel.add_argument('--dataset_dir', type=str, required=True) # Path to the UrbanSound8K folder.
    parser_logmel.add_argument('--workspace', type=str, required=True) # Directory where extracted features, model and logs of experiments are stored.
    parser_logmel.add_argument('--features_file_name', type=str, required=True) # logmel-features.h5
    args = parser.parse_args()
    
    if args.mode == 'logmel':
        calculate_features(args)
    else:
        raise Exception('Incorrect arguments!')
        
