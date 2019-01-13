import argparse
import matplotlib.pyplot as plt
import os
from random import shuffle

from features import LogMelExtractor, calculate_logmel
from utilities import create_folder
import config


def plot_logmel(args):
    """Plot log Mel feature of one audio per class. 
    """

    # Arguments.
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    audios_dir = os.path.join(dataset_dir, 'audio')

    # Parameters.
    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins
    labels = config.labels
 
    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)
    
    feature_list = []
    
    shuffled_fold_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    shuffle(shuffled_fold_ids)

    # Select one audio per class and extract feature
    for class_label in range(10):
        for fold_id in shuffled_fold_ids:
            if len(feature_list) == class_label + 1:
                continue
            audios_fold_dir = os.path.join(audios_dir, 'fold{}'.format(fold_id))
            audio_names = [audio_name for audio_name in os.listdir(audios_fold_dir) if audio_name.endswith('.wav')]
            shuffle(audio_names)
            for audio_name in audio_names:
                if len(feature_list) == class_label + 1:
                    continue
                
                if class_label == int(audio_name.strip().split('-')[1]):
                    
                    audio_path = os.path.join(audios_fold_dir, audio_name)
                    
                    feature = calculate_logmel(audio_path=audio_path, 
                                            sample_rate=sample_rate, 
                                            feature_extractor=feature_extractor)
                         
                    feature_list.append(feature)
        
    # Plot
    rows_num = 3
    cols_num = 4
    n = 0
    
    fig, axs = plt.subplots(rows_num, cols_num, figsize=(10, 5))
    
    classes_num = len(labels)
    
    for n in range(classes_num):
        row = n // cols_num
        col = n % cols_num
        axs[row, col].matshow(feature_list[n].T, origin='lower', aspect='auto', cmap='jet')
        axs[row, col].set_title(labels[n])
        axs[row, col].set_ylabel('log mel')
        axs[row, col].yaxis.set_ticks([])
        axs[row, col].xaxis.set_ticks([0, seq_len])
        axs[row, col].xaxis.set_ticklabels(['0', '4 s'], fontsize='small')
        axs[row, col].xaxis.tick_bottom()
    
    for n in range(classes_num, rows_num * cols_num):
        row = n // cols_num
        col = n % cols_num
        axs[row, col].set_visible(False)
    
    fig.tight_layout()
    if args.saveplot:
        create_folder(os.path.join(workspace, 'plots'))
        fig.savefig(os.path.join(workspace, 'plots', 'log-mel-per-class.png'))
    else:
        plt.show()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_plot_logmel = subparsers.add_parser('plot_logmel')
    parser_plot_logmel.add_argument('--dataset_dir', type=str, required=True)
    parser_plot_logmel.add_argument('--workspace', type=str, required=True)
    parser_plot_logmel.add_argument('--saveplot', action='store_true', default=False)
    
    args = parser.parse_args()
    
    if args.mode == 'plot_logmel':
        plot_logmel(args)   
    else:
        raise Exception("Incorrect arguments!")