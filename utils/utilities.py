import numpy as np
import librosa
import os
from sklearn import metrics
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import config
import shutil

AUDIO_AUG_MODE_RANDOM = 0
AUDIO_AUG_MODE_NOISE_DATA = 1
AUDIO_AUG_MODE_STRETCH = 2
AUDIO_AUG_MODE_ROLL = 3

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def read_audio(path, target_fs=None, mode='repeat'):
    audio, fs = librosa.core.load(path, sr=target_fs, mono=True, duration=4)

    # If audio length is less than 4 seconds, appends silence (zeros) at end.
    target_num_samples = fs * 4
    if target_num_samples != len(audio):
        if mode == 'append_zeros':
            audio = np.concatenate((audio, np.zeros(target_num_samples - len(audio))))
        else: # mode == 'repeat':
            audio = np.tile(audio, int(target_num_samples / len(audio)) + 1)[:target_num_samples]

    return audio, fs

def write_audio_file(file_name, audio_data, sample_rate=config.sample_rate):
    librosa.output.write_wav(file_name, audio_data, sample_rate)

def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def create_logging(log_dir, filemode):

    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '%04d.log' % i1)):
        i1 += 1

    log_path = os.path.join(log_dir, '%04d.log' % i1)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging

def calculate_scalar(x):

    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def scale(x, mean, std):

    return (x - mean) / std


def inverse_scale(x, mean, std):

    return x * std + mean


def calculate_accuracy(target, predict, classes_num, average=None):
    """Calculate accuracy.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)

    Outputs:
      accuracy: float
    """
    target = target.astype(int)
    samples_num = len(target)
    
    correctness = np.zeros(classes_num)
    total = np.zeros(classes_num)

    for n in range(samples_num):
        
        total[target[n]] += 1
        
        if target[n] == predict[n]:
            correctness[target[n]] += 1

    accuracy = correctness / total

    if average is None:
        return accuracy
        
    elif average == 'macro':
        return np.mean(accuracy)
        
    else:
        raise Exception('Incorrect average!')


def calculate_confusion_matrix(target, predict, classes_num):
    """Calculate confusion matrix.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)
      classes_num: int, number of classes

    Outputs:
      confusion_matrix: (classes_num, classes_num)
    """
    target = target.astype(int)
    confusion_matrix = np.zeros((classes_num, classes_num))
    samples_num = len(target)


    for n in range(samples_num):
        assert(int(target[n]) in range(10) and predict[n] in range(10))
        confusion_matrix[target[n], predict[n]] += 1

    return confusion_matrix


def print_accuracy(class_wise_accuracy, labels):

    print('{:<30}{}'.format('Scene label', 'accuracy'))
    print('------------------------------------------------')
    for (n, label) in enumerate(labels):
        print('{:<30}{:.3f}'.format(label, class_wise_accuracy[n]))
    print('------------------------------------------------')
    print('{:<30}{:.3f}'.format('Average', np.mean(class_wise_accuracy)))


def plot_confusion_matrix(confusion_matrix, title, labels, values, save_plot, workspace):
    """Plot confusion matrix.

    Inputs:
      confusion_matrix: matrix, (classes_num, classes_num)
      labels: list of labels
      values: list of values to be shown in diagonal

    Ouputs:
      None
    """

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

    if labels:
        ax.set_xticklabels([''] + labels, rotation=90, ha='left')
        ax.set_yticklabels([''] + labels)
        ax.xaxis.set_ticks_position('bottom')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    for n in range(len(values)):
        plt.text(n - 0.4, n, '{:.2f}'.format(values[n]), color='yellow')

    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.tight_layout()
    
    if save_plot:
        create_folder(os.path.join(workspace, 'plots'))
        fig.savefig(os.path.join(workspace, 'plots', 'confusion-{}.png'.format(title)))
    else:
        plt.show()


def add_noisy_audio(audio_data, noise_data, mixing_parameter):
    new_data = (1 - mixing_parameter) * audio_data + mixing_parameter * noise_data
    return new_data


def add_random_noise(audio_data, mixing_parameter):
    noise = np.random.randn(len(audio_data))
    noisy_data = add_noisy_audio(audio_data, noise, mixing_parameter)
    return noisy_data


def audio_shift(audio_data, shift_count=config.sample_rate):
    return np.roll(audio_data, shift_count)


def audio_stretch(audio_data, rate=1):
    stretched_data = librosa.effects.time_stretch(audio_data, rate)
    return stretched_data


def augment_audio_data(audio_data, mixing_param=0.005, noise_data=None, mode=AUDIO_AUG_MODE_RANDOM):
    ret_data = audio_data

    if mode == AUDIO_AUG_MODE_RANDOM:
        ret_data = add_random_noise(audio_data, mixing_param)
    elif mode == AUDIO_AUG_MODE_NOISE_DATA:
        ret_data = add_noisy_audio(audio_data, noise_data, mixing_param)
    elif mode == AUDIO_AUG_MODE_ROLL:
        ret_data = audio_shift(audio_data)
    elif mode == AUDIO_AUG_MODE_STRETCH:
        ret_data = audio_stretch(audio_data)

    return ret_data


def augment_audio_file(audio_file, out_filename=None, mixing_param=0.005, noise_file=None, mode=AUDIO_AUG_MODE_RANDOM):
    audio_data, fs = read_audio(audio_file, config.sample_rate)
    noise_data = None
    if noise_file is not None:
        noise_data, fs = read_audio(noise_file, config.sample_rate)

    aug_audio = augment_audio_data(audio_data, mixing_param, noise_data, mode)

    if out_filename is not None:
        write_audio_file(out_filename, aug_audio, config.sample_rate)

    return aug_audio


# AUDIO_AUG_MODE_NOISE_DATA - noise_file should be specified which will be mixed with each file in input_folder
# AUDIO_AUG_MODE_RANDOM - random noise will be added
# AUDIO_AUG_MODE_ROLL - data will be rolled left to right by 1 sec
# AUDIO_AUG_MODE_STRETCH - data will be stretched
def augment_audio_folder(input_folder, output_folder,  mixing_param=0.005, noise_file=None, mode=AUDIO_AUG_MODE_RANDOM, recursive=False):
    count = 0

    #create directory if absent
    create_folder(output_folder)

    if recursive:
        for dirName, subdirList, fileList in os.walk(input_folder):

            for subdir in subdirList:
                new_folder_path = os.path.join(output_folder, os.path.relpath(os.path.join(dirName, subdir), input_folder))
                print("Creating folder : {}".format(new_folder_path))
                create_folder(new_folder_path)

            print("Processing folder {}".format(dirName))

            for file in fileList:
                if file.endswith('.wav'):
                    filepath = os.path.join(dirName, file)
                    print ("Processing file : {}".format(filepath))
                    out_full_path = os.path.join(output_folder, os.path.relpath(filepath, input_folder))
                    augment_audio_file(filepath, out_full_path, mixing_param, noise_file, mode)
                    count += 1
                else: # Copy other files as it as.
                    filepath = os.path.join(dirName, file)
                    out_full_path = os.path.join(output_folder, os.path.relpath(filepath, input_folder))
                    shutil.copy2(filepath, out_full_path)
    else:
        for filename in glob.glob(os.path.join(input_folder, '*.wav')):
            print("Processing file : {}".format(filename))
            out_full_path = os.path.join(output_folder, os.path.basename(filename))
            augment_audio_file(filename, out_full_path, mixing_param, noise_file, mode)
            count += 1

    print("Augmented {} files generated in folder {}".format(count, output_folder))
