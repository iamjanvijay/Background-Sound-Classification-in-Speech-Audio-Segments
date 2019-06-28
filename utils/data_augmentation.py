import argparse

from utilities import augment_audio_folder, separate_fg_bg, random_augmentation

AUDIO_AUG_MODE_RANDOM = 0
AUDIO_AUG_MODE_NOISE_DATA = 1
AUDIO_AUG_MODE_STRETCH = 2
AUDIO_AUG_MODE_ROLL = 3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Performs data augmentation.

    	Modes:
    	mode_1: Augments all audio files within a directory with a certain audio file.
		mode_2: Seprates out all bg and fg files.
        mode_3: Augments all audio files within a directory with random audio files in other directory.
		''')
    
    subparsers = parser.add_subparsers(dest='mode')

    # Arguments for augmentation in mode_1.
    mode_1_aug = subparsers.add_parser('mode_1')
    mode_1_aug.add_argument('--audio_dir', type=str, required=True)
    mode_1_aug.add_argument('--audio_file', type=str, required=True)
    mode_1_aug.add_argument('--mixing_param', type=float, default=0.005)
    mode_1_aug.add_argument('--output_folder', type=str, required=True)

    # Arguments for augmentation in mode_2.
    mode_2_aug = subparsers.add_parser('mode_2')
    mode_2_aug.add_argument('--audio_dir', type=str, required=True)
    mode_2_aug.add_argument('--output_folder', type=str, required=True)  

    # Arguments for augmentation in mode_3.  
    mode_3_aug = subparsers.add_parser('mode_3')
    mode_3_aug.add_argument('--audio_dir', type=str, required=True) # noise datset like UrbanSound8K
    mode_3_aug.add_argument('--speech_dir', type=str, required=True) # speech dir dataset for augmentation
    mode_3_aug.add_argument('--augment_files_per_audio', type=int, required=True)
    mode_3_aug.add_argument('--mixing_param_range_min', type=float, required=True)
    mode_3_aug.add_argument('--mixing_param_range_max', type=float, required=True)
    mode_3_aug.add_argument('--output_folder', type=str, required=True)  

    args = parser.parse_args()

    if args.mode == 'mode_1':
    	augment_audio_folder(input_folder=args.audio_dir, output_folder=args.output_folder,  mixing_param=args.mixing_param, noise_file=args.audio_file, mode=AUDIO_AUG_MODE_NOISE_DATA, recursive=True)
    elif args.mode == 'mode_2':
        separate_fg_bg(input_folder=args.audio_dir, output_folder=args.output_folder)
    elif args.mode == 'mode_3':
        assert (args.mixing_param_range_min <= args.mixing_param_range_max)
        assert (args.augment_files_per_audio > 0)
        random_augmentation(input_audio_folder=args.audio_dir, input_speech_folder=args.speech_dir, files_per_audio=args.augment_files_per_audio, range_min=args.mixing_param_range_min, range_max=args.mixing_param_range_max, output_folder=args.output_folder)




