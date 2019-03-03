import argparse

from utilities import augment_audio_folder, separate_fg_bg

AUDIO_AUG_MODE_RANDOM = 0
AUDIO_AUG_MODE_NOISE_DATA = 1
AUDIO_AUG_MODE_STRETCH = 2
AUDIO_AUG_MODE_ROLL = 3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Performs data augmentation.

    	Modes:
    	mode_1: Augments all audio files within a directory with a certain audio file.
		mode_2: Seprates out all bg and fg files.

		''')
    
    subparsers = parser.add_subparsers(dest='mode')

    # Arguments for augmentation in mode_1.
    mode_1_aug = subparsers.add_parser('mode_1')
    mode_1_aug.add_argument('--audio_dir', type=str, required=True)
    mode_1_aug.add_argument('--audio_file', type=str, required=True)
    mode_1_aug.add_argument('--mixing_param', type=float, default=0.005)
    mode_1_aug.add_argument('--output_folder', type=str, required=True)

    # Arguments for augmentation in mode_1.
    mode_2_aug = subparsers.add_parser('mode_1')
    mode_2_aug.add_argument('--audio_dir', type=str, required=True)
    mode_2_aug.add_argument('--output_folder', type=str, required=True)    

    args = parser.parse_args()

    if args.mode == 'mode_1':
    	augment_audio_folder(input_folder=args.audio_dir, output_folder=args.output_folder,  mixing_param=args.mixing_param, noise_file=args.audio_file, mode=AUDIO_AUG_MODE_NOISE_DATA, recursive=True)
    elif args.mode == 'mode_2':
        separate_fg_bg(input_folder=args.audio_dir, output_folder=args.output_folder)




