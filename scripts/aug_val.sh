#!/bin/bash

GPU_IND=$1

for val_fold in 1 2 3 4 5 6 7 8 9 10
do
	for mix_param in 1 2 3 4 5
	do
		echo "Training on:" "urban_sound_8K_0.""$mix_param""_base.h5"
		CUDA_VISIBLE_DEVICES=$GPU_IND python pytorch/main_pytorch.py train --workspace='workspace' --validation_fold=$val_fold --model='vgg' --max_iters='10000' --validate --cuda --learning_rate='10e-4' --batch_size='64' --ckpt_interval='1000' --val_interval='10' --lrdecay_interval='500' --features_type='logmel' --features_file_name="urban_sound_8K_0.""$mix_param""_base.h5" --va_features_file_name='urban_sound_8K_base.h5';
	done
done
