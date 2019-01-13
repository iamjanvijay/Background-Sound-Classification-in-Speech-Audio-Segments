# Sound Classification
Contains code for CNNs and log-mel-spectrograms based sound classification.

### What does this code support?

1. **Feature Extraction**: Extracts log-mel-spectrograms from audios in UrbanSound8K and stores them in the hdf5 file. \\
**Usage**: python utils/features.py logmel --dataset_dir=$URBANSOUND8K_DATASET_DIR --workspace=$WORKSPACE

2. **Training**: Trains the model for 2000 iterations at max (can be changed), computes the training and validation accuracy after every 100 iterations. Saves the corresponding logs and models (after every 1000 steps) in $WORKSPACE. \\
**Usage**: python pytorch/main_pytorch.py train --workspace='workspace' --validation_fold='10' --validate --cuda

3. **Plotting**: There is support to plot confusion matrix, and log-mel-specs as well, look out for file: utils/plot_figures.py and utils/utilities.py

### How to set up virtual environment?
Execute the following commands in sequence:
- pip install virtualenv
- virtualenv -p python3 venv_noise
- source venv_noise/bin/activate
- pip install -r requirements.txt





