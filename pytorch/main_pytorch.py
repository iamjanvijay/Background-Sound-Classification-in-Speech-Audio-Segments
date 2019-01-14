import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_generator import DataGenerator #, TestDataGenerator
from utilities import (create_folder, get_filename, create_logging,
                       calculate_confusion_matrix, calculate_accuracy, 
                       plot_confusion_matrix, print_accuracy)
from models_pytorch import move_data_to_gpu, BaselineCnn, Vggish
import config

# Global flags and variables.
Model = Vggish
batch_size = 64
PLOT_CONFUSION_MATRIX = True
SAVE_PLOT = True


def evaluate(model, generator, data_type, max_iteration, plot_title, workspace, cuda):
    """Evaluate
    
    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      max_iteration: int, maximum iteration for validation
      cuda: bool.
      
    Returns:
      accuracy: float
    """
    
    # Generate function
    generate_func = generator.generate_validate(data_type=data_type,  
                                                shuffle=True, 
                                                max_iteration=max_iteration)
            
    # Forward
    dict = forward(model=model, 
                   generate_func=generate_func, 
                   cuda=cuda, 
                   return_target=True)

    outputs = dict['output']    # (audios_num, classes_num)
    targets = dict['target']    # (audios_num, 1)

    audios_num, classes_num = outputs.shape

    targets = targets.reshape(audios_num) # Reshaping to (audios_num,)    
    predictions = np.argmax(outputs, axis=-1)   # (audios_num,) 

    loss = F.nll_loss(torch.Tensor(outputs), torch.LongTensor(targets)).numpy()
    loss = float(loss)

    class_wise_accuracy = calculate_accuracy(targets, predictions, classes_num)
    
    if PLOT_CONFUSION_MATRIX:
        confusion_matrix = calculate_confusion_matrix(targets, predictions, classes_num) # Can be used if confustion matrix is to be plotted.
        plot_confusion_matrix(confusion_matrix, plot_title, config.labels, class_wise_accuracy, SAVE_PLOT, workspace)

    accuracy = np.mean(class_wise_accuracy)

    return accuracy, loss


def forward(model, generate_func, cuda, return_target):
    """Forward data to a model.
    
    Args:
      generate_func: generate function
      cuda: bool
      return_target: bool
      
    Returns:
      dict, keys: 'output'; optional keys: 'target'
    """
    
    outputs = []
    
    if return_target:
        targets = []

    # Evaluate on mini-batch
    for data in generate_func:
            
        (batch_x, batch_y) = data
        batch_x = move_data_to_gpu(batch_x, cuda)

        # Predict
        model.eval()
        batch_output = model(batch_x)

        # Append data
        outputs.append(batch_output.data.cpu().numpy())
        
        if return_target:
            targets.append(batch_y)

    dict = {}

    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs
 
    if return_target:
        targets = np.concatenate(targets, axis=0)
        dict['target'] = targets
        
    return dict


def train(args):

    # Arugments.
    workspace = args.workspace
    cuda = args.cuda
    validate = args.validate
    validation_fold = args.validation_fold

    # Parameters.
    labels = config.labels

    classes_num = len(labels)
    hdf5_path = os.path.join(workspace, 'features', 'logmel', 'logmel-features.h5') # Features to be used for training.
    models_dir = os.path.join(workspace, 'models') # Directory to save models.

    create_folder(models_dir)

    # Model.
    model = Model(classes_num)
    if cuda:
        model.cuda()

    # Data generator.
    generator = DataGenerator(hdf5_path=hdf5_path, batch_size=batch_size, validation_fold=validation_fold)

    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)

    train_bgn_time = time.time()

    best_va_acc = 0
    best_tr_acc = 0
    # Train on mini batches.
    for (iteration, (batch_x, batch_y)) in enumerate(generator.generate_train()):
        
        # Evaluate both on training data and validation data. (After every 100 iterations)
        if iteration % 100 == 0:

            train_fin_time = time.time()

            (tr_acc, tr_loss) = evaluate(model=model,
                                         generator=generator,
                                         data_type='train',
                                         max_iteration=None,
                                         plot_title='train_iter_{}'.format(iteration),
                                         workspace=workspace,
                                         cuda=cuda)
            best_tr_acc = max(best_tr_acc, tr_acc)
            logging.info('best_tr_acc: {:.3f}, tr_acc: {:.3f}, tr_loss: {:.3f}'.format(best_tr_acc, tr_acc, tr_loss))

            if validate:
                
                (va_acc, va_loss) = evaluate(model=model,
                                             generator=generator,
                                             data_type='validate',
                                             max_iteration=None,
                                             plot_title='val_iter_{}'.format(iteration),
                                             workspace=workspace,
                                             cuda=cuda)
                best_va_acc = max(best_va_acc, va_acc)                
                logging.info('best_va_acc: {:.3f}, va_acc: {:.3f}, va_loss: {:.3f}'.format(best_va_acc, va_acc, va_loss))

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()

        # Save model
        if iteration % 1000 == 0 and iteration > 0:

            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }
            save_out_path = os.path.join(
                models_dir, 'md_{}_iters.tar'.format(iteration))
            torch.save(save_out_dict, save_out_path)
            logging.info('Model saved to {}'.format(save_out_path))
            
        # Reduce learning rate
        if iteration % 200 == 0 > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

        # Train : That's where the training begins.
        audios_num = batch_y.shape[0]
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y.astype(int).reshape(audios_num), cuda) # (audios_num, 1)

        model.train()
        batch_output = model(batch_x) # (audios_num, classes_num)
        loss = F.nll_loss(batch_output, batch_y) # output: (N, C) and Target: (N)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop learning
        if iteration == 2000:
            break

# USAGE: python pytorch/main_pytorch.py train --workspace='workspace' --validation_fold='10' --validate --cuda

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser.')
    subparsers = parser.add_subparsers(dest='mode')

    # Arguments for training mode.
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--validation_fold', type=int, default=False)
    parser_train.add_argument('--validate', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)

    # Arguments for inference mode. [Can be added, if required].
    # parser_inference_evaluation_data = subparsers.add_parser('inference')
    # parser_inference_evaluation_data.add_argument('--dataset_dir', type=str, required=True)
    # parser_inference_evaluation_data.add_argument('--workspace', type=str, required=True)
    # parser_inference_evaluation_data.add_argument('--iteration', type=int, required=True)
    # parser_inference_evaluation_data.add_argument('--cuda', action='store_true', default=False)  

    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log.
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        assert(args.validation_fold in range(1, 11))
        train(args)
    # elif args.mode == 'inference':
    #     inference(args)
    else:
        raise Exception('Error argument!')