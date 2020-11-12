from data import get_cifar10, get_train_transforms, DataGenerator
from model import get_model
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

def args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-l', '--learning-rate', default = 2e-4, type = float, help = 'learning rate to train')
    ap.add_argument('-e', '--epochs', default = 40, type = int, help = '# of epochs')
    ap.add_argument('-b', '--batch-size', default = 32, type = int, help = 'batch size')
    return ap.parse_args()

def get_optimizer(initial_lr):
    return Adam(learning_rate = initial_lr)

def get_callbacks():
    return [
        ModelCheckpoint(filepath = os.path.join('checkpoints', 'model'), save_weights_only = True, save_best_only = True, verbose = 1),
        ReduceLROnPlateau(factor = .9, patience = 5, verbose = 1)
    ]

def main(args):
    lr = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    optimizer = get_optimizer(lr)
    model = get_model((32, 32, 3))
    
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    loss = SparseCategoricalCrossentropy(from_logits = True)
    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])

    (data_train, label_train), (data_test, label_test) = get_cifar10()
    train_dataset = DataGenerator(data_train, label_train, transforms = get_train_transforms(), batch_size = batch_size)
    val_dataset = DataGenerator(data_test, label_test, batch_size = 32, train = False, shuffle = False)

    callbacks = get_callbacks()
    model.fit_generator(train_dataset, validation_data = val_dataset, epochs = epochs, callbacks = callbacks)

if __name__ == "__main__":
    main(args())