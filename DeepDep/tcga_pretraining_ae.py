import torch
import numpy as np
import os
import pickle
import time
from keras import models
from keras.layers import Dense
from keras.callbacks import EarlyStopping

def AE_dense_3layers(input_dim, first_layer_dim, second_layer_dim, third_layer_dim, activation_func, init='he_uniform'):
    """
    Creates a dense autoencoder model with 3 hidden layers.
    
    :param input_dim: Dimension of the input layer
    :param first_layer_dim: Dimension of the first hidden layer
    :param second_layer_dim: Dimension of the second hidden layer
    :param third_layer_dim: Dimension of the third hidden layer
    :param activation_func: Activation function for all layers
    :param init: Weight initialization method
    :return: Compiled Keras Sequential model
    """
    model = models.Sequential()
    model.add(Dense(units=first_layer_dim, input_dim=input_dim, activation=activation_func, kernel_initializer=init))
    model.add(Dense(units=second_layer_dim, activation=activation_func, kernel_initializer=init))
    model.add(Dense(units=third_layer_dim, activation=activation_func, kernel_initializer=init))
    model.add(Dense(units=second_layer_dim, activation=activation_func, kernel_initializer=init))
    model.add(Dense(units=first_layer_dim, activation=activation_func, kernel_initializer=init))
    model.add(Dense(units=input_dim, activation=activation_func, kernel_initializer=init))
    return model

def save_weight_to_pickle(model, file_name):
    """
    Saves the weights of a trained Keras model to a pickle file.
    
    :param model: Trained Keras model whose weights need to be saved
    :param file_name: Path to the pickle file where the weights will be stored
    """
    weight_list = []
    for layer in model.layers:
        weight_list.append(layer.get_weights())
    with open(file_name, 'wb') as handle:
        pickle.dump(weight_list, handle)

def load_split_data(split_num, omic):
    """
    Loads the train, validation, and test datasets for a specific omic type and split number.
    
    :param split_num: Split number to identify the data split
    :param omic: Omic data type (e.g., 'cna', 'mut', etc.)
    :return: Tuple containing train, validation, and test datasets
    """
    base_path = f'./TCGASplits/split_{split_num}'
    train_file = os.path.join(base_path, f'train_dataset_{omic}_split_{split_num}.pth')
    val_file = os.path.join(base_path, f'val_dataset_{omic}_split_{split_num}.pth')
    test_file = os.path.join(base_path, f'test_dataset_{omic}_split_{split_num}.pth')
    
    train_dataset = torch.load(train_file)
    val_dataset = torch.load(val_file)
    test_dataset = torch.load(test_file)
    
    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    omic = "cna" # Specify the omic type (e.g., 'cna', 'mut')
    
    for split_num in range(1, 6):
        train_dataset, val_dataset, test_dataset = load_split_data(split_num, omic)
        
        train_data = np.array([data[0].numpy() for data in train_dataset])
        val_data = np.array([data[0].numpy() for data in val_dataset])
        test_data = np.array([data[0].numpy() for data in test_dataset])

        input_dim = train_data.shape[1]
        first_layer_dim = 500
        second_layer_dim = 200
        third_layer_dim = 50
        batch_size = 500
        epoch_size = 100
        activation_function = 'relu'
        init = 'he_uniform'
        
        model = AE_dense_3layers(input_dim, first_layer_dim, second_layer_dim, third_layer_dim, activation_function, init)
        model.compile(loss='mse', optimizer='adam')
        
        t = time.time()
        model.fit(train_data, train_data, epochs=epoch_size, batch_size=batch_size, shuffle=True, validation_data=(val_data, val_data), callbacks=[EarlyStopping(patience=3)])
        cost = model.evaluate(test_data, test_data, verbose=0)
        
        save_weight_to_pickle(model, f'./tcga_{omic}_ae_best_split_{split_num}.pickle')
