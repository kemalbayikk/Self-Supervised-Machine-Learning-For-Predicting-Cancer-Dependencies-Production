import pickle
import torch
from torch.utils.data import DataLoader
from keras import models
from keras.layers import Dense, Concatenate, Input
from keras.callbacks import EarlyStopping
from scipy.stats import pearsonr
import numpy as np

# This function loads the pre-split train, validation, and test datasets from a pickle file.
def load_split(split_num):
    with open(f'./data_split_{split_num}.pickle', 'rb') as f:
        train_dataset, val_dataset, test_dataset = pickle.load(f)
    return train_dataset, val_dataset, test_dataset

split_num = 1
train_dataset, val_dataset, test_dataset = load_split(split_num)
batch_size = 10000

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# This function concatenates batches of data into full numpy arrays for training, validation, and testing
def prepare_data(loader):
    data_mut, data_exp, data_cna, data_meth, data_fprint, data_dep = [], [], [], [], [], []
    for batch in loader:
        data_mut.append(batch[0])
        data_exp.append(batch[1])
        data_cna.append(batch[2])
        data_meth.append(batch[3])
        data_fprint.append(batch[4])
        data_dep.append(batch[5])
    data_mut = torch.cat(data_mut).numpy()
    data_exp = torch.cat(data_exp).numpy()
    data_cna = torch.cat(data_cna).numpy()
    data_meth = torch.cat(data_meth).numpy()
    data_fprint = torch.cat(data_fprint).numpy()
    data_dep = torch.cat(data_dep).numpy()
    return data_mut, data_exp, data_cna, data_meth, data_fprint, data_dep

data_mut_train, data_exp_train, data_cna_train, data_meth_train, data_fprint_train, data_dep_train = prepare_data(train_loader)
data_mut_val, data_exp_val, data_cna_val, data_meth_val, data_fprint_val, data_dep_val = prepare_data(val_loader)
data_mut_test, data_exp_test, data_cna_test, data_meth_test, data_fprint_test, data_dep_test = prepare_data(test_loader)

premodel_mut = pickle.load(open(f'./tcga_mut_ae_best_split_{split_num}.pickle', 'rb'))
premodel_exp = pickle.load(open(f'./tcga_exp_ae_best_split_{split_num}.pickle', 'rb'))
premodel_cna = pickle.load(open(f'./tcga_cna_ae_best_split_{split_num}.pickle', 'rb'))
premodel_meth = pickle.load(open(f'./tcga_meth_ae_best_split_{split_num}.pickle', 'rb'))

activation_func = 'relu'
activation_func2 = 'linear'
init = 'he_uniform'
dense_layer_dim = 100
num_epoch = 100
num_DepOI = 1298

input_mut = Input(shape=(premodel_mut[0][0].shape[0],))
input_exp = Input(shape=(premodel_exp[0][0].shape[0],))
input_cna = Input(shape=(premodel_cna[0][0].shape[0],))
input_meth = Input(shape=(premodel_meth[0][0].shape[0],))
input_gene = Input(shape=(data_fprint_train.shape[1],))

model_mut = Dense(1000, activation=activation_func)(input_mut)
model_mut = Dense(100, activation=activation_func)(model_mut)
model_mut = Dense(50, activation=activation_func)(model_mut)

model_exp = Dense(500, activation=activation_func)(input_exp)
model_exp = Dense(200, activation=activation_func)(model_exp)
model_exp = Dense(50, activation=activation_func)(model_exp)

model_cna = Dense(500, activation=activation_func)(input_cna)
model_cna = Dense(200, activation=activation_func)(model_cna)
model_cna = Dense(50, activation=activation_func)(model_cna)

model_meth = Dense(500, activation=activation_func)(input_meth)
model_meth = Dense(200, activation=activation_func)(model_meth)
model_meth = Dense(50, activation=activation_func)(model_meth)

model_gene = Dense(1000, activation=activation_func, kernel_initializer=init)(input_gene)
model_gene = Dense(100, activation=activation_func, kernel_initializer=init)(model_gene)
model_gene = Dense(50, activation=activation_func, kernel_initializer=init)(model_gene)

merged = Concatenate()([model_mut, model_exp, model_cna, model_meth, model_gene])
x = Dense(dense_layer_dim, activation=activation_func, kernel_initializer=init)(merged)
x = Dense(dense_layer_dim, activation=activation_func, kernel_initializer=init)(x)
output = Dense(1, activation=activation_func2, kernel_initializer=init)(x)

model_final = models.Model(inputs=[input_mut, input_exp, input_cna, input_meth, input_gene], outputs=output)

model_final.compile(loss='mse', optimizer='adam')
history = EarlyStopping(monitor='val_loss', patience=3, mode='min')

train_predictions = []
for epoch in range(num_epoch):
    model_final.fit(
        [data_mut_train, data_exp_train, data_cna_train, data_meth_train, data_fprint_train],
        data_dep_train,
        epochs=1,
        validation_data=([data_mut_val, data_exp_val, data_cna_val, data_meth_val, data_fprint_val], data_dep_val),
        batch_size=batch_size,
        callbacks=[history]
    )

    train_loss = history.model.history.history['loss'][-1]
    val_loss = history.model.history.history['val_loss'][-1]

    val_predictions = model_final.predict([data_mut_val, data_exp_val, data_cna_val, data_meth_val, data_fprint_val])
    pearson_corr_val, _ = pearsonr(val_predictions.flatten(), data_dep_val.flatten())

    print(f"Val Pearson correlation: {pearson_corr_val}")
    
train_predictions = model_final.predict([data_mut_test, data_exp_test, data_cna_test, data_meth_test, data_fprint_test])

test_loss = model_final.evaluate([data_mut_test, data_exp_test, data_cna_test, data_meth_test, data_fprint_test], data_dep_test)
test_predictions = model_final.predict([data_mut_test, data_exp_test, data_cna_test, data_meth_test, data_fprint_test])
pearson_corr_test, _ = pearsonr(test_predictions.flatten(), data_dep_test.flatten())

np.savetxt(f'./y_true_train_DeepDep_Split_{split_num}.txt', data_dep_train.flatten(), fmt='%.6f')
np.savetxt(f'./y_pred_train_DeepDep_Split_{split_num}.txt', train_predictions.flatten(), fmt='%.6f')

np.savetxt(f'./y_true_test_DeepDep_Split_{split_num}.txt', data_dep_test.flatten(), fmt='%.6f')
np.savetxt(f'./y_pred_test_DeepDep_Split_{split_num}.txt', test_predictions.flatten(), fmt='%.6f')

print(f"Test loss: {test_loss}")
print(f"Test Pearson correlation: {pearson_corr_test}")

model_final.save(f"./deepdep_model_full_{split_num}.h5")
