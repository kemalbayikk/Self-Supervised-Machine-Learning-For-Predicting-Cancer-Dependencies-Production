import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class MaskedAutoencoder(nn.Module):
    """
    Defines the Masked Autoencoder (MAE) architecture.
    
    :param input_dim: Input dimension
    :param first_layer_dim: Dimension of the first hidden layer
    :param second_layer_dim: Dimension of the second hidden layer
    :param latent_dim: Dimension of the latent space
    """
    def __init__(self, input_dim, first_layer_dim, second_layer_dim, latent_dim):
        super(MaskedAutoencoder, self).__init__()
        self.encoder_fc1 = nn.Linear(input_dim, first_layer_dim)
        self.encoder_fc2 = nn.Linear(first_layer_dim, second_layer_dim)
        self.encoder_fc3 = nn.Linear(second_layer_dim, latent_dim)
        self.decoder_fc1 = nn.Linear(latent_dim, second_layer_dim)
        self.decoder_fc2 = nn.Linear(second_layer_dim, first_layer_dim)
        self.decoder_fc3 = nn.Linear(first_layer_dim, input_dim)

    def forward(self, x):
        """
        Forward pass through the encoder to produce latent representation.
        
        :param x: Input data
        :return: Latent representation of the input data
        """

        encoded = torch.relu(self.encoder_fc1(x))
        encoded = torch.relu(self.encoder_fc2(encoded))
        latent = self.encoder_fc3(encoded)

        return latent

class MAE_DeepDEP(nn.Module):
    """
    Defines the MAE_DeepDep Model.
    
    :param premodel_mut: Pretrained MAE for mutation data
    :param premodel_exp: Pretrained MAE for expression data
    :param premodel_cna: Pretrained MAE for copy number alteration data
    :param premodel_meth: Pretrained MAE for methylation data
    :param premodel_fprint: Pretrained MAE for fingerprint data
    :param latent_dim: Latent dimension size
    :param dense_layer_dim: Dimension of the dense layers
    """
    def __init__(self, premodel_mut, premodel_exp, premodel_cna, premodel_meth, premodel_fprint, latent_dim, dense_layer_dim):
        super(MAE_DeepDEP, self).__init__()
        self.mae_mut = premodel_mut
        self.mae_exp = premodel_exp
        self.mae_cna = premodel_cna
        self.mae_meth = premodel_meth
        self.mae_fprint = premodel_fprint

        latent_dim_total = latent_dim * 5 
        self.fc_merged1 = nn.Linear(latent_dim_total, dense_layer_dim)
        self.fc_merged2 = nn.Linear(dense_layer_dim, dense_layer_dim)
        self.fc_out = nn.Linear(dense_layer_dim, 1)

    def forward(self, mut, exp, cna, meth, fprint):
        """
        Forward pass through the MAE_DeepDep Model.
        
        :param mut: Mutation data input
        :param exp: Expression data input
        :param cna: Copy number alteration data input
        :param meth: Methylation data input
        :param fprint: Fingerprint data input
        :return: Prediction output
        """
        latent_mut = self.mae_mut(mut)
        latent_exp = self.mae_exp(exp)
        latent_cna = self.mae_cna(cna)
        latent_meth = self.mae_meth(meth)
        latent_fprint = self.mae_fprint(fprint)
        
        merged = torch.cat([latent_mut, latent_exp, latent_cna, latent_meth, latent_fprint], dim=1)
        merged = torch.relu(self.fc_merged1(merged))
        merged = torch.relu(self.fc_merged2(merged))
        output = self.fc_out(merged)
        return output

def load_pretrained_mae(filepath, input_dim, first_layer_dim, second_layer_dim, latent_dim):
    """
    Loads a pretrained Masked Autoencoder (MAE) from a pickle file.
    
    :param filepath: Path to the pretrained model file
    :param input_dim: Input dimension of the MAE
    :param first_layer_dim: First hidden layer dimension
    :param second_layer_dim: Second hidden layer dimension
    :param latent_dim: Latent space dimension
    :return: Loaded MAE model
    """
    mae = MaskedAutoencoder(input_dim, first_layer_dim, second_layer_dim, latent_dim)
    mae_state = pickle.load(open(filepath, 'rb'))
    
    for key in mae_state:
        if isinstance(mae_state[key], np.ndarray):
            mae_state[key] = torch.tensor(mae_state[key])
    
    mae.load_state_dict(mae_state)
    return mae

def train_model(model, train_loader, val_loader, num_epoch, learning_rate, split_num):
    """
    Trains the model using the provided training and validation data.
    
    :param model: The model
    :param train_loader: DataLoader for the training data
    :param val_loader: DataLoader for the validation data
    :param num_epoch: Number of epochs
    :param learning_rate: Learning rate
    :param split_num: Data split number
    :return: The best model state dictionary, training predictions, and training targets
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    best_loss = float('inf')

    for epoch in range(num_epoch):

        training_predictions = []
        training_targets_list = []

        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}", leave=False)
        for batch in progress_bar:
            inputs = [tensor.to(device) for tensor in batch[:-1]]
            targets = batch[-1].to(device)

            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            training_predictions.extend(outputs.detach().cpu().numpy())
            training_targets_list.extend(targets.detach().cpu().numpy())

        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}")

        model.eval()
        val_loss = 0.0
        predictions = []
        targets_list = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = [tensor.to(device) for tensor in batch[:-1]]
                targets = batch[-1].to(device)
                outputs = model(*inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                predictions.extend(outputs.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        print(f"Val Loss: {val_loss}")

        predictions = np.array(predictions).flatten()
        targets = np.array(targets_list).flatten()
        pearson_corr, _ = pearsonr(predictions, targets)
        print(f"Pearson Correlation: {pearson_corr}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict, f'./MAE_DeepDep_Prediction_Network_Split_{split_num}.pth')
            print("Model saved")

    return best_model_state_dict, training_predictions, training_targets_list

if __name__ == '__main__':
    
    for split_num in range(1, 6):

        with open(f'./data_split_{split_num}.pickle', 'rb') as f:
            train_dataset, val_dataset, test_dataset = pickle.load(f)

        learning_rate = 1e-3
        batch_size = 10000
        epochs = 100
        latent_dim = 50

        dims_mut = (train_dataset[:][0].shape[1], 1000, 100, 50)
        dims_exp = (train_dataset[:][1].shape[1], 500, 200, 50)
        dims_cna = (train_dataset[:][2].shape[1], 500, 200, 50)
        dims_meth = (train_dataset[:][3].shape[1], 500, 200, 50)
        dims_fprint = (train_dataset[:][4].shape[1], 1000, 100, 50)
          
        premodel_mut = load_pretrained_mae(f'./ccl_mut_mae_best_split_{split_num}.pickle', *dims_mut)
        premodel_exp = load_pretrained_mae(f'./ccl_exp_mae_best_split_{split_num}.pickle', *dims_exp)
        premodel_cna = load_pretrained_mae(f'./ccl_cna_mae_best_split_{split_num}.pickle', *dims_cna)
        premodel_meth = load_pretrained_mae(f'./ccl_meth_mae_best_split_{split_num}.pickle', *dims_meth)
        premodel_fprint = load_pretrained_mae(f'./ccl_fprint_mae_best_split_{split_num}.pickle', *dims_fprint)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = MAE_DeepDEP(premodel_mut, premodel_exp, premodel_cna, premodel_meth, premodel_fprint, latent_dim, 250)
        best_model_state_dict, training_predictions, training_targets_list = train_model(model, train_loader, val_loader, epochs, learning_rate, split_num)

        model.load_state_dict(best_model_state_dict)
        model.eval()
        predictions = []
        targets_list = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = [tensor.to(device) for tensor in batch[:-1]]
                targets = batch[-1].to(device)
                outputs = model(*inputs)
                predictions.extend(outputs.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())

        predictions = np.array(predictions).flatten()
        targets = np.array(targets_list).flatten()
        pearson_corr, _ = pearsonr(predictions, targets)
        print(f"Test Pearson Correlation: {pearson_corr}")

        y_true_train = np.array(training_targets_list).flatten()
        y_pred_train = np.array(training_predictions).flatten()
        y_true_test = np.array(targets_list).flatten()
        y_pred_test = np.array(predictions).flatten()

        np.savetxt(f'./y_true_train_CCL_MAE_Split_{split_num}.txt', y_true_train, fmt='%.6f')
        np.savetxt(f'./y_pred_train_CCL_MAE_Split_{split_num}.txt', y_pred_train, fmt='%.6f')
        np.savetxt(f'./y_true_test_CCL_MAE_Split_{split_num}.txt', y_true_test, fmt='%.6f')
        np.savetxt(f'./y_pred_test_CCL_MAE_Split_{split_num}.txt', y_pred_test, fmt='%.6f')