import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class VariationalAutoencoder(nn.Module):
    """
    Defines the Variational Autoencoder (VAE) architecture.
    
    :param input_dim: Input dimension
    :param first_layer_dim: Dimension of the first hidden layer
    :param second_layer_dim: Dimension of the second hidden layer
    :param latent_dim: Dimension of the latent space
    """
    def __init__(self, input_dim, first_layer_dim, second_layer_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, first_layer_dim)
        self.fc2 = nn.Linear(first_layer_dim, second_layer_dim)
        self.fc31 = nn.Linear(second_layer_dim, latent_dim)
        self.fc32 = nn.Linear(second_layer_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, second_layer_dim)
        self.fc5 = nn.Linear(second_layer_dim, first_layer_dim)
        self.fc6 = nn.Linear(first_layer_dim, input_dim)

    def encode(self, x):
        """
        Encodes the input data into latent space by calculating the mean (mu) and log variance (logvar).
        
        :param x: Input data
        :return: mu and logvar of the latent space
        """
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar, var_scale=1.0):
        """
        Reparameterization trick to sample from the latent space distribution.
        
        :param mu: Mean of the latent space
        :param logvar: Log variance of the latent space
        :return: Sampled latent variable
        """
        std = torch.exp(0.5 * logvar * var_scale)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decodes the latent variable back into the original data space.
        
        :param z: Latent variable
        :return: Reconstructed data
        """
        h3 = torch.relu(self.fc4(z))
        h4 = torch.relu(self.fc5(h3))
        return self.fc6(h4)

    def forward(self, x):
        """
        Forward pass through the VAE model: encode, reparameterize, and decode.
        
        :param x: Input data
        :return: Latent variable (z), mu (mean), and logvar (log variance)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class VAE_DeepDEP(nn.Module):
    """
    Defines the VAE_DeepDep model
    
    :param premodel_mut: Pretrained VAE for mutation data
    :param premodel_exp: Pretrained VAE for expression data
    :param premodel_cna: Pretrained VAE for copy number alteration data
    :param premodel_meth: Pretrained VAE for methylation data
    :param premodel_fprint: Pretrained VAE for fingerprint data
    :param dense_layer_dim: Dimension of the dense layers
    """
    def __init__(self, premodel_mut, premodel_exp, premodel_cna, premodel_meth, premodel_fprint, dense_layer_dim):
        super(VAE_DeepDEP, self).__init__()
        self.vae_mut = premodel_mut
        self.vae_exp = premodel_exp
        self.vae_cna = premodel_cna
        self.vae_meth = premodel_meth
        self.vae_fprint = premodel_fprint

        self.fc_merged1 = nn.Linear(250, dense_layer_dim)
        self.fc_merged2 = nn.Linear(dense_layer_dim, dense_layer_dim)
        self.fc_out = nn.Linear(dense_layer_dim, 1)

    def forward(self, mut, exp, cna, meth, fprint):
        """
        Forward pass through the VAE_DeepDEP model.
        
        :param mut: Mutation data input
        :param exp: Expression data input
        :param cna: Copy number alteration data input
        :param meth: Methylation data input
        :param fprint: Fingerprint data input
        :return: Prediction output
        """
        z_mut, mu_mut, logvar_mut = self.vae_mut(mut)
        z_exp, mu_exp, logvar_exp = self.vae_exp(exp)
        z_cna, mu_cna, logvar_cna = self.vae_cna(cna)
        z_meth, mu_meth, logvar_meth = self.vae_meth(meth)
        z_fprint, mu_fprint, logvar_gene = self.vae_fprint(fprint)
        
        merged = torch.cat([mu_mut, mu_exp, mu_cna, mu_meth, mu_fprint], dim=1)
        merged = torch.relu(self.fc_merged1(merged))
        merged = torch.relu(self.fc_merged2(merged))
        output = self.fc_out(merged)
        return output

def load_pretrained_vae(filepath, input_dim, first_layer_dim, second_layer_dim, latent_dim):
    """
    Loads a pretrained VAE model from a given file path.
    
    :param filepath: Path to the pretrained model file
    :param input_dim: Input dimension of the VAE
    :param first_layer_dim: First hidden layer dimension
    :param second_layer_dim: Second hidden layer dimension
    :param latent_dim: Latent space dimension
    :return: Loaded VAE model
    """
    vae = VariationalAutoencoder(input_dim, first_layer_dim, second_layer_dim, latent_dim)
    vae_state = pickle.load(open(filepath, 'rb'))

    for key in vae_state:
        if isinstance(vae_state[key], np.ndarray):
            vae_state[key] = torch.tensor(vae_state[key])

    vae.load_state_dict(vae_state)
    return vae

def train_model(model, train_loader, test_loader, num_epoch, learning_rate, split_num):
    """
    Trains the model.
    
    :param model: The model
    :param train_loader: DataLoader for the training data
    :param test_loader: DataLoader for the test data
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
        test_loss = 0.0
        predictions = []
        targets_list = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = [tensor.to(device) for tensor in batch[:-1]]
                targets = batch[-1].to(device)
                outputs = model(*inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                predictions.extend(outputs.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())

        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss}")

        predictions = np.array(predictions).flatten()
        targets = np.array(targets_list).flatten()
        pearson_corr, _ = pearsonr(predictions, targets)
        print(f"Pearson Correlation: {pearson_corr}")

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict, f'./VAE_Prediction_Network_Split_{split_num}.pth')
            print("Model saved")

    return best_model_state_dict, training_predictions, training_targets_list

if __name__ == '__main__':

    for split_num in range(1,6):
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
  
        premodel_mut = load_pretrained_vae(f'./ccl_finetuned_mut_vae_deepdep_best_split_{split_num}.pickle', *dims_mut)
        premodel_exp = load_pretrained_vae(f'./ccl_finetuned_exp_vae_deepdep_best_split_{split_num}.pickle', *dims_exp)
        premodel_cna = load_pretrained_vae(f'./ccl_finetuned_cna_vae_deepdep_best_split_{split_num}.pickle', *dims_cna)
        premodel_meth = load_pretrained_vae(f'./ccl_finetuned_meth_vae_deepdep_best_split_{split_num}.pickle', *dims_meth)
        premodel_fprint = load_pretrained_vae(f'./ccl_finetuned_fprint_vae_deepdep_best_split_{split_num}.pickle', *dims_fprint)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = VAE_DeepDEP(premodel_mut, premodel_exp, premodel_cna, premodel_meth, premodel_fprint, 250)
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

        np.savetxt(f'./y_true_train_Prediction_Network_VAE_Split_{split_num}.txt', y_true_train, fmt='%.6f')
        np.savetxt(f'./y_pred_train_Prediction_Network_VAE_Split_{split_num}.txt', y_pred_train, fmt='%.6f')
        np.savetxt(f'./y_true_test_Prediction_Network_VAE_Split_{split_num}.txt', y_true_test, fmt='%.6f')
        np.savetxt(f'./y_pred_test_Prediction_Network_VAE_Split_{split_num}.txt', y_pred_test, fmt='%.6f')
