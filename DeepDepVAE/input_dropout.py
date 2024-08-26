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

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent space distribution.
        
        :param mu: Mean of the latent space
        :param logvar: Log variance of the latent space
        :return: Sampled latent variable
        """
        std = torch.exp(0.5 * logvar)
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
        :return: Reconstructed data, mean, and log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

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

    def forward(self, mut, exp, cna, meth, fprint, drop_mask=None):
        """
        Forward pass through the VAE_DeepDEP model.
        
        :param mut: Mutation data input
        :param exp: Expression data input
        :param cna: Copy number alteration data input
        :param meth: Methylation data input
        :param fprint: Fingerprint data input
        :param drop_mask: List of binary values indicating which modalities to drop
        :return: Prediction output
        """
        recon_mut, mu_mut, logvar_mut = self.vae_mut(mut)
        recon_exp, mu_exp, logvar_exp = self.vae_exp(exp)
        recon_cna, mu_cna, logvar_cna = self.vae_cna(cna)
        recon_meth, mu_meth, logvar_meth = self.vae_meth(meth)
        recon_gene, mu_fprint, logvar_gene = self.vae_fprint(fprint)

        if drop_mask is not None:
            mu_mut = self.apply_dropout(mu_mut, drop_mask[0])
            mu_exp = self.apply_dropout(mu_exp, drop_mask[1])
            mu_cna = self.apply_dropout(mu_cna, drop_mask[2])
            mu_meth = self.apply_dropout(mu_meth, drop_mask[3])
            mu_fprint = self.apply_dropout(mu_fprint, drop_mask[4])
        
        merged = torch.cat([mu_mut, mu_exp, mu_cna, mu_meth, mu_fprint], dim=1)
        merged = torch.relu(self.fc_merged1(merged))
        merged = torch.relu(self.fc_merged2(merged))
        output = self.fc_out(merged)
        return output
    
    def apply_dropout(self, x, drop):
        """
        Applies dropout to the input based on the drop mask.
        
        :param x: Input tensor
        :param drop: Boolean flag indicating whether to drop this modality
        :return: Modified input tensor
        """
        if drop:
            return torch.zeros_like(x)
        return x


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

def train_model(model, train_loader, val_loader, num_epoch, learning_rate, p_drop):
    """
    Trains the model
    
    :param model: The model
    :param train_loader: DataLoader for the training data
    :param val_loader: DataLoader for the validation data
    :param num_epoch: Number of epochs
    :param learning_rate: Learning rate
    :param p_drop: Probability of applying dropout to the input
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
        
            # drop_mask = [torch.rand(1).item() < p_drop for _ in range(4)] + [False]  # Uncomment this if you don't want to dropout fingerprint in training
            drop_mask = [torch.rand(1).item() < p_drop for _ in range(5)]
            
            outputs = model(*inputs, drop_mask=drop_mask)
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
                outputs = model(*inputs, drop_mask=[0, 0, 0, 0, 0])
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                predictions.extend(outputs.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss}")

        predictions = np.array(predictions).flatten()
        targets = np.array(targets_list).flatten()
        pearson_corr, _ = pearsonr(predictions, targets)
        print(f"Pearson Correlation: {pearson_corr}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict, './best_model_vae_input_dropout_newVAE_withsplit2_Best_Model.pth')
            print("Model saved")

    return best_model_state_dict, training_predictions, training_targets_list


def test_model(model, test_loader, device):
    """
    Evaluates the trained model with different dropout masks applied to the test data.
    
    :param model: The trained model
    :param test_loader: DataLoader for the test data
    :param device: Device to run the evaluation on (e.g., 'cpu' or 'mps')
    :return: A dictionary of evaluation results for different dropout masks
    """
    model.eval()
    drop_masks = [
        [1, 0, 0, 0, 0],  # mut dropped
        [0, 1, 0, 0, 0],  # exp dropped
        [0, 0, 1, 0, 0],  # cna dropped
        [0, 0, 0, 1, 0],  # meth dropped
        [0, 0, 0, 0, 1],  # fprint dropped
        [0, 0, 0, 0, 0],  # nothing dropped
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 1, 0],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 1],
        [1, 1, 0, 1, 0],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]
    ]

    results = {}
    with torch.no_grad():
        for mask in drop_masks:
            predictions = []
            targets_list = []
            for batch in test_loader:
                inputs = [tensor.to(device) for tensor in batch[:-1]]
                targets = batch[-1].to(device)
                outputs = model(*inputs, drop_mask=mask)
                predictions.extend(outputs.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())

            predictions = np.array(predictions).flatten()
            targets = np.array(targets_list).flatten()
            pearson_corr, _ = pearsonr(predictions, targets)
            mask_str = "_".join(map(str, mask))
            results[mask_str] = {
                "predictions": predictions,
                "targets": targets,
                "pearson_corr": pearson_corr
            }
            print(f"Mask: {mask_str}, Pearson Correlation: {pearson_corr}")
    
    return results



if __name__ == '__main__':

    split_num = 1

    with open(f'./data_split_{split_num}.pickle', 'rb') as f:
            train_dataset, val_dataset, test_dataset = pickle.load(f)

    learning_rate = 1e-3
    batch_size = 10000
    epochs = 100
    p_drop = 0.5 

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
    best_model_state_dict, training_predictions, training_targets_list = train_model(model, train_loader, val_loader, epochs, learning_rate, p_drop)
    model.load_state_dict(best_model_state_dict)
    
    results = test_model(model, test_loader, device)
    for mask, res in results.items():
        print(f"Mask: {mask}, Pearson Correlation: {res['pearson_corr']}")
        np.savetxt(f'./y_true_test_mask_{mask}_VAE_Input_Dropout.txt', res['targets'], fmt='%.6f')
        np.savetxt(f'./y_pred_test_mask_{mask}_VAE_Input_Dropout.txt', res['predictions'], fmt='%.6f')

    torch.save(best_model_state_dict, './vae_deepdep_model_input_dropout.pth')

    y_true_train = np.array(training_targets_list).flatten()
    y_pred_train = np.array(training_predictions).flatten()
    y_true_test = results["0_0_0_0_0"]["targets"].flatten()
    y_pred_test = results["0_0_0_0_0"]["predictions"].flatten()

    np.savetxt(f'./y_true_train_VAE_Input_Dropout.txt', y_true_train, fmt='%.6f')
    np.savetxt(f'./y_pred_train_VAE_Input_Dropout.txt', y_pred_train, fmt='%.6f')
    np.savetxt(f'./y_true_test_VAE_Input_Dropout.txt', y_true_test, fmt='%.6f')
    np.savetxt(f'./y_pred_test_VAE_Input_Dropout.txt', y_pred_test, fmt='%.6f')

