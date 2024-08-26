import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pickle
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
        :return: Reconstructed data, mean, and log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, data_name, beta):
    """
    Calculates the VAE loss, which includes reconstruction loss and KL divergence.
    
    :param recon_x: Reconstructed data
    :param x: Original input data
    :param mu: Mean of the latent space
    :param logvar: Log variance of the latent space
    :param data_name: Type of data (e.g., "mut" for mutation)
    :param beta: Beta value to weight the KL divergence term
    :return: Total loss, reconstruction loss, and KL divergence loss
    """
    if data_name == "mut" or data_name == "fprint":
        recon_loss = nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    else:
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

    kl_loss = beta * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

def train_vae(model, train_loader, val_loader, num_epochs, learning_rate, device, data_name, beta):
    """
    Trains the VAE model using the provided training and validation data.
    
    :param model: The VAE model
    :param train_loader: DataLoader for the training data
    :param val_loader: DataLoader for the validation data
    :param num_epochs: Number of epochs
    :param learning_rate: Learning rate
    :param device: Device to run the training on (e.g., 'cpu' or 'mps')
    :param data_name: Type of data (e.g., "mut" for mutation)
    :param beta: Beta value to weight the KL divergence term
    :return: Trained VAE model
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data in tqdm(train_loader):
            inputs = data[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(inputs)
            loss, _, _ = vae_loss_function(recon_batch, inputs, mu, logvar, data_name, beta)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss /= len(train_loader.dataset)


        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        with torch.no_grad():
            for data in val_loader:
                inputs = data[0].to(device)
                recon_batch, mu, logvar = model(inputs)
                loss, recon_loss, kl_loss = vae_loss_function(recon_batch, inputs, mu, logvar, data_name, beta)
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
        val_loss /= len(val_loader.dataset)
        val_recon_loss /= len(val_loader.dataset) 
        val_kl_loss /= len(val_loader.dataset) 

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Recon Loss: {val_recon_loss:.6f}, KL Loss: {val_kl_loss:.6f}')

        if val_loss < best_loss:
            best_loss = val_loss
            save_weights_to_pickle(model, f'./ccl_finetuned_{data_name}_vae_deepdep_best_split_{split_num}.pickle')

    return model

def evaluate_vae(model, test_loader, device, data_name, beta):
    """
    Evaluates the VAE model using the provided test data.
    
    :param model: The trained VAE model
    :param test_loader: DataLoader for the test data
    :param device: Device to run the evaluation on (e.g., 'cpu' or 'mps')
    :param data_name: Type of data (e.g., "mut" for mutation)
    :param beta: Beta value to weight the KL divergence term
    """
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)
            recon_batch, mu, logvar = model(inputs)
            loss, recon_loss, kl_loss = vae_loss_function(recon_batch, inputs, mu, logvar, data_name, beta)
            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_kl_loss += kl_loss.item()
    test_loss /= len(test_loader.dataset)
    test_recon_loss /= len(test_loader.dataset)
    test_kl_loss /= len(test_loader.dataset)

    print(f'Test Loss: {test_loss:.6f}, Test Recon Loss: {test_recon_loss:.6f}, Test KL Loss: {test_kl_loss:.6f}')

def save_weights_to_pickle(model, file_name):
    """
    Saves the model's weights to a pickle file.
    
    :param model: The model whose weights are to be saved
    :param file_name: Path to save the pickle file
    """
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    weights = {name: param.to('cpu').detach().numpy() for name, param in model.named_parameters()}
    with open(file_name, 'wb') as handle:
        pickle.dump(weights, handle)
    print(f"Model weights saved to {file_name}")

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

if __name__ == '__main__':

    for split_num in range(1, 6):
        with open(f'./data_split_{split_num}.pickle', 'rb') as f:
            train_dataset, val_dataset, test_dataset = pickle.load(f)

        data_dict = {
            'mut': {
                'train':train_dataset[:][0],
                'val':val_dataset[:][0],
                'test':test_dataset[:][0]
                },  
            'exp': {
                'train':train_dataset[:][1],
                'val':val_dataset[:][1],
                'test':test_dataset[:][1]
                },
            'cna': {
                'train':train_dataset[:][2],
                'val':val_dataset[:][2],
                'test':test_dataset[:][2]
                },
            'meth': {
                'train':train_dataset[:][3],
                'val':val_dataset[:][3],
                'test':test_dataset[:][3]
                },
            'fprint': {
                'train':train_dataset[:][4],
                'val':val_dataset[:][4],
                'test':test_dataset[:][4]
                },
        }

        for data_type, data_ccl in data_dict.items():

            learning_rate = 1e-3
            batch_size = 10000
            epochs = 100
            beta = 1

            train_tensors = torch.tensor(data_ccl["train"], dtype=torch.float32).to(device)
            val_tensors = torch.tensor(data_ccl["val"], dtype=torch.float32).to(device)
            test_tensors = torch.tensor(data_ccl["test"], dtype=torch.float32).to(device)

            train_loader = DataLoader(TensorDataset(train_tensors), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(val_tensors), batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(TensorDataset(test_tensors), batch_size=batch_size, shuffle=False)

            if data_type == 'mut':
                vae = load_pretrained_vae(f'tcga_mut_vae_deepdep_best_split_{split_num}.pickle', train_tensors.shape[1], 1000, 100, 50)
            elif data_type == 'exp':
                vae = load_pretrained_vae(f'tcga_exp_vae_deepdep_best_split_{split_num}.pickle', train_tensors.shape[1], 500, 200, 50)
            elif data_type == 'cna':
                vae = load_pretrained_vae(f'tcga_cna_vae_deepdep_best_split_{split_num}.pickle', train_tensors.shape[1], 500, 200, 50)
            elif data_type == 'meth':
                vae = load_pretrained_vae(f'tcga_meth_vae_deepdep_best_split_{split_num}.pickle', train_tensors.shape[1], 500, 200, 50)
            elif data_type == 'fprint':
                vae = VariationalAutoencoder(input_dim=train_tensors.shape[1], first_layer_dim=1000, second_layer_dim=100, latent_dim=50)
            
            trained_vae = train_vae(vae, train_loader, val_loader, num_epochs=epochs, learning_rate=learning_rate, device=device, data_name=data_type, beta=beta)
            evaluate_vae(trained_vae, test_loader, device, data_name=data_type, beta=beta)
