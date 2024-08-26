import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle

def load_data(filename):
    """
    Loads data from a given file, converts it to a PyTorch tensor.
    
    :param filename: Path to the data file
    :return: Data loaded as a PyTorch tensor
    """
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines()

        for line in lines[1:]:
            values = line.strip().split('\t')[1:]
            data.append(values)

    data = np.array(data, dtype='float32')
    return torch.tensor(data, dtype=torch.float32)

def load_split(split_num, omic, base_path=''):
    """
    Loads training, validation, and test datasets for a given omic type and data split number.
    
    :param split_num: The split number of the data (e.g., 1, 2, 3, etc.)
    :param omic: Type of omic data (e.g., "exp", "cna", "meth")
    :param base_path: Base path to the data files (optional)
    :return: Loaded training, validation, and test datasets
    """
    train_data = torch.load(os.path.join(base_path, f'TCGASplits/split_{split_num}/train_dataset_{omic}_split_{split_num}.pth'))
    val_data = torch.load(os.path.join(base_path, f'TCGASplits/split_{split_num}/val_dataset_{omic}_split_{split_num}.pth'))
    test_data = torch.load(os.path.join(base_path, f'TCGASplits/split_{split_num}/test_dataset_{omic}_split_{split_num}.pth'))

    return train_data, val_data, test_data

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
    if data_name == "mut":
        recon_loss = nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    else:
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

    kl_loss = beta * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

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

if __name__ == '__main__':
    omics = ["exp","cna","meth"]
    # omics = ["mut"] # Uncomment this line to process mutation data instead
    for omic in omics:
        for split_num in range(1, 6):
            learning_rate = 1e-3
            batch_size = 500
            epochs = 100
            first_layer_dim = 500 # Change it to 1000 for mut
            second_layer_dim = 200 # Change it to 100 for mut
            latent_dim = 50
            beta = 1

            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

            train_data, val_data, test_data = load_split(split_num, omic, "")

            input_dim = train_data.dataset.tensors[0].shape[1]

            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

            model = VariationalAutoencoder(input_dim=input_dim, first_layer_dim=first_layer_dim, second_layer_dim=second_layer_dim, latent_dim=latent_dim)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            best_loss = float('inf')
            early_stop_counter = 0

            for epoch in range(epochs):
                model.train()
                train_loss = 0
                for data in train_loader:
                    inputs = data[0].to(device)
                    optimizer.zero_grad()
                    recon_batch, mu, logvar = model(inputs)
                    loss, recon_loss, kl_loss = vae_loss_function(recon_batch, inputs, mu, logvar, omic, beta)
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
                        loss, recon_loss, kl_loss = vae_loss_function(recon_batch, inputs, mu, logvar, omic, beta)
                        val_recon_loss += recon_loss.item()
                        val_kl_loss += kl_loss.item()
                        val_loss += loss.item()
                val_loss /= len(val_loader.dataset)
                val_recon_loss /= len(val_loader.dataset) 
                val_kl_loss /= len(val_loader.dataset) 

                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, Recon Loss: {val_recon_loss:.6f}, KL Loss: {val_kl_loss:.6f}')

                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stop_counter = 0
                    save_weights_to_pickle(model, f'./tcga_{omic}_vae_deepdep_best_split_{split_num}.pickle')

            model.eval()
            test_loss = 0
            test_recon_loss = 0
            test_kl_loss = 0
            with torch.no_grad():
                for data in test_loader:
                    inputs = data[0].to(device)
                    recon_batch, mu, logvar = model(inputs)
                    loss, recon_loss, kl_loss = vae_loss_function(recon_batch, inputs, mu, logvar, omic, beta)
                    test_recon_loss += recon_loss.item()
                    test_kl_loss += kl_loss.item()
                    test_loss += loss.item()
            test_loss /= len(test_loader.dataset)
            test_recon_loss /= len(test_loader.dataset)
            test_kl_loss /= len(test_loader.dataset)

            print(f'\nTest Loss: {test_loss:.6f}, Test Recon Loss: {test_recon_loss:.6f}, Test KL Loss: {test_kl_loss:.6f}')


