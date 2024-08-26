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
    
    :param split_num: The split number
    :param omic: Type of omic data (e.g., "exp", "cna", "meth")
    :param base_path: Base path to the data files
    :return: Loaded training, validation, and test datasets
    """
    train_data = torch.load(os.path.join(base_path, f'TCGASplits/split_{split_num}/train_dataset_{omic}_split_{split_num}.pth'))
    val_data = torch.load(os.path.join(base_path, f'TCGASplits/split_{split_num}/val_dataset_{omic}_split_{split_num}.pth'))
    test_data = torch.load(os.path.join(base_path, f'TCGASplits/split_{split_num}/test_dataset_{omic}_split_{split_num}.pth'))

    return train_data, val_data, test_data

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

    def encode(self, x_masked):
        """
        Encodes the masked input data into a latent space.
        
        :param x_masked: Masked input data
        :return: Latent representation
        """
        encoded = torch.relu(self.encoder_fc1(x_masked))
        encoded = torch.relu(self.encoder_fc2(encoded))
        latent = self.encoder_fc3(encoded)
        return latent
    
    def decode(self, latent):
        """
        Decodes the latent representation back into the original input space.
        
        :param latent: Latent representation
        :return: Reconstructed input data
        """
        decoded = torch.relu(self.decoder_fc1(latent))
        decoded = torch.relu(self.decoder_fc2(decoded))
        reconstructed = self.decoder_fc3(decoded)
        return reconstructed

    def forward(self, x, mask_ratio=0.75):
        """
        Forward pass through the Masked Autoencoder. Applies masking, encodes, and reconstructs.
        
        :param x: Input data
        :param mask_ratio: Ratio of input data to be masked (default is 75%)
        :return: Reconstructed data and the mask used
        """
        mask = torch.rand(x.shape).to(x.device) < mask_ratio
        x_masked = x * mask.float()

        latent = self.encode(x_masked)
        reconstructed = self.decode(latent)

        return reconstructed, mask

def mae_loss_function(recon_x, x, mask, data_name):
    """
    Calculates the loss for the Masked Autoencoder, either binary cross-entropy or MSE.
    
    :param recon_x: Reconstructed data
    :param x: Original input data
    :param mask: The mask applied to the input data
    :param data_name: Type of data (e.g., "mut" for mutation data)
    :return: Loss value
    """
    if data_name == "mut":
        loss = nn.functional.binary_cross_entropy_with_logits(recon_x * mask.float(), x * mask.float(), reduction='sum') / mask.float().sum()
    else:
        loss = nn.functional.mse_loss(recon_x * mask.float(), x * mask.float(), reduction='sum') / mask.float().sum()
    return loss

def save_weights_to_pickle(model, file_name):
    """
    Saves the model's weights to a pickle file.
    
    :param model: The model
    :param file_name: Path to save the pickle file
    """
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    weights = {name: param.to('cpu').detach().numpy() for name, param in model.named_parameters()}
    with open(file_name, 'wb') as handle:
        pickle.dump(weights, handle)
    print(f"Model weights saved to {file_name}")

if __name__ == '__main__':
    omics = ["cna", "exp", "meth"]
    # omics = ["mut"] # Uncomment this line to process mutation data instead

    for omic in omics:
        for split_num in range(1, 6):
            learning_rate = 1e-3
            batch_size = 500
            epochs = 100
            first_layer_dim = 500 # Change it to 1000 for mut
            second_layer_dim = 200 # Change it to 100 for mut
            latent_dim = 50

            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

            train_data, val_data, test_data = load_split(split_num, omic, "")

            input_dim = train_data.dataset.tensors[0].shape[1]

            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

            model = MaskedAutoencoder(input_dim=input_dim, first_layer_dim=first_layer_dim, second_layer_dim=second_layer_dim, latent_dim=latent_dim)
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
                    recon_batch, mask = model(inputs)
                    loss = mae_loss_function(recon_batch, inputs, mask, omic)
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
                        recon_batch, mask = model(inputs)
                        loss = mae_loss_function(recon_batch, inputs, mask, omic)
                        val_loss += loss.item()
                val_loss /= len(val_loader.dataset)

                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')

                if val_loss < best_loss:
                    best_loss = val_loss
                    save_weights_to_pickle(model, f'./tcga_{omic}_mae_deepdep_best_split_{split_num}.pickle')

            model.eval()
            test_loss = 0
            test_recon_loss = 0
            test_kl_loss = 0
            with torch.no_grad():
                for data in test_loader:
                    inputs = data[0].to(device)
                    recon_batch, mask = model(inputs)
                    loss = mae_loss_function(recon_batch, inputs, mask, omic)
                    test_loss += loss.item()
            test_loss /= len(test_loader.dataset)
            test_recon_loss /= len(test_loader.dataset)
            test_kl_loss /= len(test_loader.dataset)

