import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pickle
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
    if data_name == "mut" or data_name == "fprint":
        loss = nn.functional.binary_cross_entropy_with_logits(recon_x * mask.float(), x * mask.float(), reduction='sum') / mask.float().sum()
    else:
        loss = nn.functional.mse_loss(recon_x * mask.float(), x * mask.float(), reduction='sum') / mask.float().sum()
    return loss

def train_mae(model, train_loader, val_loader, num_epochs, learning_rate, device, data_name):
    """
    Trains the Masked Autoencoder (MAE) model using the provided training and validation data.
    
    :param model: The MAE model
    :param train_loader: DataLoader for the training data
    :param val_loader: DataLoader for the validation data
    :param num_epochs: Number of epochs
    :param learning_rate: Learning rate
    :param device: Device to run the training on (e.g., 'cpu' or 'mps')
    :param data_name: Name of the data type (e.g., 'mut', 'exp', etc.)
    :return: Trained MAE model
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
            recon_batch, mask = model(inputs)
            loss = mae_loss_function(recon_batch, inputs, mask, data_name)
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
                loss = mae_loss_function(recon_batch, inputs, mask, data_name)
                val_loss += loss.item()
        val_loss /= len(val_loader.dataset)
        val_recon_loss /= len(val_loader.dataset) 
        val_kl_loss /= len(val_loader.dataset) 

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        if val_loss < best_loss:
            best_loss = val_loss
            save_weights_to_pickle(model, f'./ccl_{data_name}_mae_deepdep_best_split_{split_num}.pickle')

    return model

def evaluate_mae(model, test_loader, device, data_name):
    """
    Evaluates the trained Masked Autoencoder (MAE) model on the test data.
    
    :param model: The trained MAE model
    :param test_loader: DataLoader for the test data
    :param device: Device to run the evaluation on (e.g., 'cpu' or 'mps')
    :param data_name: Name of the data type (e.g., 'mut', 'exp', etc.)
    """
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)
            recon_batch, mask = model(inputs)
            loss = mae_loss_function(recon_batch, inputs, mask, data_name)
            test_loss += loss.item()
    test_loss /= len(test_loader.dataset)
    test_recon_loss /= len(test_loader.dataset)
    test_kl_loss /= len(test_loader.dataset)

    print(f'Test Loss: {test_loss:.6f}')

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

def load_pretrained_mae(filepath, input_dim, first_layer_dim, second_layer_dim, latent_dim):
    """
    Loads a pretrained Masked Autoencoder (MAE) model from a pickle file.
    
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

            train_tensors = torch.tensor(data_ccl["train"], dtype=torch.float32).to(device)
            val_tensors = torch.tensor(data_ccl["val"], dtype=torch.float32).to(device)
            test_tensors = torch.tensor(data_ccl["test"], dtype=torch.float32).to(device)

            train_loader = DataLoader(TensorDataset(train_tensors), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(val_tensors), batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(TensorDataset(test_tensors), batch_size=batch_size, shuffle=False)

            if data_type == 'mut':
                mae = load_pretrained_mae(f'./tcga_mut_mae_deepdep_best_split_{split_num}.pickle', train_tensors.shape[1], 1000, 100, 50)
            elif data_type == 'exp':
                mae = load_pretrained_mae(f'./tcga_exp_mae_deepdep_best_split_{split_num}.pickle', train_tensors.shape[1], 500, 200, 50)
            elif data_type == 'cna':
                mae = load_pretrained_mae(f'./tcga_cna_mae_deepdep_best_split_{split_num}.pickle', train_tensors.shape[1], 500, 200, 50)
            elif data_type == 'meth':
                mae = load_pretrained_mae(f'./tcga_meth_mae_deepdep_best_split_{split_num}.pickle', train_tensors.shape[1], 500, 200, 50)
            elif data_type == 'fprint':
                mae = MaskedAutoencoder(input_dim=train_tensors.shape[1], first_layer_dim=1000, second_layer_dim=100, latent_dim=50)

            trained_mae = train_mae(mae, train_loader, val_loader, num_epochs=epochs, learning_rate=learning_rate, device=device, data_name=data_type)
            evaluate_mae(trained_mae, test_loader, device, data_name=data_type)