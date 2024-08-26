import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

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
        :return: Latent variable (z), mu (mean), and logvar (log variance)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return z, mu, logvar

class VAE_DeepDEP(nn.Module):
    """
    Defines the VAE_DeepDep model
    
    :param dims_mut: Mut VAE input dimension
    :param fprint_dim: Fprint VAE input dimension
    :param dense_layer_dim: Dimension of the dense layers
    """
    def __init__(self, dims_mut, fprint_dim, dense_layer_dim):
        super(VAE_DeepDEP, self).__init__()
        self.vae_mut = VariationalAutoencoder(dims_mut, 1000, 100, 50)
        self.vae_fprint = VariationalAutoencoder(fprint_dim, 1000, 100, 50)

        self.fc_merged1 = nn.Linear(dense_layer_dim, dense_layer_dim)
        self.fc_merged2 = nn.Linear(dense_layer_dim, dense_layer_dim)
        self.fc_out = nn.Linear(dense_layer_dim, 1)

    def forward(self, mut, fprint):
        """
        Forward pass through the VAE_DeepDEP model.
        
        :param mut: Mutation data input
        :param fprint: Fingerprint data input
        :return: Prediction output
        """
        z, mu_mut, logvar_mut = self.vae_mut(mut)
        z, mu_fprint, logvar_fprint = self.vae_fprint(fprint)

        merged = torch.cat([mu_mut, mu_fprint], dim=1)
        merged = torch.relu(self.fc_merged1(merged))
        merged = torch.relu(self.fc_merged2(merged))
        output = self.fc_out(merged)
        return output
    
def load_data(filename):
    """
    Loads data from a file and processes it into a format suitable for the model.
    
    :param filename: Path to the data file
    :return: Data, labels, sample names, gene names
    """
    data = []
    gene_names = []
    data_labels = []
    lines = open(filename).readlines()
    sample_names = lines[0].replace('\n', '').split('\t')[1:]
    dx = 1

    for line in lines[dx:]:
        values = line.replace('\n', '').split('\t')
        gene = str.upper(values[0])
        gene_names.append(gene)
        data.append(values[1:])
    data = np.array(data, dtype='float32')
    data = np.transpose(data)

    return data, data_labels, sample_names, gene_names

if __name__ == '__main__':
    device = "mps"

    # List of genes to analyze
    genes = ["BRCA1", "BRCA2", "TP53", "PTEN"]

    for gene in genes:

        fprint_dim = 3115
        dims_mut = 4539
        dense_layer_dim = 100 # 50 x 2 VAEs = 100

        data_mut_tcga, data_labels_mut_tcga, sample_names_mut_tcga, gene_names_mut_tcga = load_data("./ccl_mut_data_paired_with_tcga.txt") 
        # data_mut_tcga, data_labels_mut_tcga, sample_names_mut_tcga, gene_names_mut_tcga = load_data("Data/TCGA/tcga_mut_data_paired_with_ccl.txt") # Uncomment this if you want to make predictions on TCGA Tumors
        data_fprint_1298DepOIs, data_labels_fprint, gene_names_fprint, function_names_fprint = load_data("./crispr_gene_fingerprint_cgp.txt")
        print("\n\nDatasets successfully loaded.\n\n")

        model = VAE_DeepDEP(dims_mut, fprint_dim, dense_layer_dim).to(device)
        model.load_state_dict(torch.load(f"./VAE_Prediction_Network_Split_2_Only_Mutation.pth", map_location=device))
        model.eval()

        gene_index = gene_names_mut_tcga.index(gene)
        data_pred = np.zeros((len(sample_names_mut_tcga), data_fprint_1298DepOIs.shape[0]))

        t = time.time()
        for z in np.arange(0, len(sample_names_mut_tcga)):
            data_mut_batch = np.zeros((data_fprint_1298DepOIs.shape[0], dims_mut), dtype='float32')
            data_mut_batch[:, :] = data_mut_tcga[z, :]
            data_mut_batch = torch.tensor(data_mut_batch, dtype=torch.float32).to(device)

            data_fprint_batch = torch.tensor(data_fprint_1298DepOIs, dtype=torch.float32).to(device)

            with torch.no_grad():
                data_mut_batch[:, gene_index] = 1.0
                output_mut = model(data_mut_batch, data_fprint_batch).cpu().numpy()

                data_mut_batch[:, gene_index] = 0.0
                output_wt = model(data_mut_batch, data_fprint_batch).cpu().numpy()
            
            sl_scores = output_mut - output_wt
            data_pred[z] = np.transpose(sl_scores)
            print("Sample %d predicted..." % z)

        data_pred_df = pd.DataFrame(data=np.transpose(data_pred), index=gene_names_fprint, columns=sample_names_mut_tcga)
        data_pred_df.to_csv(f"./{gene}_sl_predictions.csv", index_label='DepOI', float_format='%.4f')
        
