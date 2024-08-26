import torch
import torch.nn as nn
import numpy as np
import pandas as pd

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
    :param dims_exp: Exp VAE input dimension
    :param dims_cna: Cna VAE input dimension
    :param dims_meth: Meth VAE input dimension
    :param fprint_dim: Fprint VAE input dimension
    :param dense_layer_dim: Dimension of the dense layers
    """
    def __init__(self, dims_mut, dims_exp, dims_cna, dims_meth, fprint_dim, dense_layer_dim):
        super(VAE_DeepDEP, self).__init__()
        self.vae_mut = VariationalAutoencoder(dims_mut, 1000, 100, 50)
        self.vae_exp = VariationalAutoencoder(dims_exp, 500, 200, 50)
        self.vae_cna = VariationalAutoencoder(dims_cna, 500, 200, 50)
        self.vae_meth = VariationalAutoencoder(dims_meth, 500, 200, 50)

        self.vae_fprint = VariationalAutoencoder(fprint_dim, 1000, 100, 50)

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
        z, mu_mut, logvar_mut = self.vae_mut(mut)
        z, mu_exp, logvar_exp = self.vae_exp(exp)
        z, mu_cna, logvar_cna = self.vae_cna(cna)
        z, mu_meth, logvar_meth = self.vae_meth(meth)
        z, mu_fprint, logvar_fprint = self.vae_fprint(fprint)
        
        merged = torch.cat([mu_mut, mu_exp, mu_cna, mu_meth, mu_fprint], dim=1)
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
    
    dims_mut = 4539
    dims_exp = 6016
    dims_cna = 7460
    dims_meth = 6617
    fprint_dim = 3115
    dense_layer_dim = 250

    model = VAE_DeepDEP(dims_mut, dims_exp, dims_cna, dims_meth, fprint_dim, dense_layer_dim).to(device)

    model.load_state_dict(torch.load("./VAE_Prediction_Network_Split_2_LR_0.001.pth", map_location=device))
    model.eval()

    # Load TCGA or CCL genomics data and gene fingerprints
    data_mut, data_labels_mut, sample_names_mut, gene_names_mut = load_data("Data/CCL/ccl_mut_data_paired_with_tcga.txt")
    data_exp, data_labels_exp, sample_names_exp, gene_names_exp = load_data("Data/CCL/ccl_exp_data_paired_with_tcga.txt")
    data_cna, data_labels_cna, sample_names_cna, gene_names_cna = load_data("Data/CCL/ccl_cna_data_paired_with_tcga.txt")
    data_meth, data_labels_meth, sample_names_meth, gene_names_meth = load_data("Data/CCL/ccl_meth_data_paired_with_tcga.txt")
    data_fprint_1298DepOIs, data_labels_fprint, gene_names_fprint, function_names_fprint = load_data("Data/crispr_gene_fingerprint_cgp.txt")
    print("\n\nDatasets successfully loaded.\n\n")

    batch_size = 10000
    sample_size = 278 # Change this to 8238 if you are predicting TCGA data
    data_pred = np.zeros((sample_size, data_fprint_1298DepOIs.shape[0]))
    
    for z in np.arange(0, sample_size):
        data_mut_batch = torch.tensor(data_mut[np.repeat(z, data_fprint_1298DepOIs.shape[0])], dtype=torch.float32).to(device)
        data_exp_batch = torch.tensor(data_exp[np.repeat(z, data_fprint_1298DepOIs.shape[0])], dtype=torch.float32).to(device)
        data_cna_batch = torch.tensor(data_cna[np.repeat(z, data_fprint_1298DepOIs.shape[0])], dtype=torch.float32).to(device)
        data_meth_batch = torch.tensor(data_meth[np.repeat(z, data_fprint_1298DepOIs.shape[0])], dtype=torch.float32).to(device)
        data_fprint_batch = torch.tensor(data_fprint_1298DepOIs, dtype=torch.float32).to(device)

        with torch.no_grad():
            data_pred_tmp = model(data_mut_batch, data_exp_batch, data_cna_batch, data_meth_batch, data_fprint_batch).cpu().numpy()
        
        data_pred[z] = np.transpose(data_pred_tmp)
        print("Sample %d predicted..." % z)

    # Write prediction results to txt
    data_pred_df = pd.DataFrame(data=np.transpose(data_pred), index=gene_names_fprint, columns=sample_names_mut[0:sample_size])
    data_pred_df.to_csv("./predicted_samples.txt", sep='\t', index_label='CRISPR_GENE', float_format='%.4f')
