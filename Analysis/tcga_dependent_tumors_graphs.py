import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the TCGA predictions
data_path = 'Analysis/tcga_predicted_data.txt'
data = pd.read_csv(data_path, sep='\t')

# Specifying the cancer type and loading the associated TCGA IDs for that cancer type
cancer_type = "Brain_Lower_Grade_Glioma"
cancer_type_path = f'Analysis/CancerTCGAMappings/{cancer_type}.txt'

# Reading the TCGA IDs from the file associated with the selected cancer type
with open(cancer_type_path, 'r') as file:
    lung_adenocarcinoma_tcga_ids = file.read().splitlines()

# Filtering the data
filtered_data = data[['CRISPR_GENE'] + lung_adenocarcinoma_tcga_ids]

# Setting the 'CRISPR_GENE' column as the index of the DataFrame
filtered_data.set_index('CRISPR_GENE', inplace=True)

# Calculating the mean dependency score for each gene across the cancer samples
mean_distribution = filtered_data.mean(axis=1)

# Identifying the 20 genes with the lowest mean dependency scores
lowest_20_genes = mean_distribution.nsmallest(20).index

# Filtering the data to include only these 20 most dependent genes
lowest_20_genes_data = filtered_data.loc[lowest_20_genes]
lowest_20_genes_data = lowest_20_genes_data.reset_index().melt(id_vars='CRISPR_GENE', var_name='TCGA_ID', value_name='Dependency Score')

# Creating a boxplot
plt.figure(figsize=(20, 8))
sns.boxplot(x='CRISPR_GENE', y='Dependency Score', data=lowest_20_genes_data)
plt.title(f'The 20 Genes Brain Lower Grade Glioma Tumors Are Most Dependent On', fontsize=28)
plt.xlabel('Genes', fontsize=28)
plt.ylabel('Dependency Score', fontsize=28)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

