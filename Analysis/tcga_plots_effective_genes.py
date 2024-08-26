import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the TCGA predictions
data_path = 'Analysis/tcga_predicted_data.txt'
txt_data = pd.read_csv(data_path, delimiter='\t') 

# Transforming the DataFrame into long format
txt_melted = txt_data.melt(id_vars=['CRISPR_GENE'], var_name='TCGA', value_name='Dependency Score').assign(Source='Predicted Data')

# Calculating the average dependency score for each gene across all TCGA samples
txt_average_impact = txt_melted.groupby('CRISPR_GENE')['Dependency Score'].mean().reset_index()

# Identifying the top 20 genes with the lowest average dependency scores
top_genes_txt = txt_average_impact.nsmallest(20, 'Dependency Score')['CRISPR_GENE']

# Filtering the melted data
filtered_txt_data = txt_melted[txt_melted['CRISPR_GENE'].isin(top_genes_txt)]

# Ordering the genes by their average dependency scores
gene_order = filtered_txt_data.groupby('CRISPR_GENE')['Dependency Score'].mean().sort_values().index

# Creating a boxplot
plt.figure(figsize=(20, 8))
sns.boxplot(x='CRISPR_GENE', y='Dependency Score', data=filtered_txt_data, order=gene_order, palette='Set2')
plt.title('Most 20 Effective Genes from Predicted TCGA Dependency Scores', fontsize=26)
plt.xticks(rotation=45, fontsize=28)
plt.yticks(fontsize=14) 
plt.xlabel('Gene', fontsize=28)
plt.ylabel('Dependency Score', fontsize=28)
plt.tight_layout()
plt.show()