

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Defining the gene for the analysis
gene = "TP53"

# Loading the data
all_se_scores_df = pd.read_csv(F'Analysis/{gene}_sl_predictions.csv')

# Transforming the DataFrame into long format
df_long = pd.melt(all_se_scores_df, id_vars=['DepOI'], var_name='Sample', value_name='SE_Score')
df = pd.DataFrame(all_se_scores_df)

# Calculating the mean SL score across all samples for each gene
df['Mean'] = df.iloc[:, 1:].mean(axis=1)

# Filtering out the rows where the gene in the 'DepOI' column matches the gene
df_filtered = df[df['DepOI'] != gene]

# Selecting only the 'DepOI' and 'Mean' columns for sorting
df_mean = df_filtered[['DepOI', 'Mean']]

# Sorting the genes by their mean SL score in ascending order
df_sorted = df_mean.sort_values(by='Mean', ascending=True)

# Getting the top 10 genes with the lowest mean SL scores
lowest_genes = df_sorted['DepOI'].head(10).unique()

# Preparing the order of genes
gene_order = df_sorted[['DepOI']].head(10)

# Filtering the long-format DataFrame
df_long_filtered = df_long[df_long['DepOI'].isin(gene_order['DepOI'])]


# Creating a boxplot
plt.figure(figsize=(18, 8))
sns.boxplot(x='DepOI', y='SE_Score', data=df_long_filtered, order=gene_order['DepOI'], palette='tab10')
plt.ylabel('SL Score', fontsize=26)
plt.xlabel('Gene', fontsize=28)
plt.xticks(rotation=45, fontsize=24)
plt.yticks(fontsize=18) 
plt.title(f'Distribution of SL Scores for Top 10 Most Synthetic Lethal Genes with {gene} Mutation', fontsize=26)
plt.tight_layout()
plt.show()