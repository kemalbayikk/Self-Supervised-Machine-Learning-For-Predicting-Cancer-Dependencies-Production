import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# File paths for the original and predicted data
csv_data_path = './data_dep_updated_2.csv'
txt_data_path = './ccl_predicted_data_best_model_vae_split_2.txt'

# Loading the original and predicted data
csv_data = pd.read_csv(csv_data_path)
txt_data = pd.read_csv(txt_data_path, delimiter='\t')

# Converting the original and predicted data into a long format for easier plotting
csv_melted = csv_data.melt(id_vars=['CRISPR_GENE'], var_name='CCL', value_name='Dependency Score').assign(Source='Original Data')
txt_melted = txt_data.melt(id_vars=['CRISPR_GENE'], var_name='CCL', value_name='Dependency Score').assign(Source='Predicted Data')

# Calculating the mean dependency score for each gene from the original data
csv_average = csv_melted.groupby('CRISPR_GENE')['Dependency Score'].mean().reset_index()

# Filtering the original and predicted data to include only the top 10 most effective genes
top_genes_csv = csv_average.nsmallest(10, 'Dependency Score')['CRISPR_GENE']
filtered_csv_data = csv_melted[csv_melted['CRISPR_GENE'].isin(top_genes_csv)]
filtered_txt_data = txt_melted[txt_melted['CRISPR_GENE'].isin(top_genes_csv)]

# Combining the filtered original and predicted data for comparison
combined_data = pd.concat([filtered_csv_data, filtered_txt_data])

# Determining the order of genes based on the mean dependency scores
gene_order = combined_data.groupby('CRISPR_GENE')['Dependency Score'].mean().sort_values().index

# Plotting a violin plot
plt.figure(figsize=(20, 10))
sns.violinplot(x='CRISPR_GENE', y='Dependency Score', hue='Source', data=combined_data, split=True, palette=['blue', 'orange'], order=gene_order)
plt.title('Comparison of Dependency Scores for Top 10 Most Effective Genes Across All CCLs', fontsize=28)
plt.xticks(rotation=45, fontsize=28)
plt.yticks(fontsize=14) 
plt.xlabel('Gene', fontsize=28)
plt.ylabel('Dependency Score', fontsize=28)
plt.legend(title='Data Source', loc='lower right', fontsize=18, title_fontsize=18)
plt.tight_layout()
plt.show()

# Plotting a box plot
plt.figure(figsize=(20, 10))
sns.boxplot(x='CRISPR_GENE', y='Dependency Score', hue='Source', data=combined_data, split=True, palette=['blue', 'orange'], order=gene_order)
plt.title('Comparison of Dependency Scores for Top 10 Most Effective Genes Across All CCLs', fontsize=28)
plt.xticks(rotation=45, fontsize=28)
plt.yticks(fontsize=14) 
plt.xlabel('Gene', fontsize=28)
plt.ylabel('Dependency Score', fontsize=28)
plt.legend(title='Data Source', loc='lower right', fontsize=18, title_fontsize=18)
plt.tight_layout()
plt.show()