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

# Calculating the average dependency score for each CCL
csv_ccl_average = csv_melted.groupby('CCL')['Dependency Score'].mean().reset_index()

# Identifying the top 10 CCLs with the lowest average dependency score
lowest_10_ccls_csv = csv_ccl_average.nsmallest(10, 'Dependency Score')['CCL']

# Filtering the original and predicted data
filtered_csv_data = csv_melted[csv_melted['CCL'].isin(lowest_10_ccls_csv)]
filtered_txt_data = txt_melted[txt_melted['CCL'].isin(lowest_10_ccls_csv)]

# Combining the filtered original and predicted data for comparison
combined_data = pd.concat([filtered_csv_data, filtered_txt_data])

# Plotting a violin plot
plt.figure(figsize=(20, 10))
sns.violinplot(x='CCL', y='Dependency Score', hue='Source', data=combined_data, split=True, palette=['blue', 'orange'])
plt.title('Comparison of Dependency Scores for 10 Highest Dependent CCLs from Original and Predicted Data', fontsize=26)
plt.xticks(rotation=45, fontsize=28)
plt.yticks(fontsize=14)
plt.xlabel('CCL', fontsize=28)
plt.ylabel('Dependency Score', fontsize=28)
plt.legend(title='Data Source', loc='upper right', fontsize=18, title_fontsize=18)

# Plotting a box plot
plt.figure(figsize=(20, 8))
sns.boxplot(x='CCL', y='Dependency Score', hue='Source', data=combined_data, palette=['blue', 'orange'])
plt.title('Comparison of Dependency Scores for 10 Highest Dependent CCLs from Original and Predicted Data', fontsize=26)
plt.xticks(rotation=45, fontsize=28)
plt.yticks(fontsize=14)
plt.xlabel('CCL', fontsize=28)
plt.ylabel('Dependency Score', fontsize=28)
plt.legend(title='Data Source', loc='upper right', fontsize=14, title_fontsize=18)
plt.tight_layout()
plt.show()
