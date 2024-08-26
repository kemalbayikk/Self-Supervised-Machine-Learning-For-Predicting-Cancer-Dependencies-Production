from scipy.stats import ttest_rel
import numpy as np

# This function reads the contents of a file and returns a list of float values
def read_results_from_file(file_path):
    with open(file_path, 'r') as file:
        results = file.readlines()

    return [float(line.strip()) for line in results]

# This function writes a list of correlation values to a file
def write_correlations_to_file(correlations, file_path):
    with open(file_path, 'w') as file:
        for corr in correlations:
            file.write(f"{corr}\n")

# This function calculates Cohen's d effect size between two sets of data
def cohen_d(x, y):
    diff = np.array(x) - np.array(y)
    return np.mean(diff) / np.std(diff, ddof=1)

# Defining file paths for the prediction results and target values
VAE_DeepDep_file = './y_pred_test_Prediction_Network_VAE_Split_2_LR_0.001.txt'
DeepDep_file = './y_pred_test_CCL_Split_2_Original.txt'
target_file = './y_true_test_CCL_VAE_Split_2.txt'

# Reading the model prediction results and true target values from the files
VAE_DeepDep_results = read_results_from_file(VAE_DeepDep_file)
DeepDep_results = read_results_from_file(DeepDep_file)
target_values = read_results_from_file(target_file)

# Calculating absolute differences between the model predictions and the true target values
VAE_DeepDep_differences = [abs(VAE_DeepDep_results[i] - target_values[i]) for i in range(len(VAE_DeepDep_results))]
DeepDep_differences = [abs(DeepDep_results[i] - target_values[i]) for i in range(len(DeepDep_results))]

# Calculating mean and standard deviation for the differences
VAE_DeepDep_mean = np.mean(VAE_DeepDep_differences)
VAE_DeepDep_std = np.std(VAE_DeepDep_differences, ddof=1)
DeepDep_mean = np.mean(DeepDep_differences)
DeepDep_std = np.std(DeepDep_differences, ddof=1)

# Performing paired t-tests to compare the differences between the two models
t_statistic_greater, p_value_greater = ttest_rel(VAE_DeepDep_differences, DeepDep_differences, alternative='greater')
t_statistic_less, p_value_less = ttest_rel(VAE_DeepDep_differences, DeepDep_differences, alternative='less')
t_statistic, p_value = ttest_rel(VAE_DeepDep_differences, DeepDep_differences, alternative='two-sided')

# Calculating Cohen's d
cohen_d_value = cohen_d(VAE_DeepDep_differences, DeepDep_differences)

# Printing out the calculated statistics and test results
print(f"VAE-DeepDep Mean Error: {VAE_DeepDep_mean:.4f}, Standard Deviation: {VAE_DeepDep_std:.4f}")
print(f"DeepDep Mean Error: {DeepDep_mean:.4f}, Standard Deviation: {DeepDep_std:.4f}")
print(f"Cohen's d: {cohen_d_value:.4f}")

print(f"T-statistics Greater: {t_statistic_greater}")
print(f"P-value Greater: {p_value_greater}")

print(f"T-statistics Less: {t_statistic_less}")
print(f"P-value Less: {p_value_less}")

print(f"T-statistics Equal: {t_statistic}")
print(f"P-value Equal: {p_value}")