import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import seaborn as sns
import numpy as np
import pandas as pd

# Function to plot the density of dependency scores for training (true and predicted) and TCGA predictions
def plot_density(y_true_train, y_pred_train, tcga_predictions):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(y_true_train, label='CCL original', color='blue')
    sns.kdeplot(y_pred_train, label='CCL predicted', color='orange')
    sns.kdeplot(tcga_predictions, label='Tumor predicted', color='brown')
    plt.xlabel('Dependency score')
    plt.ylabel('Density (x0.01)')
    plt.title(f'Density plot of Dependency Scores')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Function to plot scatter plots for true vs predicted dependency scores, along with regression lines and performance metrics
def plot_results(y_true_train, y_pred_train, y_true_test, y_pred_test):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_pred_train, y_true_train, alpha=0.5)
    coef_train = np.polyfit(y_pred_train, y_true_train, 1)
    poly1d_fn_train = np.poly1d(coef_train)
    plt.plot(np.unique(y_pred_train), poly1d_fn_train(np.unique(y_pred_train)), color='red')
    plt.xlabel('DeepDEP-predicted score')
    plt.ylabel('Original dependency score')
    plt.title(f'Training/validation')
    plt.xlim(-4, 5)
    plt.ylim(-4, 5)
    pearson_corr_train, _ = pearsonr(y_pred_train, y_true_train)
    mse_train = mean_squared_error(y_true_train, y_pred_train)
    plt.text(0.1, 0.9, f'$\\rho$ = {pearson_corr_train:.2f}\nMSE = {mse_train:.3f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f'y = {coef_train[0]:.2f}x + {coef_train[1]:.2f}', color='red', transform=plt.gca().transAxes)

    plt.subplot(1, 2, 2)
    plt.scatter(y_pred_test, y_true_test, alpha=0.5)
    coef_test = np.polyfit(y_pred_test, y_true_test, 1)
    poly1d_fn_test = np.poly1d(coef_test)
    plt.plot(np.unique(y_pred_test), poly1d_fn_test(np.unique(y_pred_test)), color='red')
    plt.xlabel('VAE DeepDEP Predicted Score')
    plt.ylabel('Original Dependency Score')
    plt.title(f'Testing')
    plt.xlim(-4, 5)
    plt.ylim(-4, 5)
    pearson_corr_test, _ = pearsonr(y_pred_test, y_true_test)
    mse_test = mean_squared_error(y_true_test, y_pred_test)
    plt.text(0.1, 0.9, f'$\\rho$ = {pearson_corr_test:.2f}\nMSE = {mse_test:.3f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f'y = {coef_test[0]:.2f}x + {coef_test[1]:.2f}', color='red', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()

y_true_train = np.loadtxt('./y_true_train_Prediction_Network_VAE_Split_2_LR_0.001.txt', dtype=float)
y_pred_train = np.loadtxt('./y_pred_train_Prediction_Network_VAE_Split_2_LR_0.001.txt', dtype=float)
y_true_test = np.loadtxt('./y_true_test_Prediction_Network_VAE_Split_2_LR_0.001.txt', dtype=float)
y_pred_test = np.loadtxt('./y_pred_test_Prediction_Network_VAE_Split_2_LR_0.001.txt', dtype=float)

tcga_predictions_df = pd.read_csv('Analysis/tcga_predicted_data.txt', delimiter='\t', index_col="CRISPR_GENE")
tcga_predictions = tcga_predictions_df.values.T

plot_results(y_true_train, y_pred_train, y_true_test, y_pred_test)
plot_density(y_true_train, y_pred_train, tcga_predictions)