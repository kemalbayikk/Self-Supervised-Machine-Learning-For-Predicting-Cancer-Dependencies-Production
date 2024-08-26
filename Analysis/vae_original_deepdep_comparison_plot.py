import numpy as np
import matplotlib.pyplot as plt

# Loading true and predicted dependency scores for VAE-DeepDep model (training and testing data)
y_true_train = np.loadtxt('Anaylsis/y_true_train_Prediction_Network_VAE_Split_2_LR_0.001.txt', dtype=float)
y_pred_train = np.loadtxt('Anaylsis/y_pred_train_Prediction_Network_VAE_Split_2_LR_0.001.txt', dtype=float)
y_true_test = np.loadtxt('Anaylsis/y_true_test_Prediction_Network_VAE_Split_2_LR_0.001.txt', dtype=float)
y_pred_test = np.loadtxt('Anaylsis/y_pred_test_Prediction_Network_VAE_Split_2_LR_0.001.txt', dtype=float)

# Loading true and predicted dependency scores for the original DeepDep model (training and testing data)
y_true_train_original = np.loadtxt('Anaylsis/y_true_train_CCL_Split_2_Original.txt', dtype=float)
y_pred_train_original = np.loadtxt('Anaylsis/y_pred_train_CCL_Split_2_Original.txt', dtype=float)
y_true_test_original = np.loadtxt('Anaylsis/y_true_test_CCL_Split_2_Original.txt', dtype=float)
y_pred_test_original = np.loadtxt('Anaylsis/y_pred_test_CCL_Split_2_Original.txt', dtype=float)

plt.figure(figsize=(10, 8))

# Scatter plot for the original DeepDep model (testing data)
plt.scatter(y_pred_test_original, y_true_test_original, alpha=0.5, label='Original Model', color='blue')
for i in range(len(y_true_test)):
    plt.plot([y_pred_test_original[i], y_true_test_original[i]], [y_true_test_original[i], y_true_test_original[i]], 'b-', alpha=0.5)

# Scatter plot for the VAE-DeepDep model (testing data)
plt.scatter(y_pred_test, y_true_test, alpha=0.5, label='VAE DeepDep', color='green')
for i in range(len(y_true_test)):
    plt.plot([y_pred_test[i], y_true_test[i]], [y_true_test[i], y_true_test[i]], 'g-', alpha=0.5)

plt.plot(np.linspace(-4, 5, 100), np.linspace(-4, 5, 100), 'r--', label='y = x')
plt.title('VAE-DeepDep and Original-DeepDep Model Comparisons')
plt.xlabel('Predicted dependency score')
plt.ylabel('Original dependency score')
plt.legend()
plt.grid(True)
plt.show()
