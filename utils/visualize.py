import matplotlib.pyplot as plt
import numpy as np

def plot_regression_line(x, y, y_pred, save_path=None):
    plt.scatter(x, y, color='blue', label='Actual')

    # Sort x for smooth red line
    sorted_idx = np.argsort(x.flatten())
    x_sorted = x[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]

    plt.plot(x_sorted, y_pred_sorted, color='red', label='Prediction')
    plt.xlabel("Cholesterol")
    plt.ylabel("Diastolic BP")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()