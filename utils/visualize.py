import matplotlib.pyplot as plt

def plot_regression_line(x, y, y_pred, save_path=None):
    plt.scatter(x, y, color='blue', label='Actual')
    plt.plot(x, y_pred, color='red', label='Prediction')
    plt.xlabel("Cholesterol")
    plt.ylabel("Diastolic BP")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()