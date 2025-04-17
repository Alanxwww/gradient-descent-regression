![Convergence Plot](result/convergence_plot.png)

from utils.visualize import plot_regression_line
plot_regression_line(x, y, model.predict(x), save_path="result/convergence_plot.png")
