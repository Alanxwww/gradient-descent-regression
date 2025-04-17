from model.linear_regression import LinearRegressionGD
from utils.visualize import plot_regression_line
import numpy as np
import pandas as pd
import os

# Ensure output directory exists
os.makedirs("result", exist_ok=True)

# Load dataset without headers
data = pd.read_csv("data/data_chol_dias_pressure.csv", header=None)
data.columns = ["cholesterol", "diastolic_bp"]

# Prepare input and target
x = data["cholesterol"].values.reshape(-1, 1)
y = data["diastolic_bp"].values.reshape(-1, 1)

# Normalize inputs and outputs to avoid NaN
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

# Train model
model = LinearRegressionGD(learning_rate=0.001, n_iters=5000)
model.fit(x, y)

# Output results
print("Final weights:", model.weights)
print("Final bias:", model.bias)

# Predict and plot
y_pred = model.predict(x)
plot_regression_line(x, y, y_pred, save_path="result/convergence_plot.png")
