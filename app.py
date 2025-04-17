import streamlit as st
import pandas as pd
import numpy as np
from model.linear_regression import LinearRegressionGD
from utils.visualize import plot_regression_line
import matplotlib.pyplot as plt

st.title("Gradient Descent Linear Regression")

uploaded_file = st.file_uploader("Upload your CSV data", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, header=None)
    data.columns = ["cholesterol", "diastolic_bp"]

    x = data["cholesterol"].values.reshape(-1, 1)
    y = data["diastolic_bp"].values.reshape(-1, 1)

    # Normalize
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)

    model = LinearRegressionGD(learning_rate=0.0001, n_iters=1000)
    model.fit(x, y)
    y_pred = model.predict(x)

    st.write("### Final Weights:", model.weights.flatten())
    st.write("### Final Bias:", model.bias)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(x, y, color="blue", label="Actual")
    ax.plot(x[np.argsort(x[:, 0])], y_pred[np.argsort(x[:, 0])], color="red", label="Prediction")
    ax.set_xlabel("Cholesterol")
    ax.set_ylabel("Diastolic BP")
    ax.legend()
    st.pyplot(fig)
