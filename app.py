import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model.linear_regression import LinearRegressionGD

st.title("📉 Gradient Descent Linear Regression")

st.markdown("Upload your CSV file with two columns: cholesterol and diastolic_bp")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file, header=None)
    data.columns = ["cholesterol", "diastolic_bp"]

    st.write("### Preview of Uploaded Data")
    st.dataframe(data.head())

    # Prepare input
    x = data["cholesterol"].values.reshape(-1, 1)
    y = data["diastolic_bp"].values.reshape(-1, 1)

    # Normalize
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)

    # Sidebar controls
    st.sidebar.header("Model Hyperparameters")
    lr = st.sidebar.slider("Learning rate", 0.0001, 0.01, 0.001, step=0.0001)
    n_iters = st.sidebar.slider("Iterations", 100, 2000, 1000, step=100)

    # Train model
    model = LinearRegressionGD(learning_rate=lr, n_iters=n_iters)
    model.fit(x, y)
    y_pred = model.predict(x)

    st.write(f"**Final weight:** {model.weights.flatten()[0]:.4f}")
    st.write(f"**Final bias:** {model.bias:.4f}")

    # Plot
    sorted_idx = np.argsort(x.flatten())
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]

    fig, ax = plt.subplots()
    ax.scatter(x_sorted, y_sorted, color="blue", label="Actual")
    ax.plot(x_sorted, y_pred_sorted, color="red", label="Prediction")
    ax.set_xlabel("Cholesterol (normalized)")
    ax.set_ylabel("Diastolic BP (normalized)")
    ax.legend()
    st.pyplot(fig)
