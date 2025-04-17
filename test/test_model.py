import numpy as np
from model.linear_regression import LinearRegressionGD

def test_gradient_descent():
    x = np.array([[1], [2], [3]])
    y = np.array([[2], [4], [6]])
    model = LinearRegressionGD(learning_rate=0.1, n_iters=1000)
    model.fit(x, y)
    y_pred = model.predict(x)
    assert np.allclose(y_pred, y, atol=1e-1), "Prediction not close to actual values"

if __name__ == "__main__":
    test_gradient_descent()
    print("Test passed!")
