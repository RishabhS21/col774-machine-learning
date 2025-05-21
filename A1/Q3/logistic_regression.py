# Imports - you can add any other permitted libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

# Normalizing the data
def normalize(data):
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    # std[std == 0] = 1e-8
    return (data - mean) / std

class LogisticRegressor:
    # Assume Binary Classification
    def __init__(self):
        self.theta = None
        self.iterations = 0
        pass
    
    #define sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # Define hypothesis function(sigmoid here)
    def h(self, theta, X):
        return 1 / (1 + np.exp(-np.dot(X, theta)))

    # Define log-likelihood function
    def log_likelihood(self, theta, X, Y):
        m = len(Y)
        H = self.sigmoid(np.dot(X, theta))
        return (1 / m) * np.sum(Y * np.log(H) + (1 - Y) * np.log(1 - H))

    # Define hessian function
    def hessian(self, theta, X):
        m = len(X)
        H = self.sigmoid(np.dot(X, theta))
        return (1 / m) * np.dot(X.T, np.dot(np.diag(H * (1 - H)), X))

    def fit(self, X, y, learning_rate=0.01):
        """
        Fit the linear regression model to the data using Newton's Method.
        Remember to normalize the input data X before fitting the model.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        y : numpy array of shape (n_samples,)
            The target labels - 0 or 1.
        
        learning_rate : float
            The learning rate to use in the update rule.
        
        Returns
        -------
        List of Parameters: numpy array of shape (n_iter, n_features+1,)
            The list of parameters obtained after each iteration of Newton's Method.
        """

        X = normalize(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)  # Add term for interceptum
        theta = np.zeros(X.shape[1])
        eps = 1e-6
        m = len(y)
        iterations = 0
        converged = False
        prev_cost = self.log_likelihood(theta, X, y)
        theta_history = []
        while not converged:
            iterations += 1
            # prev_theta = theta
            theta_history.append(theta)
            H = self.sigmoid(np.dot(X, theta))
            # performing newton update
            gradient = np.dot(X.T, (H - y)) / m
            diag_mat = np.diag((H * (1 - H)).flatten())
            # print(temp)
            hessian = np.dot(X.T, np.dot(diag_mat, X)) / m
            theta -= np.dot(np.linalg.inv(hessian), gradient)
            curr_cost = self.log_likelihood(theta, X, y)
            if abs(curr_cost-prev_cost) < eps or iterations > 10000:
                converged = True
            prev_cost = curr_cost
        self.theta = theta
        self.iterations = iterations
        return np.array(theta_history)
        # pass
        
    
    def predict(self, X):
        """
        Predict the target values for the input data.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        Returns
        -------
        y_pred : numpy array of shape (n_samples,)
            The predicted target label.
        """
        X = normalize(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        y_pred = self.h(self.theta, X)
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        return y_pred
        # pass

    
    def plot(self, X, Y):
        # Separate the data based on labels
        X = normalize(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)  # Add term for interceptum
        X_label_0 = np.array([X[i] for i in range(len(Y)) if Y[i] == 0])
        X_label_1 = np.array([X[i] for i in range(len(Y)) if Y[i] == 1])
        
        # Define x1 range for plotting using the second column (x₁)
        x1_range = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
        # Compute x2 range using the decision boundary (θ₀ + θ₁*x₁ + θ₂*x₂ = 0)
        x2_range = -(self.theta[0] + self.theta[1] * x1_range) / self.theta[2]
        
        # Plot decision boundary and the data
        
        plt.plot(x1_range, x2_range, color='green', label="Decision Boundary")
        plt.scatter(X_label_0[:, 1], X_label_0[:, 2], color='red', label='Label 0')
        plt.scatter(X_label_1[:, 1], X_label_1[:, 2], color='blue', label='Label 1')
        plt.xlabel(r'$X_{1}$')
        plt.ylabel(r'$X_{2}$')
        plt.legend()
        plt.title('Logistic Regression')
        plt.show()


def main():
    # Read the CSV files from the previous directory; assuming no header
    X = pd.read_csv("../data/Q3/logisticX.csv", header=None)
    y = pd.read_csv("../data/Q3/logisticY.csv", header=None)
    
    # Convert to numpy arrays
    X = X.values
    y = y.values.flatten()
    
    model = LogisticRegressor()
    theta_history = model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Compute training accuracy
    accuracy = (y_pred == y).mean()
    
    # Display final model parameters and performance
    print("Final theta:", model.theta)
    print("Iterations:", model.iterations)
    print("Training Accuracy: {:.2%}".format(accuracy))
    model.plot(X, y)

if __name__ == "__main__":
    main()