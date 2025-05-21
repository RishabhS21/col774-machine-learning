# Imports - you can add any other permitted libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1e-8
    return (data - mean) / std

class GaussianDiscriminantAnalysis:
    # Assume Binary Classification
    def __init__(self):
        self.mu_0 = None
        self.mu_1 = None
        self.sigma = None         
        self.sigma_0 = None       
        self.sigma_1 = None 
        self.phi = None           
        self.assume_same_covariance = None
        # pass
    
    def fit(self, X, y, assume_same_covariance=False):
        """
        Fit the Gaussian Discriminant Analysis model to the data.
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
        Parameters: 
            If assume_same_covariance = True - 3-tuple of numpy arrays mu_0, mu_1, sigma 
            If assume_same_covariance = False - 4-tuple of numpy arrays mu_0, mu_1, sigma_0, sigma_1
            The parameters learned by the model.
        """
        X = normalize(X)
        m = X.shape[0]
        self.assume_same_covariance = assume_same_covariance
        y = np.array([0 if label.strip()=='Alaska' else 1 for label in y])
        phi = np.mean(y == 1)
        self.phi = phi

        # Separate the data into the two classes
        X_class_0 = X[y == 0]
        X_class_1 = X[y == 1]

        # Compute class means
        mu_0 = np.mean(X_class_0, axis=0)
        mu_1 = np.mean(X_class_1, axis=0)
        self.mu_0 = mu_0
        self.mu_1 = mu_1

        if assume_same_covariance:
            # Compute common covariance by subtracting the corresponding class mean
            X_modified = []
            for i in range(m):
                if y[i] == 0:
                    X_modified.append(X[i] - mu_0)
                else:
                    X_modified.append(X[i] - mu_1)
            X_modified = np.array(X_modified)
            sigma = np.dot(X_modified.T, X_modified) / m
            self.sigma = sigma
            return (mu_0, mu_1, sigma)
        else:
            # Compute separate covariance matrices for each class
            sigma_0 = np.dot((X_class_0 - mu_0).T, (X_class_0 - mu_0)) / X_class_0.shape[0]
            sigma_1 = np.dot((X_class_1 - mu_1).T, (X_class_1 - mu_1)) / X_class_1.shape[0]
            self.sigma_0 = sigma_0
            self.sigma_1 = sigma_1
            return (mu_0, mu_1, sigma_0, sigma_1)

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
        # Normalize the data as done during training.
        X = normalize(X)
        m = X.shape[0]
        y_pred = np.zeros(m)
        if self.assume_same_covariance:
            sigma_inv = np.linalg.pinv(self.sigma)
            for i in range(m):
                x = X[i]
                # Discriminant scores (log-posterior up to constants) for each class:
                score_alaska = -0.5 * np.dot(np.dot((x - self.mu_0).T, sigma_inv), (x - self.mu_0)) + np.log(1 - self.phi)
                score_canada = -0.5 * np.dot(np.dot((x - self.mu_1).T, sigma_inv), (x - self.mu_1)) + np.log(self.phi)
                y_pred[i] = 1 if score_canada > score_alaska else 0
        else:
            sigma_0_inv = np.linalg.pinv(self.sigma_0)
            sigma_1_inv = np.linalg.pinv(self.sigma_1)
            det0 = np.abs(np.linalg.det(self.sigma_0))
            det1 = np.abs(np.linalg.det(self.sigma_1))
            for i in range(m):
                x = X[i]
                score_alaska = (-0.5 * np.dot(np.dot((x - self.mu_0).T, sigma_0_inv), (x - self.mu_0))
                                - 0.5 * np.log(det0 + 1e-8)
                                + np.log(1 - self.phi))
                score_canada = (-0.5 * np.dot(np.dot((x - self.mu_1).T, sigma_1_inv), (x - self.mu_1))
                                - 0.5 * np.log(det1 + 1e-8)
                                + np.log(self.phi))
                y_pred[i] = 1 if score_canada > score_alaska else 0
        # Map numeric predictions back to label names
        y_pred_labels = np.array(['Canada' if pred==1 else 'Alaska' for pred in y_pred])
        return y_pred_labels
        # pass
    
    def plot_linear(self):
        sigma_inv = np.linalg.pinv(self.sigma)
        coef = np.dot((self.mu_1 - self.mu_0).T, sigma_inv)
        # Note: constant term is moved to the right side.
        const = 0.5 * (np.dot(np.dot(self.mu_1.T, sigma_inv), self.mu_1) - np.dot(np.dot(self.mu_0.T, sigma_inv), self.mu_0)) - np.log(self.phi/(1-self.phi))
        # For 2-D features, let x = [x1, x2]. We plot x2 in terms of x1.
        x1_vals = np.linspace(-3, 3, 100)
        x2_vals = (const - coef[0] * x1_vals) / coef[1]
        return x1_vals, x2_vals
    
    def plot_quadratic(self):
        sigma_0_inv = np.linalg.pinv(self.sigma_0)
        sigma_1_inv = np.linalg.pinv(self.sigma_1)
        d_0 = np.sqrt(np.abs(np.linalg.det(self.sigma_0)))
        d_1 = np.sqrt(np.abs(np.linalg.det(self.sigma_1)))
        coeff_term = 0.5 * (sigma_1_inv - sigma_0_inv)
        linear_term = np.dot(self.mu_1.T, sigma_1_inv) - np.dot(self.mu_0.T, sigma_0_inv)
        const_term = (0.5 * (np.dot(np.dot(self.mu_1.T, sigma_1_inv), self.mu_1) - np.dot(np.dot(self.mu_0.T, sigma_0_inv), self.mu_0))
                      + np.log((1-self.phi)*d_0/(self.phi*d_1)))
        # Create a meshgrid for plotting
        x1_vals = np.linspace(-3, 3, 200)
        x2_vals = np.linspace(-3, 3, 200)
        x1, x2 = np.meshgrid(x1_vals, x2_vals)
        term1 = coeff_term[0, 0]*x1**2 + (coeff_term[0, 1] + coeff_term[1, 0])*x1*x2 + coeff_term[1, 1]*x2**2
        term2 = linear_term[0]*x1 + linear_term[1]*x2
        h = term1 - term2 + const_term
        contour_plot = plt.contour(x1, x2, h, levels=[0], colors='orange')
        return contour_plot
    


def main():
    # Read the data using pandas (assumed whitespace separated and no header)
    X = pd.read_csv('../data/Q4/q4x.dat', sep='\s+', header=None).values
    Y = pd.read_csv('../data/Q4/q4y.dat', sep='\s+', header=None).values.flatten()
    
    # Create an instance of the GDA model. You can change the flag to True for linear boundary.
    model = GaussianDiscriminantAnalysis()
    
    # Fit with common covariance (linear decision boundary)
    params = model.fit(X, Y, assume_same_covariance=True)
    print("Parameters with shared covariance (Linear):")
    print("mu_0:", model.mu_0)
    print("mu_1:", model.mu_1)
    print("sigma:", model.sigma)
    print("phi:", model.phi)
    
    # Predict on training data and compute accuracy
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == Y)
    print("Training Accuracy (Linear): {:.2%}".format(accuracy))
    
    # Plot the normalized data and the linear decision boundary
    X_norm = normalize(X)
    X_class_0 = X_norm[[i for i, label in enumerate(Y) if label.strip()=='Alaska']]
    X_class_1 = X_norm[[i for i, label in enumerate(Y) if label.strip()=='Canada']]
    plt.figure(figsize=(8,6))
    plt.scatter(X_class_0[:,0], X_class_0[:,1], color='red', label='Alaska')
    plt.scatter(X_class_1[:,0], X_class_1[:,1], color='blue', label='Canada')
    x1_vals, x2_vals = model.plot_linear()
    plt.plot(x1_vals, x2_vals, color='green')
    
    # Now fit with separate covariance matrices (quadratic decision boundary)
    model.fit(X, Y, assume_same_covariance=False)
    print("\nParameters with separate covariances (Quadratic):")
    print("mu_0:", model.mu_0)
    print("mu_1:", model.mu_1)
    print("sigma_0:", model.sigma_0)
    print("sigma_1:", model.sigma_1)
    y_pred_quad = model.predict(X)
    accuracy_quad = np.mean(y_pred_quad == Y)
    print("Training Accuracy (Quadratic): {:.2%}".format(accuracy_quad))
    
    # Plot quadratic decision boundary on the same figure
    model.plot_quadratic()
    plt.plot
    plt.xlabel(r'$X_1$')
    plt.ylabel(r'$X_2$')
    plt.legend()
    plt.title('Gaussian Discriminant Analysis')
    plt.show()

if __name__ == "__main__":
    main()