# Imports - you can add any other permitted libraries
import numpy as np
import matplotlib.pyplot as plt
# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.


def generate(N, theta, input_mean, input_sigma, noise_sigma):
    """
    Generate normally distributed input data and target values
    Note that we have 2 input features
    Parameters
    ----------
    N : int
        The number of samples to generate.
        
    theta : numpy array of shape (3,)
        The true parameters of the linear regression model.
        
    input_mean : numpy array of shape (2,)
        The mean of the input data.
        
    input_sigma : numpy array of shape (2,)
        The standard deviation of the input data.
        
    noise_sigma : float
        The standard deviation of the Gaussian noise.
        
    Returns
    -------
    X : numpy array of shape (N, 2)
        The input data.
        
    y : numpy array of shape (N,)
        The target values.
    """
    X = np.random.normal(loc=input_mean, scale=input_sigma, size=(N, 2))

    noise = np.random.normal(0, noise_sigma, size=N)
    
    y = theta[0] + theta[1] * X[:, 0] + theta[2] * X[:, 1] + noise
    
    return X, y
    # pass

def shuffle_data(X, Y):
    indices = np.random.permutation(len(X))
    return X[indices], Y[indices]

class StochasticLinearRegressor:
    def __init__(self):
        self.theta = None
        self.iterations = 0
        self.theta0_history = []
        self.theta1_history = []
        self.theta2_history = []
        self.loss = 0
        # pass
    
    def h(self, theta, X):
        return X.dot(theta)
    
    def J(self, theta, X, y):
        m = len(y)
        diff = self.h(theta, X) - y
        J = (1 / (2 * m)) * np.sum(diff ** 2)
        return J
    
    def fit(self, X, y, learning_rate=0.01):
        """
        Fit the linear regression model to the data using Gradient Descent.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        y : numpy array of shape (n_samples,)
            The target values.

        learning_rate : float
            The learning rate to use in the update rule.
            
        Returns
        -------
        List of Parameters: numpy array of shape (n_iter, n_features+1,)
            The list of parameters obtained after each iteration of Gradient Descent.
        """
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        X, y = shuffle_data(X, y)
        m = len(y)
        theta = np.zeros((X.shape[1], 1))
        r = 80
        k = 1000
        eps = 1e-9
        batches = m//r
        iterations = 0
        converged = False
        theta0_history = []
        theta1_history = []
        theta2_history = []
        last_k_loss = []
        last_k_loss_sum = 0
        next_k_loss = []
        next_k_loss_sum = 0
        while not converged:
            X, y = shuffle_data(X, y)

            for i in range(batches):
                iterations += 1
                X_b = X[i*r:(i+1)*r]
                y_b = y[i*r:(i+1)*r].reshape(-1, 1)
                H = self.h(theta, X_b)
                gradient = np.dot(X_b.T,(H-y_b))/(2*r)
                theta -= learning_rate * gradient
                curr_loss = self.J(theta, X_b, y_b)
                theta0_history.append(theta[0][0])
                theta1_history.append(theta[1][0])
                theta2_history.append(theta[2][0])
                if iterations <= k:
                    last_k_loss.append(curr_loss)
                    last_k_loss_sum += curr_loss
                    continue
                elif (iterations > k) and (iterations <= 2*k):
                    next_k_loss.append(curr_loss)
                    next_k_loss_sum+=curr_loss
                    continue
                if abs(last_k_loss_sum/k - next_k_loss_sum/k) < eps * abs(last_k_loss_sum/k) or iterations > 1000000:
                    converged = True
                    break
                last_k_loss_sum = last_k_loss_sum - last_k_loss.pop(0) + next_k_loss[0]
                last_k_loss.append(next_k_loss[0])
                next_k_loss_sum = next_k_loss_sum - next_k_loss.pop(0) + curr_loss
                next_k_loss.append(curr_loss)

        self.theta = theta
        self.iterations = iterations
        self.theta0_history = theta0_history
        self.theta1_history = theta1_history
        self.theta2_history = theta2_history
        self.loss = curr_loss
        return np.array([theta0_history, theta1_history, theta2_history]).T

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
            The predicted target values.
        """
        if self.theta is None:
            raise Exception('Fit the model before prediction')
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        y_pred = self.h(self.theta, X)
        return y_pred.flatten()
        # pass

    def plot_theta(self):
        # Create a figure and 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the values in 3D space
        ax.plot(self.theta0_history, self.theta1_history, self.theta2_history, color='red')

        # Set labels for each axis
        ax.set_xlabel('$\\theta_{0}$')
        ax.set_ylabel('$\\theta_{1}$')
        ax.set_zlabel('$\\theta_{2}$')
        ax.set_title('Movement of Parameters')
        plt.show()


def closed_form_solution(X, y):

    X_aug = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    
    # Compute the closed form solution
    theta = np.linalg.inv(X_aug.T.dot(X_aug)).dot(X_aug.T).dot(y)
    return theta


def main():
    # Parameters for generating data
    N = 1000000
    theta_true = np.array([3, 1, 2])
    input_mean = np.array([3, -1])
    input_sigma = np.array([2, 2])
    noise_sigma = np.sqrt(2)

    # Generate the data
    X, y = generate(N, theta_true, input_mean, input_sigma, noise_sigma)
    
    # Split into 80% training and 20% testing
    split_index = int(0.8 * N)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    theta_closed = closed_form_solution(X_train, y_train)
    print("Closed form theta:", theta_closed)
    
    model = StochasticLinearRegressor()
    theta_history = model.fit(X_train, y_train, learning_rate=0.01)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Compute the test loss using the cost function
    # Note: We need to augment X_test with a column of ones to account for the intercept.
    X_test_aug = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
    
    # Print out the results
    print(y_test[0], y_pred[0])
    model.plot_theta()

    print("Final theta from training:", model.theta.flatten())
    print(model.iterations)
    print("Final loss from training:", model.loss)
    # print the loss on the test set
    print("Loss on test set:", model.J(model.theta.flatten(), X_test_aug, y_test))
    # print the original loss with respect to known original theta
    print("Original loss:", model.J(theta_true, X_test_aug, y_test))


if __name__ == "__main__":
    main()