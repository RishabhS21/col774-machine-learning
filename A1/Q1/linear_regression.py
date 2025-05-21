# Imports - you can add any other permitted libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class LinearRegressor:
    def __init__(self):
        self.theta = None
        self.iterations = 0
        self.loss_history = []
        self.theta_history = []
        # pass
    
    # Define hypothesis function
    def h(self, theta, X):
        return X.dot(theta)

    # Define cost(loss) function
    def J(self, theta, X, y):
        m = len(y)
        # calculate the difference between the actual and predicted values
        diff = self.h(theta, X) - y
        # mean of squared sum
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
        theta = np.zeros((X.shape[1], 1))
        m = len(y)
        #learning_rate is the Learning rate
        iterations = 0
        prev_cost = self.J(theta, X, y)
        eps = 1e-9
        converged = False
        loss_history = []
        theta_history = []
        while not converged:
            iterations += 1
            H = self.h(theta, X)
            gradient = np.dot(X.T,(H-y))/(2*m)
            # print(X.T.shape, H.shape, gradient.shape)
            theta -= learning_rate * gradient
            current_cost = self.J(theta, X, y)
            if abs(current_cost - prev_cost) < eps or iterations > 1000:
                converged = True
            prev_cost = current_cost
            loss_history.append(current_cost)
            theta_history.append(theta.flatten())

        # Storing the values
        self.theta = theta
        self.iterations = iterations
        self.loss_history = loss_history
        self.theta_history = theta_history
        # Hypothesis plot
        plt.title('Input Dataset')
        plt.scatter(X[:,1:],y,label='Given Data', color='blue')
        plt.plot(X[:,1:],H,label='Hypothesis', color='red')
        plt.xlabel(' Acidity(X) ')
        plt.ylabel(' Density(Y) ')
        plt.legend()
        plt.show()

        # returning the requirement
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
            The predicted target values.
        """
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        if self.theta is None:
            raise Exception('Fit the model before prediction')
        # X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        y_pred = self.h(self.theta, X)
        return y_pred.flatten()
        # pass

    # Gradient Descent plot
    def plot_gd(self):
        fig, ax1 = plt.subplots()
        # plot thetas over time
        theta0_history = [theta[0] for theta in self.theta_history]
        theta1_history = [theta[1] for theta in self.theta_history]
        ax1.plot(theta0_history, label='$\\theta_{0}$', linestyle='--', color='blue')
        ax1.plot(theta1_history, label='$\\theta_{1}$', linestyle='-', color='blue')
        ax1.set_xlabel('Iterations'); ax1.set_ylabel('$\\theta$')

        # plot loss function over time
        ax2 = ax1.twinx()
        ax2.plot(self.loss_history, label='Loss function', color='red')
        ax2.set_title('Values of $\\theta$ and $J(\\theta)$ over iterations')
        ax2.set_ylabel('$J(\\theta)$')
        
        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='center')

        fig.tight_layout()
        plt.show()

    # Draw a 3-dimensional mesh showing the error function (J(θ)) on z-axis and the parameters in the x − y plane.
    def plot_mesh(self, X, y):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        theta0_history = [theta[0] for theta in self.theta_history]
        theta1_history = [theta[1] for theta in self.theta_history]
        # A meshgrid for theta0 and theta1
        theta0_vals = np.linspace(0, 12, 100)
        theta1_vals = np.linspace(0, 60, 100)
        Theta0, Theta1 = np.meshgrid(theta0_vals, theta1_vals)

        # filling the cost values for each combination of theta0 and theta1
        Cost = np.zeros_like(Theta0)
        for i in range(len(theta0_vals)):
            for j in range(len(theta1_vals)):
                theta = np.array([[theta0_vals[i]], [theta1_vals[j]]])
                # X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
                Cost[i, j] = self.J(theta, X, y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surface = ax.plot_surface(Theta0, Theta1, Cost, cmap='viridis')

        # labels and title
        ax.set_xlabel('$\\theta_{0}$')
        ax.set_ylabel('$\\theta_{1}$')
        ax.set_zlabel('Cost J($\\theta$)')
        ax.set_title('3-dimensional mesh for Error Function')

        # Initialising scatter plot for displaying current parameter values during GD
        scatter = ax.scatter([], [], [], color='red', s=50)

        # Function to update scatter plot data and pause for visualization
        def update_scatter(i):
            scatter._offsets3d = (theta0_history[:i+1], theta1_history[:i+1], self.loss_history[:i+1])
            plt.pause(0.2)  # Pause for 0.2 seconds for visualization

        # Animate the scatter plot over iterations
        for i in range(self.iterations):
            update_scatter(i)
        plt.show()

    # the contours of the error function 
    def plot_contours(self, X, y):
        # A meshgrid for theta0 and theta1
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        theta0_history = [theta[0] for theta in self.theta_history]
        theta1_history = [theta[1] for theta in self.theta_history]
        theta0_vals = np.linspace(0, 12, 20)
        theta1_vals = np.linspace(0, 60, 20)
        Theta0, Theta1 = np.meshgrid(theta0_vals, theta1_vals)

        # cost values for each combination of theta0 and theta1
        Cost = np.zeros_like(Theta0)
        for i in range(len(theta0_vals)):
            for j in range(len(theta1_vals)):
                theta = np.array([[theta0_vals[i]], [theta1_vals[j]]])
                Cost[i, j] = self.J(theta, X, y)

        fig, ax = plt.subplots()

        contour = ax.contour(Theta0, Theta1, Cost, cmap='viridis')

        # Set labels and title
        ax.set_xlabel('$\\theta_{0}$')
        ax.set_ylabel('$\\theta_{1}$')
        ax.set_title('Contour Plot of Error Function $J(\\theta)$')

        # Initialize scatter plot for displaying current parameter values during GD
        scatter = ax.scatter([], [], color='red', s=1)

        # Create a function to update scatter plot data and pause for visualization
        def update_scatter(i):
            scatter.set_offsets(np.column_stack((theta0_history[:i+1], theta1_history[:i+1])))
            plt.pause(0.2)  # Pause for 0.2 seconds for visualization

        # Animate the scatter plot over iterations
        for i in range(self.iterations):
            update_scatter(i)

        plt.show()

    


def main():
    # Read the CSV file
    X = pd.read_csv('../data/Q1/linearX.csv', header=None)
    y = pd.read_csv('../data/Q1/linearY.csv', header=None)
    X = X.values
    # print(X.shape)
    y = y.values
    model = LinearRegressor()
    theta_history = model.fit(X, y, 0.1)
    y_pred = model.predict(X)
    # Print the parameters
    # model.plot_gd()
    # model.plot_mesh(X, y)
    # model.plot_contours(X, y)
    print(model.theta)
    print(model.iterations)
    print(model.loss_history[-1])
    # np.savetxt('../data/Q1/linearY_pred.csv', y_pred, fmt='%.4f', delimiter=',')
    

if __name__ == "__main__":
    main()