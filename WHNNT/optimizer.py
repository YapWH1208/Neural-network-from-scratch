import numpy as np
from activation_function import ActivationFunction

class Optimizer(ActivationFunction):
    def __init__(self, activation_function_hidden, activation_function_output, w_hidden, w_output, optim, lr):
        self.acti_hidden = self.get_activation_function(activation_function_hidden)
        self.acti_output = self.get_activation_function(activation_function_output)
        self.W_hidden = w_hidden
        self.W_output = w_output
        self.learning_rate = lr

        if optim == "Adam":
            self.A_hidden = np.zeros_like(self.W_hidden)
            self.F_hidden = np.zeros_like(self.W_hidden)
            self.A_output = np.zeros_like(self.W_output)
            self.F_output = np.zeros_like(self.W_output)
            self.rho = 0.999
            self.rho_f = 0.9
            self.epsilon = 1e-8

    def get_activation_function(self, name:str):
        """
        Choosing activation functions for the neural network

        Args:
        - name: user's input of the name of activation functions

        Returns:
        - activation_functions defined by user
        """
        activation_function = {
            'sigmoid' : self.sigmoid,
            'relu' : self.relu,
            'softplus' : self.softplus,
            'softmax' : self.softmax,
            'tanh' : self.tanh,
            'gelu' : self.gelu,
            'leaky_relu' : self.leaky_relu,
            'none' : self.none
        }

        if name in activation_function:
            return activation_function[name]
        else:
            raise ValueError(f'There is no activation function {name}')

    def forward_propogation(self, X):
        """
        Forward propogation/Predict of the neural network

        Args:
        - X: The data for prediction

        Returns:
        - np.array(y_pred): Predicted target
        - x_h_i_1: w2
        - x_i_1: w1
        """
        y_pred = []
        for i in range(X.shape[0]):
            x_i = X[i: i+1] # Select a data point
            x_i_1 = np.hstack([x_i, [[1]]]) # Add Bias Term
            x_h_i = self.acti_hidden(self.W_hidden @ x_i_1.T)
            x_h_i_1 = np.vstack([x_h_i, [1]])
            x_o = self.acti_output(self.W_output @ x_h_i_1)
            y_pred.append([x_o[0][0]])
            
        return np.array(y_pred), x_h_i_1, x_i_1

    def Adam(self, X, y):
        for i in range(X.shape[0]):
            x_i = X[i: i+1]
            y_i = y[i]
            
            y_pred_i, x_h_i_1, x_i_1 = self.forward_propogation(x_i)
            delta_k = y_i - y_pred_i

            # Gradient for output layer
            Gradient_output  = - delta_k * x_h_i_1.T

            self.A_output = self.rho * self.A_output + (1 - self.rho) * (Gradient_output)**2
            self.F_output = self.rho_f * self.F_output + (1 - self.rho_f) * (Gradient_output)
            self.learning_rate_adam = self.learning_rate * (np.sqrt(1 - self.rho**(i+1))/(1 - self.rho_f**(i+1)))
            
            # Gradient Descent rule
            self.W_output = self.W_output - ((self.learning_rate_adam * self.F_output)/np.sqrt(self.A_output + self.epsilon))
            delta_h = x_h_i_1 * ( 1 - x_h_i_1) * self.W_output.T * delta_k
            
            # Update each hidden unit
            Gradient_hidden = np.zeros_like(self.W_hidden)
            for h in range(self.W_hidden.shape[0]):
                # Gradient for hidden layer
                Gradient_hidden[h, :] = - delta_h[h] * x_i_1

            self.A_hidden = self.rho * self.A_hidden + (1 - self.rho) * (Gradient_hidden)**2
            self.F_hidden = self.rho_f * self.F_hidden + (1 - self.rho_f) * (Gradient_hidden)

            # Gradient Descent rule
            self.W_hidden = self.W_hidden - ((self.learning_rate_adam * self.F_hidden)/np.sqrt(self.A_hidden + self.epsilon))
        self.learning_rate = self.learning_rate_adam

    def SGD(self, X, y):
        for i in range(X.shape[0]):
            x_i = X[i: i+1]
            y_i = y[i]
            
            y_pred_i, x_h_i_1, x_i_1 = self.forward_propogation(x_i)
            delta_k = y_i - y_pred_i

            # Gradient for output layer
            Gradient_output  = - delta_k * x_h_i_1.T
            
            # Gradient Descent rule
            self.W_output = self.W_output - self.learning_rate * Gradient_output
            delta_h = x_h_i_1 * ( 1 - x_h_i_1) * self.W_output.T * delta_k
            
            # Update each hidden unit
            Gradient_hidden = np.zeros_like(self.W_hidden)
            for h in range(self.W_hidden.shape[0]):
                # Gradient for hidden layer
                Gradient_hidden[h, :] = - delta_h[h] * x_i_1
            
            # Gradient Descent rule
            self.W_hidden = self.W_hidden - self.learning_rate * Gradient_hidden