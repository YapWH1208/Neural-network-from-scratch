import numpy as np

class NeuralNetwork:
    def __init__(self, X, y, hidden_dim, learning_rate, method):
        self.X = X
        self.y = y
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.method = method

        self.W_hidden = self.hidden_dim, self.X.shape[1] + 1
        self.W_output = 1, self.hidden_dim + 1

    def forward_propagation(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            x_i = X[i: i+1]
            x_i_1 = np.hstack([x_i, [[1]]])
            x_h_i = self.sigmoid(self.W_hidden @ x_i_1.T)
            x_h_i_1 = np.vstack([x_h_i, [1]])
            x_o = self.W_output @ x_h_i_1
            y_pred.append([x_o[0][0]])
            
        return np.array(y_pred), x_h_i_1, x_i_1
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def update(self, X, y):
        for i in range(X.shape[0]):
            x_i = X[i: i+1]
            y_i = y[i] 
            y_pred_i, x_h_i_1, x_i_1 = self.forward_propagation(x_i)
            delta_k = y_i - y_pred_i
            Gradient_output  = - delta_k * x_h_i_1.T
            self.W_output = self.W_output - self.learning_rate * Gradient_output
            delta_h = x_h_i_1 * ( 1 - x_h_i_1) * self.W_output.T * delta_k
            Gradient_hidden = np.zeros_like(self.W_hidden)
            for h in range(self.W_hidden.shape[0]):
                Gradient_hidden[h, :] = - delta_h[h] * x_i_1
            self.W_hidden = self.W_hidden - self.learning_rate * Gradient_hidden

    def Hinge_Loss(self, actual_y, predicted_y):
        loss = np.maximum(0, 1 - actual_y * predicted_y)
        return loss.mean()
    
    def Mean_Squared_Error(self, actual_y, predicted_y):
        loss = np.square(actual_y - predicted_y)
        return loss.mean()

    def train(self, epochs):
        for e in range(epochs):
            self.update()

            steps = epochs/10
            if (e+1) % steps == 0:
                y_pred = self.forward_propagation(self.X)
                if self.method == "Classification":
                    accuracy = np.sum(np.sign(self.y) == np.sign(y_pred[:, 0])) / self.y.shape[0]
                    loss = self.Hinge_Loss(self.y, y_pred)
                    print(f"Epoch {e+1}/{epochs}\t\tAccuracy: {accuracy * 100:.2f}%\tLoss: {loss:.6f}")
                elif self.method == "Regression":
                    loss = self.Mean_Squared_Error(self.y, y_pred.ravel())
                    print(f"Epoch {e+1}/{epochs}\t\tMSE: {loss:.6f}")
                else:
                    raise ValueError("Aiyo, What you cHOose!?!?!")
                


# Sample Usage
import pandas as pd
# Import X and y here

NN = NeuralNetwork(X,y,hidden_dim=2, learning_rate=0.001,method="Regression")
NN.train(epochs=20)
NN.forward_propagation(X_test)