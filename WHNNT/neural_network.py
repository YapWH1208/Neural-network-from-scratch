import numpy as np
import matplotlib.pyplot as plt
import time
from optimizer import Optimizer
from preprocessing import Initializer, Normalizer
from metrics import Metrics

class NeuralNetwork(Optimizer, Initializer, Normalizer, Metrics):
    def __init__(self, X, y, method:str,
                 optimizer:str = 'Adam',
                 activation_function_hidden:str = 'sigmoid',
                 activation_function_output:str = 'none',
                 hidden_dim:int=10,
                 learning_rate:float=0.01,
                 initializer:str="Xavier",
                 plot_graph:bool=False):
        """
        Initialization of the Neural Network

        Args:
        - X: The data for prediction
        - y: The target to be predict
        - hidden_dim: Numbers of nodes in the hidden layer. Default = 10
        - learning_rate: Learning rate of the Neural Network. Default = 0.01
        - method: ["Regression","Classification"]
        - activation_function_hidden: ["sigmoid","relu","gelu","tanh","softmax","softplus","leaky_relu"]. Default = "sigmoid"
        - activation_function_output: ["sigmoid","relu","gelu","tanh","softmax","softplus","leaky_relu"]. Default = "none"
        - optimizer: ["SGD","Adam"]. Default = "Adam"
        - initializer: ["Xavier","KaiMing"]. Default = "Xavier"
        - plot_graph: Plot graph of performance. Default = False
        """
        self.X = X
        self.y = y
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.method = method
        self.optimizer = optimizer
        self.activation_function_hidden = activation_function_hidden
        self.activation_function_output = activation_function_output
        self.initializer = initializer
        self.plot = plot_graph
    
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
    
    def param_update(self):
        if self.optimizer == "Adam":
            self.opti.Adam(self.X, self.y)
        elif self.optimizer == "SGD":
            self.opti.SGD(self.X, self.y)

    def initialize_weights(self, dim1, dim2, alpha=0.01):
        if self.initializer == "Xavier":
            return Initializer.Xavier(dim1, dim2)
        elif self.initializer == "KaiMing":
            return Initializer.KaiMing(dim1, dim2, alpha)
        else:
            return None
    
    def plot_graph(self, epochs, loss_list, accuracy_list, start):
        """
        Plot Graphs

        Args:
        - epochs: total epochs of training
        - loss_list： loss from training
        - accuracy_list: accuracy from training when classification
        - start: starting time of the training process
        """
        if self.method == "Classification":     
            plt.plot(np.arange(epochs/10, epochs+1, step=epochs/10), accuracy_list)
            plt.xlabel("Epochs")
            plt.xticks(np.arange(epochs/10, epochs+1, step=epochs/10))
            plt.ylabel("Accuracy")
            plt.title('Accuracy over Epochs')
            plt.show()

        plt.plot(np.arange(epochs/10, epochs+1, step=epochs/10), loss_list)
        plt.xlabel("Epochs")
        plt.xticks(np.arange(epochs/10, epochs+1, step=epochs/10))
        plt.ylabel("Loss")
        plt.title('Loss over Epochs')
        plt.show()
        print(f"Time Used:{(time.time() - start)/60:.2f} min")
    
    def initialize(self):
        self.W_hidden = self.initialize_weights(self.hidden_dim, self.X.shape[1] + 1)
        self.W_output = self.initialize_weights(1, self.hidden_dim + 1)
    
    def predict(self, X):
        y_pred,_,_ = self.opti.forward_propogation(X)
        return y_pred
    
    def train(self, epochs:int):
        """
        Training function of the neural network

        Args:
        - epochs: Total time the neural network train using the dataset
        """
        accuracy_list = []
        loss_list = []

        print(f"Total training steps: {epochs * self.X.shape[0]}")
        print(f"Method:{self.method}\tOptimizer:{self.optimizer}\t\tActivation Function:{self.activation_function_hidden}\n")
        start = time.time()

        self.initialize()
        self.opti = Optimizer(self.activation_function_hidden, self.activation_function_output, self.W_hidden, self.W_output, optim=self.optimizer, lr=self.learning_rate)
        for e in range(epochs):
            self.param_update()

            steps = epochs/10
            if (e+1) % steps == 0:
                y_pred = self.predict(self.X)
                if self.method == "Classification":
                    accuracy = np.sum(np.sign(self.y) == np.sign(y_pred[:, 0])) / self.y.shape[0]
                    loss = self.Hinge_Loss(self.y, y_pred)
                    accuracy_list.append(accuracy)
                    loss_list.append(loss)
                    print(f"Epoch {e+1}/{epochs}\t\tAccuracy: {accuracy * 100:.2f}%\tLoss: {loss:.6f}")
                elif self.method == "Regression":
                    loss = self.Mean_Squared_Error(self.y, y_pred.ravel())
                    loss_list.append(loss)
                    print(f"Epoch {e+1}/{epochs}\t\tMSE: {loss:.6f}")
                else:
                    raise ValueError("Aiyo, What you cHOose!?!?!")
        if self.plot:
            self.plot_graph(epochs, loss_list, accuracy_list, start)