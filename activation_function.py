import numpy as np

class ActivationFunction():
    def sigmoid(self, x):
        """
        Sigmoid activation function

        Args：
        - x: input value

        Return:
        - Value after the dataset undergoes the sigmoid function
        """
        return 1/(1 + np.exp(-x))
    
    def relu(self, x):
        """
        ReLU activation function.

        Args:
        - x: Input value.

        Returns:
        - Output of the ReLU function.
        """
        return np.maximum(0, x)
    
    def softplus(self, x):
        """
        Branches of ReLU

        Args:
        - x: Input value

        Returns:
        - Output of softplus function
        """
        return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))
    
    def gelu(self, x):
        """
        Gaussian Error Linear Unit Activation Function

        Args:
        - x: input data

        Returns:
        - data after gelu function
        """
        return 0.5*x*(1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3)))
    
    def softmax(self, x):
        """
        Softmax activation function

        Args:
        - x: input data

        Returns：
        - data after softmax function
        """
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities
    
    def tanh(self, x):
        """
        Hyperbolic tangent activation function

        Args:
        - x: input data

        Returns:
        - hyerbolic tangent of x
        """
        return np.tanh(x)
    
    def leaky_relu(self, x, constant:float=0.001):
        """
        Leaky ReLU activation function

        Args:
        - x: input data
        - constant: constant for the funtion. Default = 0.001

        Returns:
        - data after leaky relu function
        """
        return np.maximum(constant * x, x)
    
    def none(self, x):
        """
        Nothing special
        """
        return x