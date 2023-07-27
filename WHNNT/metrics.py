import numpy as np

class Metrics:
    # General
    def Mean_Squared_Error(self, actual_y, predicted_y):
        """
        Mean Squared Error

        Args:
        - actual_y: True labels
        - predicted_y: Predictions by the model

        Returns:
        - Mean of the mean squared error
        """
        loss = np.square(actual_y - predicted_y)
        return loss.mean()
    
    def Mean_Absolute_Error(self, actual_y, predicted_y):
        """
        Mean Absolute Error

        Args:
        - actual_y: True labels
        - predicted_y: Predictions by the model

        Returns:
        - Mean of the mean absolute error
        """
        loss = np.abs(actual_y - predicted_y)
        return loss.mean()
    
    # Classification
    def Cross_Entropy_Loss(self, actual_y, predicted_y):
        """
        Cross-Entropy Loss for Multi-class Classification Problem

        Args:
        - actual_y: True labels
        - predicted_y: Predictions by the model

        Returns:
        - Mean of the cross-entropy loss
        """
        loss = actual_y * np.log(predicted_y)
        return loss.mean()

    def Binary_Cross_Entropy_Loss(self, actual_y, predicted_y):
        """
        Cross-Entropy Loss for Binary Classification Problem
        For True Lables of [0, 1]

        Args:
        - actual_y: True labels
        - predicted_y: Predictions by the model

        Returns:
        - Mean of the binary cross-entropy loss
        """
        epsilon = 1e-15
        loss = -(0.5 * (1 + actual_y) * np.log(0.5 * (1 + predicted_y) + epsilon) + 0.5 * (1 - actual_y) * np.log(0.5 * (1 - predicted_y) + epsilon))
        return loss.mean()
    
    def Hinge_Loss(self, actual_y, predicted_y):
        """
        Hinge loss for Binary Classification Problems
        For True Lables of [-1, 1]
        Normally used in SVM

        Args:
        - actual_y: True labels
        - predicted_y: Predictions by the model. 

        Returns:
        - Mean of the hinge loss
        """
        loss = np.maximum(0, 1 - actual_y * predicted_y)
        return loss.mean()
    
    # Regression
    def huber_loss(self, actual_y, predicted_y, delta:float=1.0):
        """
        Huber loss for Regression Problems

        Args:
        - actual_y: True labels
        - predicted_y: Predictions by the model. 
        - Delta: Delta of the loss function. Default = 1.0

        Returns:
        - Mean of the huber loss
        """
        error = np.abs(actual_y - predicted_y)
        loss = 0.5 * np.minimum(error, delta)**2 + delta * (error - np.minimum(error, delta))
        return loss.mean()
    
    def smooth_l1_loss(self, actual_y, predicted_y, delta:float=1.0):
        """
        Smooth l1 loss loss for Regression Problems

        Args:
        - actual_y: True labels
        - predicted_y: Predictions by the model. Default = 1.0

        Returns:
        - Mean of the smooth l1 loss
        """
        error = np.abs(actual_y - predicted_y)
        loss = np.where(error < delta, 0.5 * error ** 2, delta * (error - 0.5 * delta))
        return loss.mean()