import numpy as np

class Initializer():
    def __init__(self) -> None:
        pass

    def Xavier(dim1, dim2):
        """
        Xavier Initialize
        For Tanh

        Args:
        - dim1: Dimension 1
        - dim2: Dimension 2

        Returns:
        Uniform Distribution after Xavier Initialize
        """
        limit = np.sqrt(6 / (dim1 + dim2))
        return np.random.uniform(-limit, limit, (dim1, dim2))
    
    def KaiMing(dim1, dim2, alpha):
        """
        KaiMing Initialize
        For ReLU

        Args:
        - dim1: Dimension 1
        - dim2: Dimension 2

        Returns:
        Uniform Distribution after KaiMing Initialize
        """
        limit = np.sqrt(2 / dim1 * (1 + alpha**2))
        return np.random.uniform(-limit, limit, (dim1, dim2))

class Normalizer():
    def __init__(self, x):
        self.X = x
    
    def Min_Max_Normalization(self):
        """
        Perform Min-Max normalization on the input data.

        Returns:
        list: Normalized data with values scaled between 0 and 1.
        """
        if not self.X:
            raise ValueError("Input data is empty.")

        if isinstance(self.X, np.ndarray):
            self.X = self.X.tolist()

        min_value = min(self.X)
        max_value = max(self.X)

        normalized_data = [(value - min_value) / (max_value - min_value) for value in self.X]

        return normalized_data

    def z_score_standardization(self):
        """
        Perform Z-score standardization on the input data.

        Returns:
        list: Standardized data with a mean of 0 and standard deviation of 1.
        """
        if not self.X:
            raise ValueError("Input data is empty.")

        if isinstance(self.X, np.ndarray):
            self.X = self.X.tolist()

        mean_value = np.mean(self.X)
        standard_deviation = np.std(self.X)

        standardized_data = [(value - mean_value) / standard_deviation for value in self.X]

        return standardized_data