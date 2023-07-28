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
    def __init__(self) -> None:
        pass
