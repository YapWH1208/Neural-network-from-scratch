import pandas as pd
from neural_network import NeuralNetwork

def main(filepath):
    data = pd.read_csv(filepath)

    X = data.drop(columns=["Age","Gender"])
    y = data.Age
    y_age = y/100

    NN = NeuralNetwork(X, y_age, method="Regression", learning_rate=0.001, hidden_dim=2)
    NN.train(epochs=50)

if __name__ == "__main__":
    main("Change Filepath Here")
