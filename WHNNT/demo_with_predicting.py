import pandas as pd
from neural_network import NeuralNetwork

def predict(train, predict):
    data = pd.read_csv(train)

    X = data.drop(columns=["Age","Gender"])
    y = data.Age
    y_age = y/100

    predict_data = pd.read_csv(predict)
    X_pred = predict_data

    NN = NeuralNetwork(X, y_age, method="Regression", learning_rate=0.001, hidden_dim=2)
    NN.train(epochs=50)
    NN.predict(X_pred)

if __name__ == "__main__":
    train_filepath = "Insert Filepath Here"
    predict_filepath = "Insert Filepath Here"
    predict(train_filepath, predict_filepath)
