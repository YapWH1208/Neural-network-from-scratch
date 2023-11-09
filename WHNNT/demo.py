import pandas as pd
from neural_network import NeuralNetwork

def train(epochs:int, train):
    data = pd.read_csv(train)

    X = data.drop(columns=["Age","Gender"])
    y = data.Age # data.Gender
    y_age = y/100

    NN = NeuralNetwork(X, y_age, method="Regression", learning_rate=0.001, hidden_dim=2)
    NN.train(epochs=50)

    return NN

def predict(model, predict, save_state:bool=False):
    predict_data = pd.read_csv(predict)
    X_pred = predict_data

    predictions = model.predict(X_pred)

    if save_state:
        df = pd.DataFrame({"Age":predictions})
        df.to_csv("./predictions.csv")

    return predictions


if __name__ == "__main__":
    train_filepath = "Insert Filepath Here"
    predict_filepath = "Insert Filepath Here"
    epochs = 50

    model = train(50, train_filepath)
    predictions = predict(train_filepath, predict_filepath)
    print(predictions)
