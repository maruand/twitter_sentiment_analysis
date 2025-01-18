import os
import setuptools.dist
from kaggle.api.kaggle_api_extended import KaggleApi
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow_hub as hub
from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from timeit import default_timer as timer 
from tqdm import tqdm




def download_data(path):
    # Ensure the Kaggle API credentials are set up
    os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')

    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download the dataset
    dataset_owner = 'jp797498e/twitter-entity-sentiment-analysis'

    api.dataset_download_files(dataset_owner, path=path, unzip=True)

def load_data():
    # Load the data
    data_df = pd.read_csv('data/twitter_training.csv')
    data_val_df = pd.read_csv('data/twitter_validation.csv')
    return data_df, data_val_df

def preprocess_data(data):
    # Preprocess the data
    columns = ['tweet_id', 'entity','sentiment','tweet']
    columns_to_drop =['tweet_id', 'entity']
    data.columns = columns
    data = data.drop(columns=columns_to_drop, axis=1)
    data = data.dropna()

    # Encode the sentiment
    sentiment_map = {'Positive': 2, 'Neutral': 1, 'Negative': 0, 'Irrelevant': 3}
    data['sentiment'] = data['sentiment'].map(sentiment_map)

    # Ensure all data types are correct
    data['tweet'] = data['tweet'].astype(str)
    data['sentiment'] = data['sentiment'].astype(int)

    return data

def split_data(data):
    # Split the data
    train, test= train_test_split(data, test_size=0.2, random_state=42)

    X_train = train['tweet'].values
    y_train = train['sentiment'].values

    X_test = test['tweet'].values
    y_test = test['sentiment'].values

    return X_train, X_test, y_train, y_test

def to_pytorch(X, y):
    X_numpy = X.numpy()
    X_train = torch.tensor(X_numpy)

    y_train = y.clone().detach()

    return X_train, y_train

def accuracy(y_pred, y_true):
    correct = torch.eq(y_pred, y_true).sum().item()
    acc = correct / y_true.size(0)
    return acc


def train(n_epochs, model, lr, X_train, y_train, X_test, y_test):
    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize arrays to store loss and accuracy
    train_losses = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)
    train_accuracies = np.zeros(n_epochs)
    test_accuracies = np.zeros(n_epochs)
    epochs = np.zeros(n_epochs)

    ### Training loop
    model.train()
    for epoch in range(n_epochs):
        # Forward pass

        y_logits = model(X_train)
        y_pred = F.softmax(y_logits, dim=0).argmax(dim=1)

        # Compute the loss and accuracy
        train_loss = loss_fn(y_logits, y_train)
        train_acc = accuracy(y_pred, y_train)

        # Store the loss and accuracy
        train_losses[epoch] = train_loss.item()
        train_accuracies[epoch] = train_acc

        # Zero gradients
        optimizer.zero_grad()

        # Bckward pass
        train_loss.backward()

        # Update the weights
        optimizer.step() 

        ### Test loop

        model.eval()

        with torch.inference_mode():
            y_logits = model(X_test)
            y_pred = F.softmax(y_logits, dim=1).argmax(dim=1)

            test_loss = loss_fn(y_logits, y_test)
            test_acc = accuracy(y_pred, y_test)

            # Store the loss and accuracy
            test_losses[epoch] = test_loss.item()
            test_accuracies[epoch] = test_acc
            epochs[epoch] = epoch
        if epoch % 10 == 0:
            print(f'Epoch {epoch}| train loss: {train_loss.item():.4f}, train acc: {train_acc:.4f}| test loss: {test_loss.item():.4f}, test acc: {test_acc:.4f}')

    return train_losses, test_losses, train_accuracies, test_accuracies, epochs
    
        
class SentimentAnalysisModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(128, 4)

        self.dropout = nn.Dropout(0.2)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        #x = self.fc3(x)
        #x = self.relu(x)
        x = self.fc4(x)

        return x

def plot_loss(train_loss, test_loss, epochs):
    plt.plot(epochs, train_loss, label='train loss')
    plt.plot(epochs, test_loss, label='test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def sentiment(pred):
    if pred == 0:
        return 'Negative'
    elif pred == 1:
        return 'Neutral'
    elif pred == 2:
        return 'Positive'
    elif pred == 3:
        return 'Irrelevant'

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive', 'Irrelevant'], yticklabels=['Negative', 'Neutral', 'Positive', 'Irrelevant'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def main():
    download_data('data')
    data_df, data_val_df = load_data()
    data = preprocess_data(data_df)
    X_train, X_test, y_train, y_test = split_data(data)
    X_train = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')(X_train)
    X_test = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')(X_test)

    X_train, y_train = to_pytorch(X_train, y_train)
    X_test, y_test = to_pytorch(X_test, y_test)

    model = SentimentAnalysisModel()
    n_epochs = 100
    lr = 0.001
    train_losses, test_losses, train_accuracies, test_accuracies, epochs = train(n_epochs, model, lr, X_train, y_train, X_test, y_test)
    plot_loss(train_losses, test_losses, epochs)

    model.eval()
    y_logits = model(X_test)
    y_pred = F.softmax(y_logits, dim=1).argmax(dim=1)
    y_pred = y_pred.numpy()
    y_test = y_test.numpy()

    plot_confusion_matrix(y_test, y_pred)

    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
    