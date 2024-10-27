import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

from sklearn.metrics import classification_report, confusion_matrix, f1_score, fbeta_score
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_rnn_layers, n_outputs) -> None:
        super().__init__()
        self.D = n_inputs
        self.M = n_hidden
        self.K = n_outputs
        self.L = n_rnn_layers

        self.rnn = nn.LSTM(
            input_size=self.D,
            hidden_size=self.M,
            num_layers=self.L,
            batch_first=True
        )
        # First fully connected layer
        # self.fc1 = nn.Linear(n_hidden, n_hidden)
        # self.bn1 = nn.BatchNorm1d(n_hidden)
        # self.dropout1 = nn.Dropout(0.2)
        # # Second fully connected layer
        # self.fc2 = nn.Linear(n_hidden, n_hidden)
        # self.bn2 = nn.BatchNorm1d(n_hidden)
        # self.dropout2 = nn.Dropout(0.2)
        # Third fully connected layer
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.bn3 = nn.BatchNorm1d(n_hidden)
        self.dropout3 = nn.Dropout(0.2)
        # Output layer
        self.fc4 = nn.Linear(n_hidden, n_outputs)

    def forward(self, X):
        # initial hidden states
        h0 = torch.zeros(self.L, X.size(0), self.M).to(device)
        c0 = torch.zeros(self.L, X.size(0), self.M).to(device)

        # get the rnn output unit
        out, _ = self.rnn(X, (h0, c0))

        # we only want h(T) at the final time step
        x = out[:, -1, :]
        # First layer with LeakyReLU activation
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = self.dropout1(x)
        # # Second layer
        # x = F.relu(self.bn2(self.fc2(x)))
        # x = self.dropout2(x)
        # Third layer
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        # Output layer
        x = self.fc4(x)
        return x


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                optimizer: torch.optim.Optimizer, criterion: nn.Module,
                max_epochs: int, early_stop_patience: int = 15,
                save_path: str = "best_ass2_model.pt", threshold: float = 0.5) -> tuple:
    """
    Manages the entire model training loop, including forward and backward passes, 
    optimization steps, and tracking of both training and validation losses over epochs.
    Includes early stopping functionality and saves the model with the best validation F1 score for class 1.
    The training function has been improved to handle class imbalance by incorporating class weights.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        criterion (nn.Module): Loss function.
        max_epochs (int): Maximum number of epochs to train.
        early_stop_patience (int): Number of epochs with no improvement before stopping.
        save_path (str): File path for saving the best model state.
        threshold (float): Threshold for converting probabilities to class predictions.

    Returns:
        tuple: (trained model, training losses, validation losses, training F1 scores, validation F1 scores)
    """
    best_val_f1 = 0.0
    best_epoch = 0

    model.to(device)

    # # Compute class weights based on the training data
    # class_counts = torch.zeros(2)
    # for _, targets in train_loader:
    #     class_counts += torch.bincount(targets, minlength=2)
    # class_weights = class_counts.sum() / (2 * class_counts)
    # class_weights = class_weights.to(device)

    # # Update criterion to include class weights
    # if isinstance(criterion, nn.CrossEntropyLoss):
    #     criterion.weight = class_weights
    # else:
    #     # If criterion is not CrossEntropyLoss, ensure it supports class weights
    #     raise ValueError("Criterion must be an instance of nn.CrossEntropyLoss to use class weights.")

    train_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_train_losses = []
        all_train_targets = []
        all_train_preds = []

        for inputs, targets in train_loader:
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Record loss
            epoch_train_losses.append(loss.item())

            # Get probabilities for class 1 using softmax
            probabilities = nn.functional.softmax(outputs, dim=1)[:, 1]

            # Apply threshold to get predictions
            preds = (probabilities >= threshold).long()
            all_train_targets.extend(targets.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy())

        # Compute mean training loss and F1 score
        train_loss = np.mean(epoch_train_losses)
        train_losses.append(train_loss)
        # train_f1 = f1_score(all_train_targets, all_train_preds, pos_label=1, zero_division=0)
        train_f1 = fbeta_score(all_train_targets, all_train_preds, beta=0.3, pos_label=1, zero_division=0)
        train_f1_scores.append(train_f1)

        # Validation phase
        model.eval()
        epoch_val_losses = []
        all_val_targets = []
        all_val_preds = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Record loss
                epoch_val_losses.append(loss.item())

                # Get probabilities for class 1 using softmax
                probabilities = nn.functional.softmax(outputs, dim=1)[:, 1]

                # Apply threshold to get predictions
                preds = (probabilities >= threshold).long()
                all_val_targets.extend(targets.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())

        # Compute mean validation loss and F1 score
        val_loss = np.mean(epoch_val_losses)
        val_losses.append(val_loss)
        # val_f1 = f1_score(all_val_targets, all_val_preds, pos_label=1, zero_division=0)
        val_f1 = fbeta_score(all_val_targets, all_val_preds, beta=0.3, pos_label=1, zero_division=0)
        val_f1_scores.append(val_f1)

        # Check for improvement in validation F1 score
        if val_f1 > best_val_f1:
            print(f"Epoch {epoch}: Validation F1 improved ({best_val_f1:.4f} --> {val_f1:.4f}). Saving model...")
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
        elif epoch - best_epoch >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}: validation F1 did not improve for {early_stop_patience} epochs.")
            break
        else:
            print(f"Epoch {epoch}: Validation F1 did not improve ({val_f1:.4f}).")

        # Optionally, print training progress
        print(f"Epoch {epoch}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

    # Load the best model state
    model.load_state_dict(torch.load(save_path, map_location=device))

    return model, train_losses, val_losses, train_f1_scores, val_f1_scores


def evaluate_model(model: nn.Module, test_loader: DataLoader, ticker: str = 'All Stocks',
                   save_confusion_matrix: bool = True,
                   path: str = 'C:/Users/candr/Documents/masters/Security-Analysis/LSTM_/results/confusion_matrix.png'
                   ) -> float:
    """
    Evaluates the trained model's performance on the test dataset.
    Computes accuracy, prints the classification report, and plots the confusion matrix.

    Args:
        model (nn.Module): The trained PyTorch model.
        test_loader (DataLoader): DataLoader for the test dataset.
        ticker (str, optional): The stock ticker symbol. Defaults to ''.
        save_confusion_matrix (bool, optional): Whether to save the confusion matrix plot. Defaults to False.
        path (str, optional): The file path to save the plot. Defaults to 'confusion_matrix.png'.

    Returns:
        float: The accuracy of the model on the test dataset.
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = correct / total

    # Classification report
    target_names = ['less than 20% gain', 'more than 20% gain']
    print(classification_report(all_targets, all_preds, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    if ticker:
        plt.title(f'{ticker} - Confusion Matrix')
    else:
        plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if save_confusion_matrix:
        plt.savefig(path)
    else:
        plt.show()

    return accuracy
