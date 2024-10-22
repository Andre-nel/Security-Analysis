import torch
from torch import nn
import torch.nn.functional as F


class ANN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ANN, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        # Third fully connected layer
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2)
        # Output layer
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First layer with LeakyReLU activation
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout1(x)
        # Second layer
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout2(x)
        # Third layer
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        x = self.dropout3(x)
        # Output layer
        x = self.fc4(x)
        return x


class ANN_Deeper(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ANN, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        # Third fully connected layer
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2)

        # First fully connected layer
        self.fc4 = nn.Linear(hidden_size, hidden_size*2)
        self.bn4 = nn.BatchNorm1d(hidden_size*2)
        self.dropout4 = nn.Dropout(0.2)
        # Second fully connected layer
        self.fc5 = nn.Linear(hidden_size*2, hidden_size*2)
        self.bn5 = nn.BatchNorm1d(hidden_size*2)
        self.dropout5 = nn.Dropout(0.2)
        # Third fully connected layer
        self.fc6 = nn.Linear(hidden_size*2, hidden_size*2)
        self.bn6 = nn.BatchNorm1d(hidden_size*2)
        self.dropout6 = nn.Dropout(0.2)

        # Output layer
        self.fc7 = nn.Linear(hidden_size*2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First layer with LeakyReLU activation
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout1(x)
        # Second layer
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout2(x)
        # Third layer
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        x = self.dropout3(x)

        # Four layer with LeakyReLU activation
        x = F.leaky_relu(self.bn4(self.fc4(x)), negative_slope=0.01)
        x = self.dropout4(x)
        # Five layer
        x = F.leaky_relu(self.bn5(self.fc5(x)), negative_slope=0.01)
        x = self.dropout5(x)
        # Six layer
        x = F.leaky_relu(self.bn6(self.fc6(x)), negative_slope=0.01)
        x = self.dropout6(x)

        # Output layer
        x = self.fc7(x)
        return x
