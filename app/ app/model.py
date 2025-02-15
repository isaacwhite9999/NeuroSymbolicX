# app/model.py

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from app.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class SimpleNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        A simple feed-forward neural network.
        """
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)


def train_model(epochs: int = settings.EPOCHS, lr: float = settings.LEARNING_RATE) -> SimpleNN:
    """
    Train the neural network on synthetic binary classification data and save the model.
    
    Args:
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        
    Returns:
        SimpleNN: The trained neural network model.
    """
    logger.info("Generating synthetic data for training...")
    X, y = make_classification(
        n_samples=1000,
        n_features=settings.INPUT_DIM,
        n_informative=settings.INPUT_DIM,
        n_redundant=0,
        random_state=42,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32).reshape(-1, 1)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SimpleNN(settings.INPUT_DIM, settings.HIDDEN_DIM, settings.OUTPUT_DIM)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Training on device: {device}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.from_numpy(X_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).to(device)

    logger.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    try:
        torch.save(model.state_dict(), settings.MODEL_PATH)
        logger.info(f"Model saved to {settings.MODEL_PATH}")
    except Exception as e:
        logger.exception("Error saving the model: %s", e)
        raise

    return model


def load_model() -> SimpleNN:
    """
    Load the pre-trained model from disk.
    
    Returns:
        SimpleNN: The loaded model in evaluation mode.
    """
    model = SimpleNN(settings.INPUT_DIM, settings.HIDDEN_DIM, settings.OUTPUT_DIM)
    try:
        state_dict = torch.load(settings.MODEL_PATH, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        raise
    model.eval()
    return model

