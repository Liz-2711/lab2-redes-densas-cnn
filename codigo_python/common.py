"""
Utilidades compartidas para el Laboratorio 2, de redes densas a CNNs.

Junta lo que se repite igual en las 4 partes, semilla global, carga split y
escalado de un dataset tipo MNIST, las clases de los modelos DenseNet, ConvNet,
ConvNetDropout, y el ciclo de entrenamiento y evaluacion. Cada parteN importa
de aca en vez de repetir el mismo bloque de codigo 4 veces, asi no se puede
colar un hiperparametro distinto por copiar y pegar mal entre partes.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split

from dotenv import load_dotenv

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)  # una sola vez para todo el laboratorio, no antes de cada modelo

FASHION_MNIST_CLASSES = [
    "Camiseta/Top", "Pantalón", "Suéter", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Tenis", "Bolso", "Botín",
]


def load_split_scale(train_env_var, test_env_var):
    """Carga un dataset .npz con llaves images y labels usando rutas del .env,
    separa el train en train y val con stratify y random_state=RANDOM_SEED, y
    escala los pixeles a [0, 1] dividiendo entre 255.0. Nada de StandardScaler,
    los pixeles ya viven en un rango fijo y conocido, de 0 a 255.

    Devuelve tensores aplanados (N, 784) listos para la DenseNet,
    X_train, X_val, X_test, y_train, y_val, y_test.
    """
    load_dotenv(".env")
    train_fp = os.getenv(train_env_var)
    test_fp = os.getenv(test_env_var)

    train_data = np.load(train_fp)
    test_data = np.load(test_fp)

    x = train_data["images"].reshape(-1, 28 * 28)
    y = train_data["labels"]
    x_test = test_data["images"].reshape(-1, 28 * 28)
    y_test = test_data["labels"]

    x_train, x_val, y_train, y_val = train_test_split(
        x, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    x_train_scaled = x_train / 255.0
    x_val_scaled = x_val / 255.0
    x_test_scaled = x_test / 255.0

    X_train = torch.from_numpy(x_train_scaled).float()
    X_val = torch.from_numpy(x_val_scaled).float()
    X_test = torch.from_numpy(x_test_scaled).float()
    y_train_t = torch.from_numpy(y_train).long()
    y_val_t = torch.from_numpy(y_val).long()
    y_test_t = torch.from_numpy(y_test).long()

    return X_train, X_val, X_test, y_train_t, y_val_t, y_test_t


class DenseNet(nn.Module):
    """Parte 1.3, Linear de 784 a 128, ReLU, Linear de 128 a 10, logits crudos."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        output = F.relu(self.layer1(x))
        output = self.layer2(output)
        return output


class ConvNet(nn.Module):
    """Parte 3.3, Conv de 1 a 16, Pool, Conv de 16 a 32, Pool, Flatten, Linear de 1568 a 10."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc_out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # x: (batch_size, 1, 28, 28)
        output = self.pool(F.relu(self.conv1(x)))
        output = self.pool(F.relu(self.conv2(output)))
        output = output.view(output.size(0), -1)
        output = self.fc_out(output)
        return output


class ConvNetDropout(nn.Module):
    """Parte 4: igual que ConvNet, con Dropout(p) despues de aplanar y antes de fc_out."""

    def __init__(self, dropout_p):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc_out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        output = self.pool(F.relu(self.conv1(x)))
        output = self.pool(F.relu(self.conv2(output)))
        output = output.view(output.size(0), -1)
        output = self.dropout(output)
        output = self.fc_out(output)
        return output


def train_classifier(model, loader, X_val, y_val, epochs, lr=1e-3):
    """Mismo patron del notebook de cancer de mama, adaptado a clasificacion multiclase
    (CrossEntropyLoss + logits crudos en vez de BCELoss + sigmoid)."""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_history = []
    val_loss_history = []

    print(f"{'Epoch':>5s} | {'Train Loss':>10s} | {'Val Loss':>10s}")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
        train_epoch_loss = running_loss / n_batches

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_epoch_loss = loss_fn(val_logits, y_val).item()

        train_loss_history.append(train_epoch_loss)
        val_loss_history.append(val_epoch_loss)
        print(f"{epoch + 1:5d} | {train_epoch_loss:10.4f} | {val_epoch_loss:10.4f}")

    return train_loss_history, val_loss_history


def evaluate_accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean().item()
    return accuracy, preds


def plot_loss_curves(train_loss, val_loss, title):
    epochs_range = range(1, len(train_loss) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, train_loss, marker="o", label="Train")
    plt.plot(epochs_range, val_loss, marker="o", label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (CrossEntropy)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
