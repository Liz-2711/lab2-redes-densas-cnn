"""
Parte 2, la misma DenseNet, reentrenada desde cero en Fashion MNIST.

No se reutilizan los pesos de la Parte 1, es una instancia nueva de DenseNet,
mismos hiperparametros y semilla, solo cambia el dataset. Asi cualquier
diferencia de accuracy contra la Parte 1 se atribuye a la dificultad de la
tarea, no a un cambio en el modelo.

Ejecutar desde la carpeta que tiene el .env,
    python codigo_python/parte2_densenet_fashionmnist.py
"""
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from common import (
    RANDOM_SEED,
    DenseNet,
    load_split_scale,
    train_classifier,
    evaluate_accuracy,
    plot_loss_curves,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test = load_split_scale(
        "FMNIST_TRAIN_FILEPATH", "FMNIST_TEST_FILEPATH"
    )
    print(f"X_train: {tuple(X_train.shape)}  X_val: {tuple(X_val.shape)}  X_test: {tuple(X_test.shape)}")

    train_dataset = TensorDataset(X_train, y_train)
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator
    )

    model = DenseNet()  # pesos nuevos, no los de la Parte 1
    train_loss, val_loss = train_classifier(model, train_loader, X_val, y_val, epochs=args.epochs)
    plot_loss_curves(train_loss, val_loss, "DenseNet en Fashion MNIST, loss por epoca")

    test_acc, _ = evaluate_accuracy(model, X_test, y_test)
    print(f"Accuracy de prueba (Fashion-MNIST, DenseNet): {test_acc:.4f}")


if __name__ == "__main__":
    main()
