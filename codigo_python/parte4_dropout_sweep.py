"""
Parte 4, barrido de dropout sobre la ConvNet, en Fashion MNIST.

Mismo split y semilla que el resto del laboratorio. Ojo con el
train_loss_final reportado, se mide con dropout activo, model.train(),
mientras que el val_loss_final se mide sin dropout, model.eval(). No es una
comparacion directa al 100%, hay que leerla con cuidado al analizar la
brecha entre train y val.

Ejecutar desde la carpeta que tiene el .env,
    python codigo_python/parte4_dropout_sweep.py
"""
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import (
    RANDOM_SEED,
    ConvNetDropout,
    load_split_scale,
    train_classifier,
    evaluate_accuracy,
)

DROPOUT_VALUES = [0.0, 0.2, 0.4, 0.6]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test = load_split_scale(
        "FMNIST_TRAIN_FILEPATH", "FMNIST_TEST_FILEPATH"
    )

    X_train_img = X_train.view(-1, 1, 28, 28)
    X_val_img = X_val.view(-1, 1, 28, 28)
    X_test_img = X_test.view(-1, 1, 28, 28)
    train_dataset = TensorDataset(X_train_img, y_train)

    results = []
    for p in DROPOUT_VALUES:
        print(f"\ndropout_p = {p}")
        model = ConvNetDropout(p)
        generator = torch.Generator().manual_seed(RANDOM_SEED)
        loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator
        )

        train_loss, val_loss = train_classifier(model, loader, X_val_img, y_val, epochs=args.epochs)
        acc, _ = evaluate_accuracy(model, X_test_img, y_test)

        results.append({
            "dropout_p": p,
            "Precisión de Prueba": acc,
            "train_loss_final": train_loss[-1],
            "val_loss_final": val_loss[-1],
        })

    results_df = pd.DataFrame(results)
    print("\nResumen del barrido de dropout:")
    print(results_df)


if __name__ == "__main__":
    main()
