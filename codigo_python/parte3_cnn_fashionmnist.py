"""
Parte 3, ConvNet en Fashion MNIST, entrenamiento, comparacion de 3 vias,
densa en MNIST, densa en Fashion MNIST, CNN en Fashion MNIST, y classification_report.

La comparacion de 3 vias necesita los resultados de las Partes 1 y 2. Se
imprimen los que produjo este script mas los ya obtenidos antes, para armarla
a mano en el informe, es mas simple que volver a entrenar todo aca.

Ejecutar desde la carpeta que tiene el .env,
    python codigo_python/parte3_cnn_fashionmnist.py
"""
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

from common import (
    RANDOM_SEED,
    ConvNet,
    FASHION_MNIST_CLASSES,
    load_split_scale,
    train_classifier,
    evaluate_accuracy,
    plot_loss_curves,
    count_parameters,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test = load_split_scale(
        "FMNIST_TRAIN_FILEPATH", "FMNIST_TEST_FILEPATH"
    )

    # Conv2d espera (batch, canales, alto, ancho), no el vector aplanado de 784
    X_train_img = X_train.view(-1, 1, 28, 28)
    X_val_img = X_val.view(-1, 1, 28, 28)
    X_test_img = X_test.view(-1, 1, 28, 28)

    train_dataset = TensorDataset(X_train_img, y_train)
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator
    )

    model = ConvNet()
    train_loss, val_loss = train_classifier(model, train_loader, X_val_img, y_val, epochs=args.epochs)
    plot_loss_curves(train_loss, val_loss, "ConvNet en Fashion MNIST, loss por epoca")

    test_acc, test_preds = evaluate_accuracy(model, X_test_img, y_test)
    print(f"Accuracy de prueba (Fashion-MNIST, CNN): {test_acc:.4f}")
    print(f"Cantidad de parametros (ConvNet): {count_parameters(model)}")

    print("\nclassification_report, CNN en Fashion MNIST")
    print(classification_report(y_test.numpy(), test_preds.numpy(), target_names=FASHION_MNIST_CLASSES))


if __name__ == "__main__":
    main()
