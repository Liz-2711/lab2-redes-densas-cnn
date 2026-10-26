"""
Parte 4b, el mismo barrido de dropout de la Parte 4 pero repetido con varias semillas.

En la Parte 4 hay una sola corrida por valor de p, asi que no se puede separar el
efecto del dropout del ruido de inicializacion y orden de batches. Aca cada p se
entrena con N semillas. El split train/val/test queda fijo (random_state=42, igual
que el resto del lab); la semilla solo cambia la inicializacion y el orden de batches.

Con la misma semilla, la inicializacion es identica para todos los p (Dropout no
tiene pesos), asi que la comparacion contra p=0 es pareada: misma semilla, solo
cambia p.

Ejecutar desde la carpeta que tiene el .env,
    python codigo_python/parte4b_dropout_seeds.py --seeds 5
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

from common import ConvNetDropout, load_split_scale, train_classifier, evaluate_accuracy

DROPOUT_VALUES = [0.0, 0.2, 0.4, 0.6]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", default=os.path.join("codigo_python", "resultados_dropout_seeds.csv"))
    args = parser.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test = load_split_scale(
        "FMNIST_TRAIN_FILEPATH", "FMNIST_TEST_FILEPATH"
    )
    X_train_img = X_train.view(-1, 1, 28, 28)
    X_val_img = X_val.view(-1, 1, 28, 28)
    X_test_img = X_test.view(-1, 1, 28, 28)
    train_dataset = TensorDataset(X_train_img, y_train)

    results = []
    for seed in range(args.seeds):
        for p in DROPOUT_VALUES:
            t0 = time.perf_counter()
            torch.manual_seed(seed)  # misma inicializacion para todos los p con esta semilla
            model = ConvNetDropout(p)
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator
            )
            _, val_loss = train_classifier(model, loader, X_val_img, y_val, epochs=args.epochs)
            acc, _ = evaluate_accuracy(model, X_test_img, y_test)
            results.append({"seed": seed, "dropout_p": p, "test_acc": acc * 100,
                            "val_loss_final": val_loss[-1]})
            print(f"seed={seed} p={p}: test_acc={acc * 100:.2f}% ({time.perf_counter() - t0:.0f} s)",
                  flush=True)
            pd.DataFrame(results).to_csv(args.out, index=False)  # se guarda tras cada corrida

    df = pd.DataFrame(results)

    print("\nPrecision de prueba por dropout_p (media y desviacion estandar entre semillas):")
    resumen = df.groupby("dropout_p")["test_acc"].agg(["mean", "std", "min", "max"])
    print(resumen.round(2))

    # Comparacion pareada contra p=0: diferencia por semilla, IC 95% con t de Student
    base = df[df.dropout_p == 0.0].set_index("seed")["test_acc"]
    print("\nDiferencia contra p=0 (pareada por semilla):")
    for p in DROPOUT_VALUES[1:]:
        diff = df[df.dropout_p == p].set_index("seed")["test_acc"] - base
        n = len(diff)
        half = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
        p_value = stats.ttest_rel(df[df.dropout_p == p].set_index("seed")["test_acc"], base).pvalue
        print(f"  p={p}: {diff.mean():+.2f} pts, IC 95% [{diff.mean() - half:+.2f}, {diff.mean() + half:+.2f}], "
              f"t pareada p-value={p_value:.3f}")

    spread = resumen["mean"].max() - resumen["mean"].min()
    print(f"\nSpread entre las medias de p: {spread:.2f} pts; "
          f"desviacion entre semillas (promedio sobre p): {resumen['std'].mean():.2f} pts")
    print(f"Resultados por corrida en {args.out}")


if __name__ == "__main__":
    main()
