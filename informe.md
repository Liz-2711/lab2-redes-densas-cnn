# Informe, Laboratorio 2, de redes densas a CNNs

Curso: Redes Neuronales y Aprendizaje Profundo, Q3 2026
Datasets: MNIST y Fashion MNIST
Config fija en todo el lab: random_state=42 en el split y en torch.manual_seed, pixeles escalados entre 255, CrossEntropyLoss, Adam con lr=1e-3, batch_size=128, epochs=10.

## Tabla comparativa, Parte 1 contra Parte 2

| Dataset | Precisión de Prueba (misma arquitectura) |
|---|---|
| MNIST (Parte 1) | 97.42% |
| Fashion-MNIST (Parte 2) | 87.16% |

Pregunta: si la precisión en Fashion MNIST es notablemente menor que en MNIST, qué dice eso y qué descarta.
Descarta que el modelo sea peor. Es la misma clase DenseNet, los mismos hiperparametros, mismo optimizador, misma semilla. Lo unico que cambio fue que representan las imagenes. Queda como explicacion la dificultad de la tarea en el espacio de representacion que usamos, al aplanar la imagen a 784 pixeles sin orden espacial una prenda es mas dificil de separar linealmente que un digito centrado y normalizado. La brecha de 10.26 puntos es evidencia de dificultad de tarea, no de que el modelo haya empeorado.

Pregunta: mirando las dos curvas de perdida, el modelo de Fashion MNIST llega a una perdida de entrenamiento tan baja como la de MNIST.
No. La perdida de entrenamiento de MNIST termina en 0.044 en la epoca 10, la de Fashion MNIST se queda en 0.287, casi 6.5 veces mas alta, sin bajar mucho mas. No es solo un tema de generalizacion entre train y val, el modelo ni logra ajustar bien su propio set de entrenamiento en Fashion MNIST. Una red densa sobre pixeles aplanados parece tener capacidad insuficiente para representar formas de ropa, mas alla de que le cueste generalizar.

## Tabla de 3 vias, Parte 3

| Modelo / Dataset | Precisión de Prueba | Cantidad de Parámetros |
|---|---|---|
| Densa en MNIST (Parte 1) | 97.42% | 101,770 |
| Densa en Fashion-MNIST (Parte 2) | 87.16% | 101,770 |
| CNN en Fashion-MNIST (Parte 3) | 89.01% | 20,490 |

Pregunta: cuanto de la brecha entre Parte 1 y Parte 2 logro recuperar la CNN.
La brecha original, arquitectura fija y tarea distinta, es de 10.26 puntos, 97.42% contra 87.16%. La CNN recupera 1.85 puntos, 89.01% contra 87.16%, mas o menos el 18% de esa brecha. La cierra en parte, no del todo, quedan 8.41 puntos de diferencia frente a MNIST.

Pregunta: si la CNN tiene menos parametros pero logra mas precision, que dice eso sobre supuestos estructurales correctos contra mas capacidad bruta.
La CNN tiene como 5 veces menos parametros, 20,490 contra 101,770, y aun asi supera a la red densa en Fashion MNIST. Mas parametros en la red densa no se tradujo en mejor desempeno, menos parametros bien organizados para datos de imagen, con conectividad local y peso compartido, si. Esto pinta que el sesgo inductivo correcto pesa mas que la capacidad bruta cuando la arquitectura calza con la estructura real de los datos.

## Reporte por clase, CNN en Fashion MNIST

| Clase | Precision | Recall | F1-score |
|---|---|---|---|
| Camiseta/Top | 0.89 | 0.77 | 0.82 |
| Pantalón | 0.99 | 0.97 | 0.98 |
| Suéter | 0.75 | 0.90 | 0.82 |
| Vestido | 0.84 | 0.93 | 0.89 |
| Abrigo | 0.87 | 0.74 | 0.80 |
| Sandalia | 0.99 | 0.97 | 0.98 |
| Camisa | 0.71 | 0.70 | 0.71 |
| Tenis | 0.94 | 0.98 | 0.96 |
| Bolso | 0.98 | 0.97 | 0.98 |
| Botín | 0.98 | 0.96 | 0.97 |

Accuracy global, 89.01%

Pregunta: que clases confunde mas el modelo.
Camisa con F1 0.71, Suéter 0.82, Abrigo 0.80 y Camiseta/Top 0.82 quedan notablemente mas bajas que Pantalón 0.98, Sandalia 0.98, Bolso 0.98 o Botín 0.97. Coincide con lo esperado, Camisa, Camiseta/Top, Suéter y Abrigo son todas prendas de torso con siluetas parecidas a 28x28 pixeles, mientras un pantalon, una sandalia o un bolso tienen formas bastante mas distintivas.

Pregunta: si el accuracy general es 89% pero una clase tiene F1 bien por debajo, que esconde el accuracy general.
El 89.01% global es un promedio, esconde que Camisa falla a una tasa bastante mayor que el resto, F1 0.71, 18 puntos por debajo del promedio. Un modelo puede verse suficientemente bueno en el agregado mientras falla de forma sistematica en una clase puntual. Para decidir si un modelo sirve para produccion hay que mirar el desempeno por clase, no solo el numero global.

Pregunta: ayudaria el data augmentation, rotacion o traslacion, a distinguir Camisa de Camiseta/Top.
Probablemente no mucho. Las imagenes de Fashion MNIST ya vienen centradas y normalizadas en escala y posicion, el problema no es que el modelo vea la misma prenda desplazada o rotada entre train y test. La confusion entre Camisa y Camiseta/Top es de similitud real de forma y textura a resolucion de 28x28, donde esos detalles finos casi no tienen pixeles para representarse. Rotacion y traslacion atacan invariancia posicional, no resolucion de detalle fino, entonces no es la herramienta indicada aca, ayudaria mas subir resolucion o capacidad del modelo, o dar mas señal de entrenamiento en esas clases confundibles.

## Barrido de dropout, Parte 4

| dropout_p | Precisión de Prueba | train_loss_final | val_loss_final |
|---|---|---|---|
| 0.0 | 89.09% | 0.2567 | 0.2787 |
| 0.2 | 89.90% | 0.2735 | 0.2560 |
| 0.4 | 89.50% | 0.3098 | 0.2758 |
| 0.6 | 89.03% | 0.3450 | 0.2879 |

Pregunta: el dropout cambia de forma significativa la precision de prueba, o esta dentro del margen de ruido.
La diferencia entre el mejor, 89.90% con p=0.2, y el peor, 89.03% con p=0.6, es menos de un punto porcentual, y no hay tendencia monotona, el maximo cae en un valor intermedio, no en los extremos. Con una sola corrida por valor de p esto pinta mas a ruido de inicializacion y orden de batches que a un efecto real y consistente del dropout en este dataset.

Pregunta: se reduce la brecha entre train y val conforme sube el dropout.
Ojo con esta columna, train_loss_final se mide con dropout activo, model.train(), y val_loss_final sin dropout, model.eval(), no es una comparacion directa. A medida que sube p el train loss reportado se infla porque dropout apaga neuronas tambien al medirlo, y el val loss queda por debajo del train loss. La brecha cambia de signo pero eso es un artefacto de la asimetria train eval de dropout, no evidencia real de menos sobreajuste. Para comparar de forma justa habria que medir tambien el train loss en modo eval, con dropout apagado.

Pregunta: Fashion MNIST tiene 60,000 imagenes de entrenamiento contra unas 455 del dataset de cancer de mama, cambia esto cuanto esperariamos que importe el dropout.
Si. El dropout regulariza mas cuando el modelo tiene capacidad alta frente a pocos datos de entrenamiento, el caso de cancer de mama con unas 291 muestras de train despues del segundo split. Aca la ConvNet tiene solo unos 20,490 parametros contra 48,000 imagenes de entrenamiento, hay mucho menos margen para memorizar y por lo tanto menos sobreajuste que corregir de entrada. El resultado plano y ruidoso del barrido calza con esa logica.

Pregunta: dada la brecha entre Parte 2 y Parte 3, el dropout cierra mas de esa brecha o sigue siendo la arquitectura, densa a CNN, la que explica la mayor parte de la mejora.
El cambio de arquitectura entre Parte 2 y Parte 3 aporto 1.85 puntos, de 87.16% a 89.01%. El mejor resultado del barrido de dropout, 89.90% con p=0.2, suma apenas 0.89 puntos mas sobre la CNN sin dropout, y esa mejora extra ya vimos que cae dentro del margen de ruido de una sola corrida. La arquitectura sigue siendo, con diferencia, el factor que explica la mayor parte de la mejora medida en este experimento, el dropout no aporta una ganancia clara adicional en este caso.
