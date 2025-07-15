# entrenamiento.py
from keras.datasets import mnist
from sklearn.naive_bayes import (
    GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, CategoricalNB
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

# Cargar y preparar datos
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

modelos = {
    "GaussianNB": GaussianNB(),
    "BernoulliNB": BernoulliNB(),
    "MultinomialNB": MultinomialNB(),
    "ComplementNB": ComplementNB(),
    "CategoricalNB": CategoricalNB()
}

resultados = []

for nombre, modelo in modelos.items():
    try:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        resultados.append((nombre, acc, f1))
        joblib.dump(modelo, f"modelo_{nombre.lower()}.pkl")
        print(f"{nombre} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    except Exception as e:
        print(f"{nombre} falló: {e}")

# Mostrar los dos mejores modelos
resultados.sort(key=lambda x: x[1], reverse=True)
print("Top 2 modelos:")
for r in resultados[:2]:
    print(r)
