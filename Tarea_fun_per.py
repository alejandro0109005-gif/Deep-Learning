# %%
# Importamos las librerias necesarias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# %% [markdown]
# Ahora preparemos todos los datos para después entrenar nuestros modelos MLP

# %%
# Leemos el dataset 
df = pd.read_csv("Classified_Data.csv")
df.head()

# %%
# Eliminamos la primer columna (indices)
df = df.iloc[:, 1:]
# Visualizamos nuestro df
df.head()

# %%
# Seleccionamos los inputs y el output 
X = df.drop("TARGET CLASS", axis = 1)
y = df["TARGET CLASS"]

# %%
# Dividimos nuestros datos en entrenamiento y prueba 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# %%
# Estandarizamos los inputs (restamos la media y dividimos entre su desviación estándar)
scaler = StandardScaler()
X_train_estand = scaler.fit_transform(X_train)
X_test_estand = scaler.transform(X_test)

# %% [markdown]
# Entrenaremos 3 modelos con las siguientes características: 
# 
# - Método de resolución lbfgs
# 
# - Semilla 0
#   
# - Función de activación logística.
# 
# Adicionalmente, el primer modelo tiene dos capas ocultas con 10 nodos cada una, el segundo tiene una capa oculta y 10 nodos y el último tiene una sola capa oculta con un nodo: 

# %% [markdown]
# ![imagen.png](MLP_imagen.png)

# %%
# Definimos el primer modelo que utilizaremos
modelo_1 = MLPClassifier(hidden_layer_sizes = (10, 10), solver = "lbfgs", activation = "logistic", random_state = 0, max_iter = 1000)
modelo_1.fit(X_train_estand, y_train)   # Entrenamos el modelo


# %%
# Ahora obtenemos el accuracy para entrenamiento y prueba 
acc_train = accuracy_score(y_train, modelo_1.predict(X_train_estand))
acc_test = accuracy_score(y_test, modelo_1.predict(X_test_estand))

print(f"El accuracy para este primer modelo en entrenamiento es: {acc_train: .4f} y en prueba es: {acc_test: .4f}")

# %%
# Definimos el segundo modelo que utilizaremos
modelo_2 = MLPClassifier(hidden_layer_sizes=(10,), solver = "lbfgs", activation = "logistic", random_state = 0, max_iter = 1000)
modelo_2.fit(X_train_estand, y_train)   # Entrenamos el modelo


# %%
# Ahora obtenemos el accuracy para entrenamiento y prueba 
acc_train_2 = accuracy_score(y_train, modelo_2.predict(X_train_estand))
acc_test_2 = accuracy_score(y_test, modelo_2.predict(X_test_estand))

print(f"El accuracy para este segundo modelo en entrenamiento es: {acc_train_2: .4f} y en prueba es: {acc_test_2: .4f}")

# %%
# Definimos el tercer modelo que utilizaremos
modelo_3 = MLPClassifier(hidden_layer_sizes=(1,), solver = "lbfgs", activation = "logistic", random_state = 0, max_iter = 1000)
modelo_3.fit(X_train_estand, y_train)   # Entrenamos el modelo


# %%
# Ahora obtenemos el accuracy para entrenamiento y prueba 
acc_train_3 = accuracy_score(y_train, modelo_3.predict(X_train_estand))
acc_test_3 = accuracy_score(y_test, modelo_3.predict(X_test_estand))

print(f"El accuracy para este tercer modelo en entrenamiento es: {acc_train_3: .4f} y en prueba es: {acc_test_3: .4f}")

# %% [markdown]
# Grafiquemos en un gráfico nuestros resultados

# %%
# Definimos los nombres de los modelos y agrupamos las variables de accuracy
modelos = ["Modelo 1\n(10, 10)", "Modelo 2\n(10,)", "Modelo 3\n(1,)"]
lista_acc_train = [acc_train, acc_train_2, acc_train_3]
lista_acc_test = [acc_test, acc_test_2, acc_test_3]

# Definimos la posición de las barras
x = np.arange(len(modelos))
ancho = 0.40

# Creamos la figura
plt.figure(figsize=(8, 5))
plt.bar(x - ancho/2, lista_acc_train, width = ancho, label = "Entrenamiento", color = "green")
plt.bar(x + ancho/2, lista_acc_test, width = ancho, label = "Prueba", color = "red")

# Personalizamos el gráfico
plt.title("Comparación de Accuracy por Modelo", fontsize = 12)
plt.xlabel("Estructura de la Red")
plt.ylabel("Accuracy")
plt.xticks(x, modelos)
plt.ylim(0.85, 1.02)
plt.legend()
plt.grid(axis = "y", linestyle = "--", alpha = 0.5)

plt.show()

# %% [markdown]
# Conclusión
# 
# Con estos resultados podemos concluir que elegiriamos el modelo 3 por las siguientes razones:
# 
# 1. Es el modelo mas sencillo, puesto que unicamente tiene una capa oculta y un nodo.
# 2. A pesar de que fue el que tuvo el accuracy más bajo en entrenamiento, en prueba fue el que tuvo un resultado mayor, dejandonos ver que no memorizó los datos que le dimos (puede que no sufra de overfitting) y para futuras predicciones con datos nuevos funcionará mejor.
# 
# 


