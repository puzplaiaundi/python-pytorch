# Entrenamiento o personalización de la red neuronal
El archivo que ha entrenado la red neuronal generada por Pythorch es el **dataset.csv**.  
Si se quiere, se puede reentrenar la red neuronal de una forma sencilla siguiendo estos pasos.

# 1) Editar el archivo dataset.csv
En este archivo se definen diferentes textos de entrada posibles y las clases a las que pertenecen.  
Se pueden cambiar las clases, incluso se puede cambiar la cantidad de clases sin problema.  
Es **importante** tratar de tener el mismo número de frases por clase para que los pesos del entrenamiento sean equilibrados.

# 2) Entrenar la Red Neuronal
Una vez editado y guardado el dataset.csv, basta con ejecutar desde la carpeta \ml el siguiente comando:
```python
pyhton train_nn.py
```
# 3) Arrancar el servidor.
Basta con volver a arrancar el servidor desde la carpeta raiz para probar el nuevo entrenamiento.
```python
python run.py
```


