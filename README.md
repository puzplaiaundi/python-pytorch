# Mini Proyecto – API de IA con Python, Flask y PyTorch

Este proyecto implementa una **API de Inteligencia Artificial** desarrollada con **Python**, **Flask** y **PyTorch**.

Permite exponer un modelo de IA a través de una API REST para que pueda ser consumido desde un cliente (por ejemplo, una aplicación JavaScript mediante `fetch`).

---

## 📋 Requisitos previos

Antes de comenzar, es necesario tener instalado:

- Python 3.9 o superior
- pip (incluido normalmente con Python)

Puedes comprobar la instalación con:

```python
python --version
pip --version
```

# Instrucciones de despliegue
## 1) Clonar el proyecto
```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_PROYECTO>
```
## 2) Crear un entorno virtual
```python
python -m venv venv
```
## 3) Activar el entorno virtual
El entorno virtual sirve para instalar las dependencias de forma aislada.  
Si no utilizasemos **(venv)** , las librerias se instalarian de forma global.
```python
venv\Scripts\activate
```
El resutlado debería ser:
```bash
(venv) PS <ruta del proyecto>:
```
## 4) Instalar dependencias/librerias
  Las dependencias necesarias para que todo funcione correctamente están guardadas en el archivo *"requirements.txt"*.  
  Este paso solo hay que hacerlo la primera vez.
  ```python
pip install -r requirements.txt
```
## 5) Arrancar el servidor
Para arrancar el serivdor basta con ejecutar el archivo run.py:
```python
Pyhon run.py
```
Esto arrancará el servidor en localhost, por defecto en el puerto 5000.
## 6) Ejecutar el cliente
Ahora que el servidor está en marcha ya probar el cliente html que se encuentra en \frontend\index.html


    
  
  
