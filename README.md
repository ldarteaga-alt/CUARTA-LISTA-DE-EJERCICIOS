# Cuarta lista de ejercicios - Machine Learning

## Estructura

- `data/`: archivos CSV entregados para los ejercicios.
- `src/lista4/`: funciones reutilizables.
  - `data.py`: carga y validación de datos.
  - `models.py`: ajuste de modelos y predicciones.
  - `metrics.py`: riesgos y métricas.
- `notebooks/`: ejecución de cada ejercicio.
- `requirements.txt`: dependencias.
- `pyproject.toml`: configuración del paquete local.

## Instalación

Desde la carpeta raíz del proyecto:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Luego:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Ejecución

```bash
jupyter notebook
```

Abrir los notebooks de `notebooks/`, ejecutar todas las celdas y guardar cada notebook.
