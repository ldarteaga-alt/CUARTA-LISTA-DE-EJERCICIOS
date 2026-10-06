from pathlib import Path
import pandas as pd

def load_csv(filename, data_dir=None):
    """Carga un archivo CSV ubicado en la carpeta data/."""

    if data_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        data_dir = project_root / "data"

    path = Path(data_dir) / filename

    return pd.read_csv(path)

def validate_columns(df, columns):
    """Verifica que el DataFrame tenga las columnas solicitadas."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan estas columnas: {missing}")


def split_xy(df, feature_cols, target_col):
    """Separa un DataFrame en matriz X y vector y."""
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    return X, y
