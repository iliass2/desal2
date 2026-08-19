"""Chargement, contrôle et nettoyage des jeux de données."""
from pathlib import Path
import pandas as pd

def load_potability(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    return df

def load_water_quality(path: str | Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = df.columns.astype(str).str.strip()
    return df

def basic_cleaning(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip()
    out = out.drop_duplicates().reset_index(drop=True)
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if target and target in out.columns:
        out = out.loc[out[target].notna()].reset_index(drop=True)
    return out

def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_n": df.isna().sum(),
        "missing_pct": (100 * df.isna().mean()).round(2),
        "n_unique": df.nunique(dropna=True),
    })

def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
