"""Fonctions de chargement, contrôle et nettoyage des jeux de données."""

from pathlib import Path

import pandas as pd


def load_potability(path: str | Path) -> pd.DataFrame:
    """Charger le jeu CSV de potabilité et normaliser les noms de colonnes."""
    dataframe = pd.read_csv(path)
    dataframe.columns = dataframe.columns.astype(str).str.strip()
    return dataframe


def load_water_quality(path: str | Path, sheet_name: str) -> pd.DataFrame:
    """Charger une feuille du classeur physico-chimique."""
    dataframe = pd.read_excel(path, sheet_name=sheet_name)
    dataframe.columns = dataframe.columns.astype(str).str.strip()
    return dataframe


def basic_cleaning(
    dataframe: pd.DataFrame,
    target: str | None = None,
) -> pd.DataFrame:
    """Effectuer un nettoyage minimal sans imputation globale.

    Les doublons sont supprimés et les colonnes sont converties en numérique.
    Si une cible est fournie, seules les lignes dont la cible est manquante sont
    retirées. Les valeurs manquantes des prédicteurs sont volontairement
    conservées afin que l'imputation soit apprise dans le pipeline ML.
    """
    output = dataframe.copy()
    output.columns = output.columns.astype(str).str.strip()
    output = output.drop_duplicates().reset_index(drop=True)

    for column in output.columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    if target and target in output.columns:
        output = output.loc[output[target].notna()].reset_index(drop=True)

    return output


def data_quality_report(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Retourner un tableau synthétique de qualité des données."""
    return pd.DataFrame(
        {
            "dtype": dataframe.dtypes.astype(str),
            "missing_n": dataframe.isna().sum(),
            "missing_pct": (100 * dataframe.isna().mean()).round(2),
            "n_unique": dataframe.nunique(dropna=True),
        }
    )


def save_processed(dataframe: pd.DataFrame, path: str | Path) -> None:
    """Sauvegarder un jeu de données préparé en créant le dossier si besoin."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
